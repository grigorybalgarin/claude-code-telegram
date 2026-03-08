"""Staged deploy pipeline with structured results and rollback.

Breaks deploy into explicit stages: update code, preflight, compile,
restart, health check, log inspection. Each stage produces a result.
If a known-safe rollback is possible, it's attempted automatically.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from .ops_model import DeployGateReport, DeploySafetyGate, GateResult
from .shell_executor import ShellExecutor

logger = structlog.get_logger()


class DeployStage(Enum):
    """Stages of a deploy pipeline."""

    UPDATE_CODE = "update_code"
    PREFLIGHT = "preflight"
    COMPILE = "compile"
    RESTART = "restart"
    HEALTH_CHECK = "health_check"
    LOG_INSPECT = "log_inspect"


_STAGE_LABELS = {
    DeployStage.UPDATE_CODE: "Обновление кода",
    DeployStage.PREFLIGHT: "Предварительная проверка",
    DeployStage.COMPILE: "Компиляция",
    DeployStage.RESTART: "Перезапуск сервиса",
    DeployStage.HEALTH_CHECK: "Проверка здоровья",
    DeployStage.LOG_INSPECT: "Проверка логов",
}


@dataclass(frozen=True)
class DeployStageResult:
    """Result of a single deploy stage."""

    stage: DeployStage
    success: bool
    output: str
    duration_ms: int
    command: str = ""
    skipped: bool = False


@dataclass
class DeployResult:
    """Complete deploy pipeline result."""

    correlation_id: str
    workspace_path: str
    stages: List[DeployStageResult] = field(default_factory=list)
    overall_success: bool = False
    failed_stage: Optional[DeployStage] = None
    rollback_performed: bool = False
    rollback_success: bool = False
    rollback_output: str = ""
    pre_deploy_commit: str = ""
    post_deploy_commit: str = ""
    total_duration_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON storage."""
        return {
            "correlation_id": self.correlation_id,
            "workspace_path": self.workspace_path,
            "overall_success": self.overall_success,
            "failed_stage": self.failed_stage.value if self.failed_stage else None,
            "rollback_performed": self.rollback_performed,
            "rollback_success": self.rollback_success,
            "pre_deploy_commit": self.pre_deploy_commit,
            "post_deploy_commit": self.post_deploy_commit,
            "total_duration_ms": self.total_duration_ms,
            "stages": [
                {
                    "stage": s.stage.value,
                    "success": s.success,
                    "duration_ms": s.duration_ms,
                    "skipped": s.skipped,
                }
                for s in self.stages
            ],
        }

    def format_summary(self) -> str:
        """Human-readable deploy summary."""
        if self.overall_success:
            lines = [
                "<b>Deploy: успешно</b>",
                "",
            ]
            if self.post_deploy_commit:
                lines.append(f"Коммит: <code>{self.post_deploy_commit[:8]}</code>")
            elapsed = self.total_duration_ms / 1000
            lines.append(f"Время: {elapsed:.1f}с")

            passed = sum(1 for s in self.stages if s.success and not s.skipped)
            lines.append(f"Стадии: {passed}/{len(self.stages)}")
            return "\n".join(lines)

        lines = [
            "<b>Deploy: не удалось</b>",
            "",
        ]
        if self.failed_stage:
            lines.append(
                f"<b>Сбой на стадии:</b> {_STAGE_LABELS.get(self.failed_stage, self.failed_stage.value)}"
            )

        # Show stage results
        for sr in self.stages:
            icon = "✅" if sr.success else ("⏭" if sr.skipped else "❌")
            label = _STAGE_LABELS.get(sr.stage, sr.stage.value)
            lines.append(f"  {icon} {label}")

        if self.rollback_performed:
            rb_status = "успешно" if self.rollback_success else "не удалось"
            lines.append(f"\n<b>Откат:</b> {rb_status}")
            if self.pre_deploy_commit:
                lines.append(f"Откат к: <code>{self.pre_deploy_commit[:8]}</code>")

        # Show failing stage output
        if self.failed_stage:
            for sr in self.stages:
                if sr.stage == self.failed_stage and sr.output:
                    output = sr.output[-500:] if len(sr.output) > 500 else sr.output
                    lines.append(f"\n<b>Вывод ошибки:</b>\n<pre>{output}</pre>")

        return "\n".join(lines)


@dataclass
class DeployProfile:
    """Deploy configuration for a workspace."""

    workspace_root: Path
    update_command: Optional[str] = None
    preflight_command: Optional[str] = None
    compile_command: Optional[str] = None
    restart_command: Optional[str] = None
    health_command: Optional[str] = None
    logs_command: Optional[str] = None
    rollback_safe: bool = False

    @classmethod
    def from_workspace_profile(
        cls,
        profile: Any,
        boundary_root: Path,
    ) -> "DeployProfile":
        """Build deploy profile from workspace profile."""
        commands = getattr(profile, "commands", {}) or {}
        root = profile.root_path

        # Detect service commands
        restart_cmd = commands.get("deploy")
        health_cmd = commands.get("health")
        logs_cmd = None

        services = list(getattr(profile, "services", ()))
        if services:
            svc = services[0]
            if not restart_cmd:
                restart_cmd = getattr(svc, "restart_command", None)
            if not health_cmd:
                health_cmd = getattr(svc, "health_command", None) or getattr(
                    svc, "status_command", None
                )
            if not logs_cmd:
                logs_cmd = getattr(svc, "logs_command", None)

        has_git = getattr(profile, "has_git_repo", False)

        return cls(
            workspace_root=root,
            update_command="git pull --ff-only" if has_git else None,
            preflight_command=commands.get("lint") or commands.get("typecheck"),
            compile_command=commands.get("build"),
            restart_command=restart_cmd,
            health_command=health_cmd,
            logs_command=logs_cmd,
            rollback_safe=has_git,
        )


class DeployPipeline:
    """Execute staged deploy with structured results and safety gates."""

    def __init__(self, shell: ShellExecutor):
        self.shell = shell

    async def check_safety_gates(
        self,
        deploy_profile: DeployProfile,
    ) -> DeployGateReport:
        """Run pre-deploy safety checks. Call before execute()."""
        report = DeployGateReport()
        root = deploy_profile.workspace_root

        # Gate 1: Clean worktree (hard gate for git repos)
        if deploy_profile.update_command and "git" in deploy_profile.update_command:
            r = await self.shell.execute(
                workspace_root=root,
                command="git status --porcelain --untracked-files=no",
                timeout_seconds=10,
            )
            clean = r.success and not r.stdout_text.strip()
            report.results.append(
                GateResult(
                    gate=DeploySafetyGate(
                        name="clean_worktree",
                        check_type="clean_worktree",
                        hard=True,
                        description="Рабочая директория должна быть чистой",
                    ),
                    passed=clean,
                    message="" if clean else "Есть незакоммиченные изменения",
                )
            )

        # Gate 2: Required commands present (hard gate)
        has_restart = bool(deploy_profile.restart_command)
        report.results.append(
            GateResult(
                gate=DeploySafetyGate(
                    name="restart_command",
                    check_type="profile_complete",
                    hard=True,
                    description="Команда перезапуска должна быть задана",
                ),
                passed=has_restart,
                message="" if has_restart else "Не задана команда restart/deploy",
            )
        )

        # Gate 3: Health command present (soft gate)
        has_health = bool(deploy_profile.health_command)
        report.results.append(
            GateResult(
                gate=DeploySafetyGate(
                    name="health_command",
                    check_type="profile_complete",
                    hard=False,
                    description="Команда health check желательна",
                ),
                passed=has_health,
                message=(
                    "" if has_health else "Нет health check — deploy без верификации"
                ),
            )
        )

        # Gate 4: Service currently healthy (soft gate)
        if deploy_profile.health_command:
            r = await self.shell.execute(
                workspace_root=root,
                command=deploy_profile.health_command,
                timeout_seconds=15,
            )
            report.results.append(
                GateResult(
                    gate=DeploySafetyGate(
                        name="service_healthy",
                        check_type="service_healthy",
                        hard=False,
                        description="Сервис должен быть здоров перед deploy",
                    ),
                    passed=r.success,
                    message="" if r.success else "Сервис не здоров перед deploy",
                )
            )

        return report

    async def execute(
        self,
        deploy_profile: DeployProfile,
        on_stage: Optional[Any] = None,
    ) -> DeployResult:
        """Run the full deploy pipeline.

        Args:
            deploy_profile: Deploy configuration.
            on_stage: Optional async callback(stage, label) for progress updates.

        Returns DeployResult with all stage results.
        """
        result = DeployResult(
            correlation_id=str(uuid.uuid4())[:8],
            workspace_path=str(deploy_profile.workspace_root),
        )
        start_time = time.time()
        root = deploy_profile.workspace_root

        # Capture pre-deploy commit for rollback
        pre_commit = await self._get_current_commit(root)
        result.pre_deploy_commit = pre_commit

        stages = [
            (DeployStage.UPDATE_CODE, deploy_profile.update_command),
            (DeployStage.PREFLIGHT, deploy_profile.preflight_command),
            (DeployStage.COMPILE, deploy_profile.compile_command),
            (DeployStage.RESTART, deploy_profile.restart_command),
            (DeployStage.HEALTH_CHECK, deploy_profile.health_command),
            (DeployStage.LOG_INSPECT, deploy_profile.logs_command),
        ]

        for stage, command in stages:
            if not command:
                result.stages.append(
                    DeployStageResult(
                        stage=stage,
                        success=True,
                        output="",
                        duration_ms=0,
                        skipped=True,
                    )
                )
                continue

            if on_stage:
                label = _STAGE_LABELS.get(stage, stage.value)
                await on_stage(stage, label)

            stage_start = time.time()
            shell_result = await self.shell.execute(
                workspace_root=root,
                command=command,
                timeout_seconds=180,
            )
            stage_ms = int((time.time() - stage_start) * 1000)

            output = shell_result.stderr_text or shell_result.stdout_text or ""
            if shell_result.error:
                output = shell_result.error

            stage_result = DeployStageResult(
                stage=stage,
                success=shell_result.success,
                output=output,
                duration_ms=stage_ms,
                command=command,
            )
            result.stages.append(stage_result)

            if not shell_result.success:
                result.failed_stage = stage

                # Rollback if safe and failure is after code update
                if (
                    deploy_profile.rollback_safe
                    and pre_commit
                    and stage
                    in {
                        DeployStage.PREFLIGHT,
                        DeployStage.COMPILE,
                        DeployStage.RESTART,
                        DeployStage.HEALTH_CHECK,
                    }
                ):
                    await self._rollback(result, root, pre_commit)

                break

        else:
            result.overall_success = True

        # Capture post-deploy commit
        post_commit = await self._get_current_commit(root)
        result.post_deploy_commit = post_commit
        result.total_duration_ms = int((time.time() - start_time) * 1000)

        return result

    async def _get_current_commit(self, root: Path) -> str:
        """Get current git HEAD short hash."""
        r = await self.shell.execute(
            workspace_root=root,
            command="git rev-parse --short HEAD",
            timeout_seconds=10,
        )
        return r.stdout_text.strip() if r.success else ""

    async def _rollback(
        self,
        result: DeployResult,
        root: Path,
        target_commit: str,
    ) -> None:
        """Attempt git rollback to pre-deploy commit."""
        result.rollback_performed = True
        r = await self.shell.execute(
            workspace_root=root,
            command=f"git checkout {target_commit} -- . && git checkout {target_commit}",
            timeout_seconds=30,
        )
        result.rollback_success = r.success
        result.rollback_output = r.stderr_text or r.stdout_text or ""
        if r.error:
            result.rollback_output = r.error

        logger.info(
            "Deploy rollback attempted",
            target=target_commit,
            success=r.success,
            workspace=str(root),
        )
