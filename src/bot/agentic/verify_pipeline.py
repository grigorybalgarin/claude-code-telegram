"""Deterministic verification pipeline for workspace health checks."""

from pathlib import Path
from typing import Any, Callable, Coroutine, List, Optional

from ..utils.html_format import escape_html
from .context import ShellActionResult, VerifyReport, VerifyStep
from .shell_executor import ShellExecutor


class VerifyPipeline:
    """Build and execute verification steps for a workspace profile.

    Receives a ShellExecutor and operates on raw data — no Telegram
    dependencies. The orchestrator handles all UI concerns.
    """

    def __init__(self, shell: ShellExecutor):
        self.shell = shell

    @staticmethod
    def build_steps(profile: Any) -> List[VerifyStep]:
        """Build the deterministic verification sequence for the workspace."""
        steps: List[VerifyStep] = []
        seen_commands: set[str] = set()

        health_command = profile.commands.get("health")
        if health_command:
            steps.append(VerifyStep(label="health", command=health_command))
            seen_commands.add(health_command)
        else:
            for service in getattr(profile, "services", ()):
                service_command = service.health_command or service.status_command
                if not service_command or service_command in seen_commands:
                    continue
                label = (
                    f"{service.display_name} проверка"
                    if service.health_command
                    else f"{service.display_name} статус"
                )
                steps.append(
                    VerifyStep(
                        label=label,
                        command=service_command,
                        logs_command=service.logs_command,
                    )
                )
                seen_commands.add(service_command)

        label_map = {
            "lint": "линт",
            "typecheck": "typecheck",
            "test": "тесты",
            "build": "сборка",
        }
        for key, label in label_map.items():
            command = profile.commands.get(key)
            if command and command not in seen_commands:
                steps.append(VerifyStep(label=label, command=command))
                seen_commands.add(command)

        return steps

    async def execute(
        self,
        profile: Any,
        on_step: Optional[Callable[[int, int, VerifyStep], Coroutine]] = None,
    ) -> VerifyReport:
        """Run verification steps sequentially, stopping on first failure.

        Args:
            profile: The workspace ProjectProfile.
            on_step: Optional async callback(index, total, step) for progress updates.

        Returns:
            VerifyReport with structured results.
        """
        steps = self.build_steps(profile)
        results: List[tuple[VerifyStep, ShellActionResult]] = []
        failed_step: Optional[VerifyStep] = None
        logs_result: Optional[ShellActionResult] = None

        for index, step in enumerate(steps, start=1):
            if on_step is not None:
                await on_step(index, len(steps), step)

            result = await self.shell.execute(
                workspace_root=profile.root_path,
                command=step.command,
                timeout_seconds=180,
            )
            results.append((step, result))
            if not result.success:
                failed_step = step
                if step.logs_command:
                    logs_result = await self.shell.execute(
                        workspace_root=profile.root_path,
                        command=step.logs_command,
                        timeout_seconds=30,
                    )
                break

        return VerifyReport(
            results=results,
            failed_step=failed_step,
            logs_result=logs_result,
        )

    def format_report(
        self,
        profile: Any,
        boundary_root: Path,
        report: VerifyReport,
    ) -> str:
        """Render a verification report as Telegram-friendly HTML."""
        try:
            relative = profile.root_path.resolve().relative_to(boundary_root)
            rel_path = "/" if str(relative) == "." else relative.as_posix()
        except ValueError:
            rel_path = str(profile.root_path)

        lines = [
            "✅ <b>Проверка завершена</b>"
            if report.success
            else "❌ <b>Проверка не пройдена</b>",
            "",
            f"Проект: <code>{escape_html(rel_path)}</code>",
        ]
        if report.failed_step:
            lines.append(
                f"Ошибка на шаге: <code>{escape_html(report.failed_step.label)}</code>"
            )
        lines.extend(["", "<b>Шаги</b>"])
        for step, result in report.results:
            state = "ok" if result.success else "ошибка"
            if result.timed_out:
                state = "таймаут"
            lines.append(
                f"• <code>{escape_html(step.label)}</code>: <code>{escape_html(state)}</code> "
                f"(exit <code>{result.returncode}</code>)"
            )
            summary = self.shell.summarize(result)
            if summary and summary != "нет вывода":
                lines.append(f"  <code>{escape_html(summary)}</code>")

        if report.results:
            last_step, last_result = report.results[-1]
            if not last_result.success and (
                last_result.stdout_text or last_result.stderr_text or last_result.error
            ):
                detail_text = (
                    last_result.stdout_text
                    or last_result.stderr_text
                    or last_result.error
                    or ""
                )
                lines.extend(
                    [
                        "",
                        f"<b>Вывод ошибки: {escape_html(last_step.label)}</b>",
                        f"<pre>{escape_html(detail_text)}</pre>",
                    ]
                )

        if report.logs_result and (
            report.logs_result.stdout_text
            or report.logs_result.stderr_text
            or report.logs_result.error
        ):
            log_body = (
                report.logs_result.stdout_text
                or report.logs_result.stderr_text
                or report.logs_result.error
                or ""
            )
            lines.extend(
                [
                    "",
                    "<b>Логи сервиса</b>",
                    f"<pre>{escape_html(log_body)}</pre>",
                ]
            )

        return "\n".join(lines)

    @staticmethod
    def select_background_verification(profile: Any) -> Optional[str]:
        """Select the best health command for background action verification."""
        command = profile.commands.get("health")
        if command:
            return command

        for service in getattr(profile, "services", ()):
            if getattr(service, "health_command", None):
                return service.health_command
        for service in getattr(profile, "services", ()):
            if getattr(service, "status_command", None):
                return service.status_command
        return None

    @staticmethod
    def select_primary_service(profile: Any) -> Optional[Any]:
        """Choose the most useful managed service for one-tap shortcuts."""
        services = list(getattr(profile, "services", ()))
        if not services:
            return None
        for key in ("app", "api", "web"):
            for service in services:
                if service.key == key:
                    return service
        return services[0] if len(services) == 1 else None
