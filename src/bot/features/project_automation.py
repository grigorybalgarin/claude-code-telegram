"""Project profiles and deterministic playbooks for common automation tasks."""

from __future__ import annotations

import json
import logging
import re
import tomllib
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from src.bot.utils.html_format import escape_html

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProjectPlaybook:
    """A named automation recipe for a workspace."""

    slug: str
    title: str
    description: str


@dataclass(frozen=True)
class ProjectProfile:
    """Detected project profile for the current workspace."""

    root_path: Path
    display_name: str
    aliases: tuple[str, ...]
    stacks: tuple[str, ...]
    package_managers: tuple[str, ...]
    commands: Dict[str, str]
    has_git_repo: bool
    has_docker_compose: bool
    services: tuple["ManagedService", ...] = ()
    operator_notes: str = ""
    sort_priority: int = 0

    def relative_root(self, approved_directory: Path) -> str:
        """Return project root relative to approved directory when possible."""
        try:
            relative = self.root_path.relative_to(approved_directory)
            return "/" if str(relative) == "." else relative.as_posix()
        except ValueError:
            return str(self.root_path)


@dataclass(frozen=True)
class AutomationPlan:
    """Resolved execution plan for a free-form user request."""

    prompt: str
    workspace_root: Path
    workspace_changed: bool
    profile: ProjectProfile
    matched_playbook: Optional[str]
    should_checkpoint: bool
    should_verify: bool
    read_only: bool


@dataclass(frozen=True)
class WorkspaceSummary:
    """Compact metadata for a discovered workspace under the approved root."""

    root_path: Path
    relative_path: str
    display_name: str
    aliases: tuple[str, ...]
    stacks: tuple[str, ...]
    playbooks: tuple[str, ...]
    has_git_repo: bool
    has_docker_compose: bool
    services_count: int
    operator_notes: str
    sort_priority: int

    @property
    def button_label(self) -> str:
        """Return a compact label suitable for inline buttons."""
        return self.relative_path


@dataclass(frozen=True)
class ManagedService:
    """Explicit service definition for deterministic operator actions."""

    key: str
    display_name: str
    service_type: str
    status_command: Optional[str] = None
    health_command: Optional[str] = None
    start_command: Optional[str] = None
    stop_command: Optional[str] = None
    restart_command: Optional[str] = None
    logs_command: Optional[str] = None

    def command_for(self, action_key: str) -> Optional[str]:
        """Return the shell command for a supported service action."""
        return {
            "status": self.status_command,
            "health": self.health_command,
            "start": self.start_command,
            "stop": self.stop_command,
            "restart": self.restart_command,
            "logs": self.logs_command,
        }.get(action_key)

    @property
    def available_actions(self) -> tuple[str, ...]:
        """Return service actions that are explicitly configured."""
        actions = []
        for action_key in ("status", "health", "logs", "restart", "start", "stop"):
            if self.command_for(action_key):
                actions.append(action_key)
        return tuple(actions)


class ProjectAutomationManager:
    """Detect workspace capabilities and generate deterministic playbooks."""

    _ROOT_MARKERS = (
        ".git",
        "pyproject.toml",
        "requirements.txt",
        "package.json",
        "Cargo.toml",
        "go.mod",
        "Makefile",
        "docker-compose.yml",
        "docker-compose.yaml",
        "compose.yml",
        "compose.yaml",
    )

    _PLAYBOOKS = {
        "doctor": ProjectPlaybook(
            slug="doctor",
            title="Doctor",
            description="Inspect the workspace, run safe diagnostics, and report state without modifying files.",
        ),
        "test": ProjectPlaybook(
            slug="test",
            title="Test",
            description="Run the detected test flow, fix straightforward failures, and verify again.",
        ),
        "quality": ProjectPlaybook(
            slug="quality",
            title="Quality",
            description="Run lint/format checks, apply safe fixes, and leave the workspace cleaner.",
        ),
        "setup": ProjectPlaybook(
            slug="setup",
            title="Setup",
            description="Prepare the workspace by installing dependencies and validating the baseline.",
        ),
        "review": ProjectPlaybook(
            slug="review",
            title="Review",
            description="Inspect git state, summarize changes and risks, and suggest the next safe move.",
        ),
    }
    _PLAYBOOK_PATTERNS = {
        "doctor": [
            re.compile(
                r"\b(doctor|diag|diagnostic|health|healthcheck|status|что\s+с\s+проектом|что\s+не\s+так|проверь\s+состояние|провер[ьи]\s+проект)\b",
                re.IGNORECASE,
            )
        ],
        "setup": [
            re.compile(
                r"\b(setup|bootstrap|install|dependencies|установи|настрой|подними|запусти\s+проект|подготовь)\b",
                re.IGNORECASE,
            )
        ],
        "test": [
            re.compile(
                r"\b(test|tests|pytest|vitest|unit test|почини\s+тест|прогони\s+тест|запусти\s+тест|тест[ыа])\b",
                re.IGNORECASE,
            )
        ],
        "quality": [
            re.compile(
                r"\b(lint|format|formatter|eslint|ruff|prettier|black|mypy|clippy|линт|формат|отформатируй|качество\s+кода)\b",
                re.IGNORECASE,
            )
        ],
        "review": [
            re.compile(
                r"\b(review|audit|ревью|аудит|проверь\s+изменения|посмотри\s+diff|git\s+status|оцени\s+риск)\b",
                re.IGNORECASE,
            )
        ],
    }
    _READ_ONLY_PATTERNS = [
        re.compile(
            r"\b(объясни|explain|почему|why|что\s+это|что\s+происходит|расскажи|show|покажи|review|audit|doctor|diag|status)\b",
            re.IGNORECASE,
        )
    ]
    _MUTATING_PATTERNS = [
        re.compile(
            r"\b(сделай|исправ|почини|добавь|измени|обнови|создай|рефактор|перепиши|удали|мигрируй|внедри|implement|fix|add|update|change|create|refactor|rewrite|remove|delete|migrate)\b",
            re.IGNORECASE,
        )
    ]

    def __init__(self, workspace_profiles_path: Optional[Path] = None) -> None:
        """Initialize the manager with optional explicit workspace profiles."""
        self.workspace_profiles_path = self._resolve_workspace_profiles_path(
            workspace_profiles_path
        )
        self._workspace_overrides = self._load_workspace_overrides(
            self.workspace_profiles_path
        )

    def detect_workspace_root(self, current_dir: Path, boundary_root: Path) -> Path:
        """Find the most relevant workspace root inside the approved boundary."""
        current = current_dir.resolve()
        boundary = boundary_root.resolve()
        fallback = current

        while True:
            if any((current / marker).exists() for marker in self._ROOT_MARKERS):
                return current

            if current == boundary:
                return fallback

            parent = current.parent
            if parent == current:
                return fallback
            current = parent

    def discover_workspace_roots(self, boundary_root: Path) -> List[Path]:
        """Discover likely workspace roots under the approved boundary."""
        boundary = boundary_root.resolve()
        candidates: List[Path] = []
        queue = deque([boundary])
        seen = {boundary}
        max_depth = 2
        max_candidates = 128

        while queue and len(candidates) < max_candidates:
            current = queue.popleft()
            try:
                children = sorted(
                    [
                        path
                        for path in current.iterdir()
                        if path.is_dir() and not path.name.startswith(".")
                    ],
                    key=lambda path: path.name.casefold(),
                )
            except OSError:
                continue

            for child in children:
                try:
                    resolved = child.resolve()
                except OSError:
                    continue
                if resolved in seen:
                    continue
                seen.add(resolved)

                try:
                    relative = resolved.relative_to(boundary)
                except ValueError:
                    continue

                depth = len(relative.parts)
                if depth > max_depth:
                    continue

                looks_like_workspace = self._looks_like_workspace_root(resolved)
                if depth == 1 or looks_like_workspace:
                    candidates.append(resolved)
                    if len(candidates) >= max_candidates:
                        break

                if depth < max_depth and (depth == 1 or not looks_like_workspace):
                    queue.append(resolved)

        return candidates

    def select_workspace_root(
        self, user_request: str, current_dir: Path, boundary_root: Path
    ) -> Path:
        """Select the best workspace for the request, falling back to the current one."""
        boundary = boundary_root.resolve()
        current_workspace = self.detect_workspace_root(current_dir, boundary)
        candidates = [current_workspace]
        seen = {current_workspace}
        for candidate in self.discover_workspace_roots(boundary):
            if candidate not in seen:
                candidates.append(candidate)
                seen.add(candidate)

        ranked = []
        for candidate in candidates:
            score = self._score_workspace_candidate(
                user_request, candidate, boundary, current_workspace
            )
            try:
                specificity = len(candidate.relative_to(boundary).parts)
            except ValueError:
                specificity = len(candidate.parts)
            ranked.append((score, specificity, candidate))

        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        best_score, _, best_path = ranked[0]
        second_best_score = ranked[1][0] if len(ranked) > 1 else 0
        if (
            best_path != current_workspace
            and best_score >= 5
            and best_score >= second_best_score + 2
        ):
            return best_path
        return current_workspace

    def build_profile(self, current_dir: Path, boundary_root: Path) -> ProjectProfile:
        """Detect profile details for the current workspace."""
        boundary = boundary_root.resolve()
        root = self.detect_workspace_root(current_dir, boundary)
        stacks: List[str] = []
        package_managers: List[str] = []
        commands: Dict[str, str] = {}

        package_json = root / "package.json"
        pyproject = root / "pyproject.toml"
        cargo_toml = root / "Cargo.toml"
        go_mod = root / "go.mod"
        makefile = root / "Makefile"

        if package_json.exists():
            stacks.append("node")
            package_manager = self._detect_node_package_manager(root)
            package_managers.append(package_manager)
            commands.update(self._detect_node_commands(package_json, package_manager))

        if pyproject.exists() or (root / "requirements.txt").exists():
            stacks.append("python")
            package_manager = self._detect_python_package_manager(root)
            package_managers.append(package_manager)
            commands.update(self._detect_python_commands(root, package_manager))

        if cargo_toml.exists():
            stacks.append("rust")
            package_managers.append("cargo")
            commands.setdefault("test", "cargo test")
            commands.setdefault(
                "lint", "cargo clippy --all-targets --all-features -- -D warnings"
            )
            commands.setdefault("format", "cargo fmt --all")
            commands.setdefault("build", "cargo build")

        if go_mod.exists():
            stacks.append("go")
            package_managers.append("go")
            commands.setdefault("test", "go test ./...")
            commands.setdefault("format", "gofmt -w .")
            commands.setdefault("build", "go build ./...")

        if makefile.exists():
            commands.update(self._detect_make_targets(makefile))

        compose_file = self._find_compose_file(root)
        if compose_file:
            commands.setdefault("compose_status", "docker compose ps")
            commands.setdefault("compose_up", "docker compose up -d")

        has_git_repo = (root / ".git").exists()
        if has_git_repo:
            commands.setdefault("git_status", "git status --short --branch")

        if not stacks:
            stacks.append("generic")

        profile = ProjectProfile(
            root_path=root,
            display_name=root.name or str(root),
            aliases=tuple(),
            stacks=tuple(dict.fromkeys(stacks)),
            package_managers=tuple(dict.fromkeys(package_managers)),
            commands=commands,
            has_git_repo=has_git_repo,
            has_docker_compose=compose_file is not None,
            services=tuple(),
        )
        return self._apply_workspace_override(profile, boundary)

    def list_playbooks(self, profile: ProjectProfile) -> List[ProjectPlaybook]:
        """Return playbooks relevant to the detected profile."""
        available = [self._PLAYBOOKS["doctor"]]

        if "install" in profile.commands:
            available.append(self._PLAYBOOKS["setup"])
        if "test" in profile.commands:
            available.append(self._PLAYBOOKS["test"])
        if "lint" in profile.commands or "format" in profile.commands:
            available.append(self._PLAYBOOKS["quality"])
        if profile.has_git_repo:
            available.append(self._PLAYBOOKS["review"])

        return available

    def list_operator_commands(self, profile: ProjectProfile) -> List[tuple[str, str]]:
        """Return explicit operator-facing commands worth surfacing in the UI."""
        commands: List[tuple[str, str]] = []
        for key in ("health", "build", "start", "dev", "deploy"):
            command = profile.commands.get(key)
            if command:
                commands.append((key, command))
        return commands

    def list_managed_services(self, profile: ProjectProfile) -> List[ManagedService]:
        """Return explicitly configured managed services for the workspace."""
        return list(profile.services)

    def list_workspace_summaries(self, boundary_root: Path) -> List[WorkspaceSummary]:
        """Return discovered workspaces with compact profile metadata."""
        boundary = boundary_root.resolve()
        discovered = []
        if self._looks_like_workspace_root(boundary):
            discovered.append(boundary)
        discovered.extend(self.discover_workspace_roots(boundary))

        unique_roots: dict[Path, WorkspaceSummary] = {}
        for candidate in discovered:
            profile = self.build_profile(candidate, boundary)
            root = profile.root_path
            if root in unique_roots:
                continue
            relative_path = profile.relative_root(boundary)
            unique_roots[root] = WorkspaceSummary(
                root_path=root,
                relative_path=relative_path,
                display_name=profile.display_name,
                aliases=profile.aliases,
                stacks=profile.stacks,
                playbooks=tuple(
                    playbook.slug for playbook in self.list_playbooks(profile)
                ),
                has_git_repo=profile.has_git_repo,
                has_docker_compose=profile.has_docker_compose,
                services_count=len(profile.services),
                operator_notes=profile.operator_notes,
                sort_priority=profile.sort_priority,
            )

        summaries = sorted(
            unique_roots.values(),
            key=lambda summary: (
                -summary.sort_priority,
                summary.relative_path != "/",
                summary.relative_path.casefold(),
            ),
        )
        return [
            summary
            for summary in summaries
            if not self._is_container_workspace(summary, summaries)
        ]

    def resolve_workspace_reference(
        self, reference: str, boundary_root: Path
    ) -> Optional[WorkspaceSummary]:
        """Resolve a manual workspace reference by path or display name."""
        normalized = reference.strip().strip("/")
        if not normalized:
            return None

        boundary = boundary_root.resolve()
        direct_candidate = (boundary / normalized).resolve()
        try:
            direct_candidate.relative_to(boundary)
        except ValueError:
            direct_candidate = boundary / "__outside__"

        if direct_candidate.exists() and direct_candidate.is_dir():
            target_root = self.detect_workspace_root(direct_candidate, boundary)
            for summary in self.list_workspace_summaries(boundary):
                if summary.root_path == target_root:
                    return summary

        normalized_raw = normalized.casefold()
        normalized_compact = self._normalize_match_text(normalized)
        matches: List[WorkspaceSummary] = []

        for summary in self.list_workspace_summaries(boundary):
            candidates = {
                summary.relative_path.casefold(),
                summary.display_name.casefold(),
                summary.root_path.name.casefold(),
                self._normalize_match_text(summary.relative_path),
                self._normalize_match_text(summary.display_name),
                self._normalize_match_text(summary.root_path.name),
                *(alias.casefold() for alias in summary.aliases),
                *(self._normalize_match_text(alias) for alias in summary.aliases),
            }
            if normalized_raw in candidates or normalized_compact in candidates:
                matches.append(summary)

        if len(matches) == 1:
            return matches[0]
        return None

    def describe_workspace_summary_lines(
        self,
        summary: WorkspaceSummary,
        current_workspace: Optional[Path] = None,
    ) -> List[str]:
        """Build compact Telegram lines for a workspace summary."""
        marker = " ◀" if current_workspace and summary.root_path == current_workspace else ""
        stacks = ", ".join(summary.stacks)
        badges = []
        if summary.has_git_repo:
            badges.append("git")
        if summary.has_docker_compose:
            badges.append("compose")
        if summary.services_count:
            badges.append(f"svc:{summary.services_count}")
        badges_text = f" · {' · '.join(badges)}" if badges else ""
        playbooks_text = ", ".join(summary.playbooks)
        lines = [
            (
                f"• <b>{escape_html(summary.display_name)}</b>{marker} "
                f"→ <code>{escape_html(summary.relative_path)}</code>"
            ),
            (
                f"  <code>{escape_html(stacks)}</code>{badges_text}"
                f" · playbooks: <code>{escape_html(playbooks_text)}</code>"
            ),
        ]
        if summary.aliases:
            lines.append(
                f"  aliases: <code>{escape_html(', '.join(summary.aliases))}</code>"
            )
        return lines

    def get_playbook(self, slug: str, profile: ProjectProfile) -> Optional[ProjectPlaybook]:
        """Return a playbook only if it is available for the profile."""
        normalized = slug.strip().lower()
        for playbook in self.list_playbooks(profile):
            if playbook.slug == normalized:
                return playbook
        return None

    def classify_playbook(
        self, user_request: str, profile: ProjectProfile
    ) -> Optional[ProjectPlaybook]:
        """Infer the best matching playbook from a free-form request."""
        for slug, patterns in self._PLAYBOOK_PATTERNS.items():
            playbook = self.get_playbook(slug, profile)
            if playbook is None:
                continue
            if any(pattern.search(user_request) for pattern in patterns):
                return playbook
        return None

    def build_automation_plan(
        self,
        user_request: str,
        current_dir: Path,
        boundary_root: Path,
    ) -> AutomationPlan:
        """Build an always-on autopilot plan for a natural-language request."""
        current_profile = self.build_profile(current_dir, boundary_root)
        selected_workspace = self.select_workspace_root(
            user_request, current_dir, boundary_root
        )
        profile = self.build_profile(selected_workspace, boundary_root)
        workspace_changed = profile.root_path != current_profile.root_path
        playbook = self.classify_playbook(user_request, profile)
        read_only = playbook is not None and playbook.slug in {"doctor", "review"}
        if not read_only:
            read_only = any(
                pattern.search(user_request) for pattern in self._READ_ONLY_PATTERNS
            ) and not any(
                pattern.search(user_request) for pattern in self._MUTATING_PATTERNS
            )

        mutating = playbook is not None and playbook.slug in {"setup", "test", "quality"}
        if not mutating:
            mutating = any(
                pattern.search(user_request) for pattern in self._MUTATING_PATTERNS
            )
        if read_only:
            mutating = False

        if playbook is not None:
            prompt = self.build_playbook_prompt(
                playbook.slug,
                profile,
                extra_instructions=f"Original user request: {user_request}",
            )
        else:
            prompt = self.build_general_autopilot_prompt(user_request, profile)

        verification_commands = self.get_verification_commands(profile)
        return AutomationPlan(
            prompt=prompt,
            workspace_root=profile.root_path,
            workspace_changed=workspace_changed,
            profile=profile,
            matched_playbook=playbook.slug if playbook else None,
            should_checkpoint=mutating and profile.has_git_repo,
            should_verify=mutating and bool(verification_commands),
            read_only=read_only,
        )

    def build_playbook_prompt(
        self,
        slug: str,
        profile: ProjectProfile,
        extra_instructions: str = "",
    ) -> str:
        """Generate a deterministic prompt for a named playbook."""
        playbook = self.get_playbook(slug, profile)
        if playbook is None:
            raise ValueError(f"Unknown playbook: {slug}")

        command_lines = [
            f"- {name}: {command}" for name, command in sorted(profile.commands.items())
        ]
        detected_commands = "\n".join(command_lines) or "- none detected"
        stacks = ", ".join(profile.stacks)
        managers = ", ".join(profile.package_managers) or "none detected"
        notes_block = ""
        if profile.operator_notes:
            notes_block = f"Project notes:\n{profile.operator_notes}\n\n"

        base_context = (
            f'You are running the "{playbook.slug}" project playbook.\n'
            f"Project root: {profile.root_path}\n"
            f"Detected stack: {stacks}\n"
            f"Detected package managers: {managers}\n"
            f"{notes_block}"
            "Detected workspace commands:\n"
            f"{detected_commands}\n\n"
            "When a detected command exists, prefer it over guessing an alternative.\n"
            "Stay inside the project root and use relative paths in your final report.\n"
        )

        if playbook.slug == "doctor":
            body = (
                "Goal:\n"
                "1. Inspect repository structure and current state.\n"
                "2. Run only safe diagnostic and read-only commands.\n"
                "3. If test, lint, git, or compose status commands are available, use the relevant status/check variant.\n"
                "4. Do not modify files.\n"
                "5. Return a short operational summary, blockers, and next recommended actions.\n"
            )
        elif playbook.slug == "setup":
            install_command = profile.commands.get("install")
            body = (
                "Goal:\n"
                "1. Prepare the workspace so it is ready for work.\n"
                f"2. Start with the detected install command: {install_command}\n"
                "3. If a build or test command exists, run the lightest useful verification afterward.\n"
                "4. Fix only straightforward setup issues inside the project.\n"
                "5. Report what was done and what still needs manual attention.\n"
            )
        elif playbook.slug == "test":
            body = (
                "Goal:\n"
                f"1. Run the detected test command: {profile.commands.get('test')}\n"
                "2. Investigate failures and fix straightforward issues in project files.\n"
                "3. Re-run tests after changes until they pass or you hit a real blocker.\n"
                "4. If there are blockers, stop and explain the exact failure point.\n"
                "5. End with a concise summary of changed files, test result, and remaining risk.\n"
            )
        elif playbook.slug == "quality":
            steps = []
            if "lint" in profile.commands:
                steps.append(f"Run lint first: {profile.commands['lint']}")
            if "typecheck" in profile.commands:
                steps.append(
                    f"Run type checks before finishing: {profile.commands['typecheck']}"
                )
            if "format" in profile.commands:
                steps.append(f"Run formatting when helpful: {profile.commands['format']}")
            steps.extend(
                [
                    "Fix straightforward code quality issues.",
                    "Re-run the relevant checks after modifications.",
                    "End with a concise summary of what changed and any remaining warnings.",
                ]
            )
            step_block = "\n".join(
                f"{idx}. {step}" for idx, step in enumerate(steps, start=1)
            )
            if not step_block:
                step_block = "1. Use the project's available quality tools."
            body = f"Goal:\n{step_block}\n"
        else:
            doctor_steps = [
                "1. Inspect git state and recent changes.",
                "2. Review the current workspace for risk, regressions, and missing follow-up work.",
                "3. Use git status/diff/log when relevant.",
            ]
            if "health" in profile.commands:
                doctor_steps.append(
                    f"4. Run the detected health command when useful: {profile.commands['health']}"
                )
            if "typecheck" in profile.commands:
                doctor_steps.append(
                    f"{len(doctor_steps) + 1}. Include the detected typecheck command when it helps surface risk: {profile.commands['typecheck']}"
                )
            doctor_steps.extend(
                [
                    f"{len(doctor_steps) + 1}. Do not modify files unless fixing a trivial broken state is clearly necessary.",
                    f"{len(doctor_steps) + 2}. End with findings first, then a short change summary.",
                ]
            )
            body = (
                "Goal:\n" + "\n".join(doctor_steps) + "\n"
            )

        extra = (
            f"\nOperator note: {extra_instructions.strip()}\n" if extra_instructions.strip() else ""
        )
        return f"{base_context}\n{body}{extra}"

    def build_general_autopilot_prompt(
        self, user_request: str, profile: ProjectProfile
    ) -> str:
        """Build the default always-on autopilot prompt for arbitrary work."""
        command_lines = [
            f"- {name}: {command}" for name, command in sorted(profile.commands.items())
        ]
        detected_commands = "\n".join(command_lines) or "- none detected"
        stacks = ", ".join(profile.stacks)
        managers = ", ".join(profile.package_managers) or "none detected"
        notes_block = ""
        if profile.operator_notes:
            notes_block = f"Project notes:\n{profile.operator_notes}\n\n"
        return (
            "You are operating in always-on autopilot mode for a personal coding assistant.\n"
            f"Project root: {profile.root_path}\n"
            f"Detected stack: {stacks}\n"
            f"Detected package managers: {managers}\n"
            f"{notes_block}"
            "Detected workspace commands:\n"
            f"{detected_commands}\n\n"
            "Operating rules:\n"
            "1. Prefer the detected commands over guessing alternatives.\n"
            "2. Start by understanding the current state before editing files.\n"
            "3. If the user asks for changes, make the smallest coherent set of edits.\n"
            "4. Use relative paths in your final response.\n"
            "5. The bot will automatically create a checkpoint before risky changes and automatically run final verification after your edits. "
            "You still may run targeted checks during debugging when useful.\n"
            "6. If the request is informational or diagnostic, stay read-only.\n"
            "7. End with a concise summary of what changed, what was verified, and any blockers.\n\n"
            f"User request: {user_request}"
        )

    def get_verification_commands(self, profile: ProjectProfile) -> List[str]:
        """Return project-wide verification commands in priority order."""
        commands: List[str] = []
        for key in ("lint", "typecheck", "test", "build"):
            command = profile.commands.get(key)
            if command and command not in commands:
                commands.append(command)
        return commands

    def describe_profile_lines(
        self, profile: ProjectProfile, approved_directory: Path
    ) -> List[str]:
        """Build compact human-readable profile lines for Telegram."""
        command_lines = [
            f"• <code>{escape_html(name)}</code>: <code>{escape_html(command)}</code>"
            for name, command in sorted(profile.commands.items())
        ]
        return [
            f"🧱 Stack: <code>{escape_html(', '.join(profile.stacks))}</code>",
            f"🛠️ Package managers: <code>{escape_html(', '.join(profile.package_managers) or 'none')}</code>",
            f"🔗 Git: {'yes' if profile.has_git_repo else 'no'}",
            f"🐳 Compose: {'yes' if profile.has_docker_compose else 'no'}",
            f"🧩 Services: <code>{escape_html(', '.join(service.display_name for service in profile.services) or 'none')}</code>",
            "⚙️ Detected commands:" if command_lines else "⚙️ Detected commands: none",
            *command_lines,
        ]

    def _detect_node_package_manager(self, root: Path) -> str:
        if (root / "pnpm-lock.yaml").exists():
            return "pnpm"
        if (root / "yarn.lock").exists():
            return "yarn"
        return "npm"

    def _detect_node_commands(
        self, package_json_path: Path, package_manager: str
    ) -> Dict[str, str]:
        commands: Dict[str, str] = {"install": f"{package_manager} install"}
        try:
            package_data = json.loads(package_json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return commands

        scripts = package_data.get("scripts", {})
        if not isinstance(scripts, dict):
            return commands

        for name in (
            "test",
            "lint",
            "format",
            "build",
            "typecheck",
            "check",
            "health",
            "deploy",
            "start",
            "dev",
        ):
            if name in scripts:
                commands[name] = f"{package_manager} run {name}"

        return commands

    def _detect_python_package_manager(self, root: Path) -> str:
        if (root / "uv.lock").exists():
            return "uv"
        if (root / "poetry.lock").exists():
            return "poetry"
        pyproject = root / "pyproject.toml"
        if pyproject.exists():
            try:
                data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            except (OSError, tomllib.TOMLDecodeError):
                data = {}
            tool = data.get("tool", {})
            if isinstance(tool, dict) and "poetry" in tool:
                return "poetry"
        if (root / "requirements.txt").exists():
            return "pip"
        return "python"

    def _detect_python_commands(self, root: Path, package_manager: str) -> Dict[str, str]:
        commands: Dict[str, str] = {}
        pyproject = root / "pyproject.toml"
        pyproject_text = ""
        if pyproject.exists():
            pyproject_text = pyproject.read_text(encoding="utf-8")

        prefix = {
            "poetry": "poetry run ",
            "uv": "uv run ",
            "pip": "",
            "python": "",
        }.get(package_manager, "")

        if package_manager == "poetry":
            commands["install"] = "poetry install"
        elif package_manager == "uv":
            commands["install"] = "uv sync"
        elif (root / "requirements.txt").exists():
            commands["install"] = "pip install -r requirements.txt"
        elif pyproject.exists():
            commands["install"] = "pip install -e ."

        lowered = pyproject_text.lower()
        if any(marker in lowered for marker in ("pytest", "[tool.pytest", "pytest.ini_options")) or (
            root / "pytest.ini"
        ).exists():
            commands.setdefault("test", f"{prefix}pytest".strip())

        if "ruff" in lowered or (root / "ruff.toml").exists() or (root / ".ruff.toml").exists():
            commands.setdefault("lint", f"{prefix}ruff check .".strip())
            commands.setdefault("format", f"{prefix}ruff format .".strip())
        elif "black" in lowered:
            commands.setdefault("format", f"{prefix}black .".strip())

        if "mypy" in lowered:
            commands.setdefault("typecheck", f"{prefix}mypy .".strip())

        if not commands.get("test") and (root / "tests").exists():
            commands["test"] = f"{prefix}pytest".strip()

        return commands

    def _detect_make_targets(self, makefile: Path) -> Dict[str, str]:
        targets: Dict[str, str] = {}
        try:
            content = makefile.read_text(encoding="utf-8")
        except OSError:
            return targets

        lowered = {line.split(":")[0].strip().lower() for line in content.splitlines() if ":" in line}
        if "test" in lowered:
            targets.setdefault("test", "make test")
        if "lint" in lowered:
            targets.setdefault("lint", "make lint")
        if "format" in lowered:
            targets.setdefault("format", "make format")
        if "typecheck" in lowered:
            targets.setdefault("typecheck", "make typecheck")
        if "check" in lowered:
            targets.setdefault("typecheck", "make check")
        if "install" in lowered:
            targets.setdefault("install", "make install")
        if "build" in lowered:
            targets.setdefault("build", "make build")
        if "health" in lowered:
            targets.setdefault("health", "make health")
        if "deploy" in lowered:
            targets.setdefault("deploy", "make deploy")
        if "start" in lowered:
            targets.setdefault("start", "make start")
        if "dev" in lowered:
            targets.setdefault("dev", "make dev")
        return targets

    def _find_compose_file(self, root: Path) -> Optional[Path]:
        for name in ("docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml"):
            candidate = root / name
            if candidate.exists():
                return candidate
        return None

    def _looks_like_workspace_root(self, path: Path) -> bool:
        return any((path / marker).exists() for marker in self._ROOT_MARKERS)

    def _score_workspace_candidate(
        self,
        user_request: str,
        candidate: Path,
        boundary_root: Path,
        current_workspace: Path,
    ) -> int:
        request_text = user_request.casefold()
        normalized_request = self._normalize_match_text(user_request)
        candidate_name = candidate.name.casefold()
        normalized_name = self._normalize_match_text(candidate.name)
        relative_label = self._workspace_label(candidate, boundary_root).lstrip("/")
        normalized_relative = self._normalize_match_text(relative_label)
        profile = self.build_profile(candidate, boundary_root)

        score = 1 if candidate == current_workspace else 0

        if relative_label and self._contains_path_reference(request_text, relative_label):
            score = max(score, 9 + relative_label.count("/"))
        if (
            normalized_relative
            and len(normalized_relative) >= 6
            and normalized_relative in normalized_request
        ):
            score = max(score, 8 + relative_label.count("/"))
        if len(candidate_name) >= 4 and self._contains_workspace_name(
            request_text, candidate_name
        ):
            score = max(score, 7)
        if len(normalized_name) >= 4 and normalized_name in normalized_request:
            score = max(score, 6)
        for alias in profile.aliases:
            normalized_alias = self._normalize_match_text(alias)
            if len(normalized_alias) >= 3 and normalized_alias in normalized_request:
                score = max(score, 8)
            elif len(alias) >= 3 and self._contains_workspace_name(request_text, alias.casefold()):
                score = max(score, 8)

        return score

    def _workspace_label(self, path: Path, boundary_root: Path) -> str:
        try:
            relative = path.relative_to(boundary_root.resolve())
            return "/" if str(relative) == "." else relative.as_posix()
        except ValueError:
            return str(path)

    def _contains_workspace_name(self, request_text: str, name: str) -> bool:
        pattern = re.compile(
            rf"(?<![0-9a-zа-яё]){re.escape(name)}(?![0-9a-zа-яё])",
            re.IGNORECASE,
        )
        return bool(pattern.search(request_text))

    def _contains_path_reference(self, request_text: str, relative_label: str) -> bool:
        pattern = re.compile(
            rf"(?<![0-9a-zа-яё/]){re.escape(relative_label.casefold())}(?![0-9a-zа-яё/])",
            re.IGNORECASE,
        )
        return bool(pattern.search(request_text))

    def _normalize_match_text(self, value: str) -> str:
        return re.sub(r"[^0-9a-zа-яё]+", "", value.casefold())

    def _resolve_workspace_profiles_path(
        self, workspace_profiles_path: Optional[Path]
    ) -> Optional[Path]:
        if workspace_profiles_path is not None:
            return workspace_profiles_path.resolve()
        default_path = (
            Path(__file__).resolve().parents[3] / "config" / "workspace_profiles.yaml"
        )
        return default_path if default_path.exists() else None

    def _load_workspace_overrides(
        self, workspace_profiles_path: Optional[Path]
    ) -> dict[str, dict[str, object]]:
        if workspace_profiles_path is None:
            return {}

        try:
            raw = yaml.safe_load(workspace_profiles_path.read_text(encoding="utf-8")) or {}
        except OSError as exc:
            logger.warning(
                "Failed to read workspace profiles %s: %s",
                workspace_profiles_path,
                exc,
            )
            return {}
        except yaml.YAMLError as exc:
            logger.warning(
                "Failed to parse workspace profiles %s: %s",
                workspace_profiles_path,
                exc,
            )
            return {}

        if not isinstance(raw, dict):
            return {}

        overrides: dict[str, dict[str, object]] = {}
        entries = raw.get("workspaces", [])
        if not isinstance(entries, list):
            return overrides

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            raw_path = str(entry.get("path", "")).strip().strip("/")
            if not raw_path:
                continue
            commands = entry.get("commands") or {}
            if not isinstance(commands, dict):
                commands = {}
            aliases = entry.get("aliases") or []
            if not isinstance(aliases, list):
                aliases = []
            raw_services = entry.get("services")
            services = None
            if isinstance(raw_services, list):
                parsed_services = [
                    service
                    for service in (
                        self._parse_service_override(item)
                        for item in raw_services
                        if isinstance(item, dict)
                    )
                    if service is not None
                ]
                services = tuple(parsed_services)
            overrides[raw_path] = {
                "display_name": str(entry.get("name", "")).strip() or None,
                "aliases": tuple(
                    alias.strip()
                    for alias in aliases
                    if isinstance(alias, str) and alias.strip()
                ),
                "operator_notes": str(entry.get("notes", "")).strip(),
                "commands": {
                    str(key).strip(): (
                        str(value).strip() if value is not None and str(value).strip() else None
                    )
                    for key, value in commands.items()
                    if str(key).strip()
                },
                "services": services,
                "sort_priority": int(entry.get("priority", 0) or 0),
            }

        return overrides

    def _apply_workspace_override(
        self, profile: ProjectProfile, boundary_root: Path
    ) -> ProjectProfile:
        override = self._workspace_overrides.get(profile.relative_root(boundary_root))
        if not override:
            return profile

        commands = dict(profile.commands)
        for key, value in (override.get("commands") or {}).items():
            if value is None:
                commands.pop(str(key), None)
            else:
                commands[str(key)] = str(value)

        display_name = str(override.get("display_name") or profile.display_name)
        aliases = tuple(
            dict.fromkeys(
                [*profile.aliases, *(override.get("aliases") or ())]
            )
        )
        services = profile.services
        if override.get("services") is not None:
            services = tuple(override.get("services") or ())
        return ProjectProfile(
            root_path=profile.root_path,
            display_name=display_name,
            aliases=aliases,
            stacks=profile.stacks,
            package_managers=profile.package_managers,
            commands=commands,
            has_git_repo=profile.has_git_repo,
            has_docker_compose=profile.has_docker_compose,
            services=services,
            operator_notes=str(override.get("operator_notes") or ""),
            sort_priority=int(override.get("sort_priority") or 0),
        )

    def _parse_service_override(
        self, payload: dict[str, object]
    ) -> Optional[ManagedService]:
        """Parse one explicit managed service definition from YAML."""
        service_type = str(payload.get("type", "command")).strip().lower() or "command"
        raw_name = str(payload.get("name", "")).strip()
        raw_key = str(payload.get("key", "")).strip()

        if service_type == "systemd":
            unit = str(payload.get("unit", "")).strip()
            if not unit:
                return None
            key = raw_key or self._normalize_match_text(unit) or "systemd"
            display_name = raw_name or unit
            logs_tail = max(1, int(payload.get("logs_tail", 80) or 80))
            return ManagedService(
                key=key,
                display_name=display_name,
                service_type="systemd",
                status_command=f"systemctl status {unit} --no-pager",
                health_command=f"systemctl is-active {unit}",
                start_command=f"systemctl start {unit}",
                stop_command=f"systemctl stop {unit}",
                restart_command=f"systemctl restart {unit}",
                logs_command=f"journalctl -u {unit} -n {logs_tail} --no-pager",
            )

        if service_type == "compose":
            service_name = str(payload.get("service", "")).strip()
            selector = f" {service_name}" if service_name else ""
            key = raw_key or self._normalize_match_text(service_name or raw_name) or "compose"
            display_name = raw_name or service_name or "compose"
            logs_tail = max(1, int(payload.get("logs_tail", 80) or 80))
            return ManagedService(
                key=key,
                display_name=display_name,
                service_type="compose",
                status_command=f"docker compose ps{selector}",
                start_command=f"docker compose up -d{selector}",
                stop_command=f"docker compose stop{selector}",
                restart_command=f"docker compose restart{selector}",
                logs_command=f"docker compose logs --tail {logs_tail}{selector}",
            )

        key = raw_key or self._normalize_match_text(raw_name) or "service"
        display_name = raw_name or key
        service = ManagedService(
            key=key,
            display_name=display_name,
            service_type="command",
            status_command=self._normalize_optional_command(payload.get("status")),
            health_command=self._normalize_optional_command(payload.get("health")),
            start_command=self._normalize_optional_command(payload.get("start")),
            stop_command=self._normalize_optional_command(payload.get("stop")),
            restart_command=self._normalize_optional_command(payload.get("restart")),
            logs_command=self._normalize_optional_command(payload.get("logs")),
        )
        return service if service.available_actions else None

    def _normalize_optional_command(self, value: object) -> Optional[str]:
        """Normalize an optional command from YAML."""
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _is_container_workspace(
        self, summary: WorkspaceSummary, summaries: List[WorkspaceSummary]
    ) -> bool:
        if summary.relative_path == "/" or self._looks_like_workspace_root(summary.root_path):
            return False

        prefix = f"{summary.relative_path.rstrip('/')}/"
        return any(
            other.relative_path != summary.relative_path
            and other.relative_path.startswith(prefix)
            for other in summaries
        )
