"""Project profiles and deterministic playbooks for common automation tasks."""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from src.bot.utils.html_format import escape_html


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
    stacks: tuple[str, ...]
    package_managers: tuple[str, ...]
    commands: Dict[str, str]
    has_git_repo: bool
    has_docker_compose: bool

    def relative_root(self, approved_directory: Path) -> str:
        """Return project root relative to approved directory when possible."""
        try:
            relative = self.root_path.relative_to(approved_directory)
            return "/" if str(relative) == "." else str(relative)
        except ValueError:
            return str(self.root_path)


@dataclass(frozen=True)
class AutomationPlan:
    """Resolved execution plan for a free-form user request."""

    prompt: str
    workspace_root: Path
    profile: ProjectProfile
    matched_playbook: Optional[str]
    should_checkpoint: bool
    should_verify: bool
    read_only: bool


class ProjectAutomationManager:
    """Detect workspace capabilities and generate deterministic playbooks."""

    _ROOT_MARKERS = (
        ".git",
        "pyproject.toml",
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

    def build_profile(self, current_dir: Path, boundary_root: Path) -> ProjectProfile:
        """Detect profile details for the current workspace."""
        root = self.detect_workspace_root(current_dir, boundary_root)
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

        return ProjectProfile(
            root_path=root,
            display_name=root.name or str(root),
            stacks=tuple(dict.fromkeys(stacks)),
            package_managers=tuple(dict.fromkeys(package_managers)),
            commands=commands,
            has_git_repo=has_git_repo,
            has_docker_compose=compose_file is not None,
        )

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
        profile = self.build_profile(current_dir, boundary_root)
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

        base_context = (
            f'You are running the "{playbook.slug}" project playbook.\n'
            f"Project root: {profile.root_path}\n"
            f"Detected stack: {stacks}\n"
            f"Detected package managers: {managers}\n"
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
            body = (
                "Goal:\n"
                "1. Inspect git state and recent changes.\n"
                "2. Review the current workspace for risk, regressions, and missing follow-up work.\n"
                "3. Use git status/diff/log when relevant.\n"
                "4. Do not modify files unless fixing a trivial broken state is clearly necessary.\n"
                "5. End with findings first, then a short change summary.\n"
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
        return (
            "You are operating in always-on autopilot mode for a personal coding assistant.\n"
            f"Project root: {profile.root_path}\n"
            f"Detected stack: {stacks}\n"
            f"Detected package managers: {managers}\n"
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
        for key in ("lint", "test", "build"):
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

        for name in ("test", "lint", "format", "build"):
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
        if "install" in lowered:
            targets.setdefault("install", "make install")
        if "build" in lowered:
            targets.setdefault("build", "make build")
        return targets

    def _find_compose_file(self, root: Path) -> Optional[Path]:
        for name in ("docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml"):
            candidate = root / name
            if candidate.exists():
                return candidate
        return None
