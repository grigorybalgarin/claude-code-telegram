"""YAML-backed agent registry.

Loads agent definitions from config/agents.yaml.  Each agent has a slug,
department, and a path to a Markdown system-prompt file.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass(frozen=True)
class AgentDefinition:
    """A single agent entry from the YAML config."""

    slug: str
    name: str
    department: str
    prompt_file: Path
    enabled: bool = True
    working_directory: Optional[Path] = None
    tools_allowlist: List[str] = field(default_factory=list)

    def load_prompt(self, base_dir: Path) -> str:
        """Read the Markdown prompt file relative to *base_dir*."""
        path = base_dir / self.prompt_file
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return path.read_text(encoding="utf-8").strip()


@dataclass(frozen=True)
class RoutingRule:
    """Maps an event pattern to an agent slug."""

    agent: str
    provider: Optional[str] = None
    event_type: Optional[str] = None
    department: Optional[str] = None


class AgentRegistry:
    """In-memory validated agent registry."""

    def __init__(
        self,
        agents: List[AgentDefinition],
        routing_rules: List[RoutingRule],
    ) -> None:
        self._agents = agents
        self._by_slug: Dict[str, AgentDefinition] = {a.slug: a for a in agents}
        self._by_dept: Dict[str, List[AgentDefinition]] = {}
        for a in agents:
            self._by_dept.setdefault(a.department, []).append(a)
        self.routing_rules = routing_rules

    @property
    def agents(self) -> List[AgentDefinition]:
        return list(self._agents)

    def list_enabled(self) -> List[AgentDefinition]:
        return [a for a in self._agents if a.enabled]

    def get_by_slug(self, slug: str) -> Optional[AgentDefinition]:
        return self._by_slug.get(slug)

    def get_by_department(self, department: str) -> List[AgentDefinition]:
        return list(self._by_dept.get(department, []))


def load_agent_registry(config_path: Path) -> AgentRegistry:
    """Load and validate agent definitions from YAML."""
    if not config_path.exists():
        raise ValueError(f"Agents config does not exist: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError("Agents config must be a YAML mapping")

    raw_agents = data.get("agents")
    if not isinstance(raw_agents, list) or not raw_agents:
        raise ValueError("Agents config must contain a non-empty 'agents' list")

    seen_slugs: set[str] = set()
    agents: List[AgentDefinition] = []

    for idx, raw in enumerate(raw_agents):
        if not isinstance(raw, dict):
            raise ValueError(f"Agent entry at index {idx} must be a mapping")

        slug = str(raw.get("slug", "")).strip()
        name = str(raw.get("name", "")).strip()
        department = str(raw.get("department", "")).strip()
        prompt_file = str(raw.get("prompt_file", "")).strip()

        if not slug:
            raise ValueError(f"Agent at index {idx} missing 'slug'")
        if not name:
            raise ValueError(f"Agent '{slug}' missing 'name'")
        if not department:
            raise ValueError(f"Agent '{slug}' missing 'department'")
        if not prompt_file:
            raise ValueError(f"Agent '{slug}' missing 'prompt_file'")
        if slug in seen_slugs:
            raise ValueError(f"Duplicate agent slug: {slug}")

        seen_slugs.add(slug)
        enabled = bool(raw.get("enabled", True))
        wd_raw = raw.get("working_directory")
        working_directory = Path(wd_raw) if wd_raw else None
        tools_allowlist = list(raw.get("tools_allowlist") or [])

        agents.append(
            AgentDefinition(
                slug=slug,
                name=name,
                department=department,
                prompt_file=Path(prompt_file),
                enabled=enabled,
                working_directory=working_directory,
                tools_allowlist=tools_allowlist,
            )
        )

    # Routing rules
    raw_rules = data.get("routing_rules") or []
    rules: List[RoutingRule] = []
    for rr in raw_rules:
        if not isinstance(rr, dict):
            continue
        agent_slug = str(rr.get("agent", "")).strip()
        if not agent_slug:
            continue
        rules.append(
            RoutingRule(
                agent=agent_slug,
                provider=rr.get("provider"),
                event_type=rr.get("event_type"),
                department=rr.get("department"),
            )
        )

    return AgentRegistry(agents, rules)
