"""Tests for the agent registry and router."""

import textwrap
from pathlib import Path

import pytest
import yaml

from src.agents.registry import (
    AgentDefinition,
    AgentRegistry,
    RoutingRule,
    load_agent_registry,
)
from src.agents.router import AgentRouter
from src.events.types import ScheduledEvent, WebhookEvent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_agents() -> list[AgentDefinition]:
    return [
        AgentDefinition(
            slug="deaify-text",
            name="Text Humanizer",
            department="content",
            prompt_file=Path("prompts/deaify-text.md"),
        ),
        AgentDefinition(
            slug="competitor-analysis",
            name="Competitor Analyst",
            department="rd",
            prompt_file=Path("prompts/competitor-analysis.md"),
        ),
    ]


@pytest.fixture
def registry_with_rules(two_agents: list[AgentDefinition]) -> AgentRegistry:
    rules = [
        RoutingRule(agent="competitor-analysis", provider="github", event_type="issues"),
    ]
    return AgentRegistry(two_agents, rules)


@pytest.fixture
def router(registry_with_rules: AgentRegistry) -> AgentRouter:
    return AgentRouter(registry_with_rules)


# ---------------------------------------------------------------------------
# AgentRegistry tests
# ---------------------------------------------------------------------------


class TestAgentRegistry:
    def test_list_enabled(self, two_agents: list[AgentDefinition]) -> None:
        registry = AgentRegistry(two_agents, [])
        assert len(registry.list_enabled()) == 2

    def test_list_enabled_filters_disabled(
        self, two_agents: list[AgentDefinition]
    ) -> None:
        disabled = AgentDefinition(
            slug="disabled-agent",
            name="Disabled",
            department="test",
            prompt_file=Path("x.md"),
            enabled=False,
        )
        registry = AgentRegistry([*two_agents, disabled], [])
        assert len(registry.list_enabled()) == 2

    def test_get_by_slug(self, two_agents: list[AgentDefinition]) -> None:
        registry = AgentRegistry(two_agents, [])
        agent = registry.get_by_slug("deaify-text")
        assert agent is not None
        assert agent.name == "Text Humanizer"

    def test_get_by_slug_missing(self, two_agents: list[AgentDefinition]) -> None:
        registry = AgentRegistry(two_agents, [])
        assert registry.get_by_slug("nonexistent") is None

    def test_get_by_department(self, two_agents: list[AgentDefinition]) -> None:
        registry = AgentRegistry(two_agents, [])
        content_agents = registry.get_by_department("content")
        assert len(content_agents) == 1
        assert content_agents[0].slug == "deaify-text"


# ---------------------------------------------------------------------------
# AgentRouter tests
# ---------------------------------------------------------------------------


class TestAgentRouter:
    def test_webhook_matches_rule(self, router: AgentRouter) -> None:
        event = WebhookEvent(
            provider="github",
            event_type_name="issues",
            payload={"action": "opened"},
            delivery_id="d1",
        )
        agent = router.classify(event)
        assert agent is not None
        assert agent.slug == "competitor-analysis"

    def test_webhook_no_match(self, router: AgentRouter) -> None:
        event = WebhookEvent(
            provider="notion",
            event_type_name="page_updated",
            payload={},
            delivery_id="d2",
        )
        assert router.classify(event) is None

    def test_webhook_provider_mismatch(self, router: AgentRouter) -> None:
        event = WebhookEvent(
            provider="gitlab",
            event_type_name="issues",
            payload={},
            delivery_id="d3",
        )
        assert router.classify(event) is None

    def test_scheduled_with_agent_prefix(self, router: AgentRouter) -> None:
        event = ScheduledEvent(
            job_id="j1",
            job_name="daily-deaify",
            prompt="humanize this",
            skill_name="agent:deaify-text",
        )
        agent = router.classify(event)
        assert agent is not None
        assert agent.slug == "deaify-text"

    def test_scheduled_without_agent_prefix(self, router: AgentRouter) -> None:
        event = ScheduledEvent(
            job_id="j2",
            job_name="regular-job",
            prompt="do something",
            skill_name="diag",
        )
        assert router.classify(event) is None

    def test_scheduled_agent_disabled(
        self, two_agents: list[AgentDefinition]
    ) -> None:
        disabled = AgentDefinition(
            slug="off-agent",
            name="Off",
            department="test",
            prompt_file=Path("x.md"),
            enabled=False,
        )
        registry = AgentRegistry([*two_agents, disabled], [])
        router = AgentRouter(registry)
        event = ScheduledEvent(
            job_id="j3",
            job_name="test",
            prompt="x",
            skill_name="agent:off-agent",
        )
        assert router.classify(event) is None


# ---------------------------------------------------------------------------
# YAML loader tests
# ---------------------------------------------------------------------------


class TestLoadAgentRegistry:
    def test_loads_valid_yaml(self, tmp_path: Path) -> None:
        config = {
            "agents": [
                {
                    "slug": "test-agent",
                    "name": "Test",
                    "department": "testing",
                    "prompt_file": "prompts/test.md",
                },
            ],
        }
        p = tmp_path / "agents.yaml"
        p.write_text(yaml.dump(config), encoding="utf-8")

        registry = load_agent_registry(p)
        assert len(registry.agents) == 1
        assert registry.agents[0].slug == "test-agent"

    def test_rejects_missing_slug(self, tmp_path: Path) -> None:
        config = {
            "agents": [
                {"name": "No Slug", "department": "x", "prompt_file": "x.md"},
            ],
        }
        p = tmp_path / "agents.yaml"
        p.write_text(yaml.dump(config), encoding="utf-8")

        with pytest.raises(ValueError, match="missing 'slug'"):
            load_agent_registry(p)

    def test_rejects_duplicate_slug(self, tmp_path: Path) -> None:
        agent = {
            "slug": "dup",
            "name": "Dup",
            "department": "x",
            "prompt_file": "x.md",
        }
        config = {"agents": [agent, {**agent, "name": "Dup2"}]}
        p = tmp_path / "agents.yaml"
        p.write_text(yaml.dump(config), encoding="utf-8")

        with pytest.raises(ValueError, match="Duplicate agent slug"):
            load_agent_registry(p)

    def test_loads_routing_rules(self, tmp_path: Path) -> None:
        config = {
            "agents": [
                {
                    "slug": "a1",
                    "name": "A1",
                    "department": "d",
                    "prompt_file": "p.md",
                },
            ],
            "routing_rules": [
                {"agent": "a1", "provider": "github", "event_type": "push"},
            ],
        }
        p = tmp_path / "agents.yaml"
        p.write_text(yaml.dump(config), encoding="utf-8")

        registry = load_agent_registry(p)
        assert len(registry.routing_rules) == 1
        assert registry.routing_rules[0].provider == "github"
