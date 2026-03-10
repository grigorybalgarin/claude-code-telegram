"""Agent router — classifies events and selects the right agent.

Tries routing rules top-down (first match wins).  Falls back to
``None`` so the caller can use the default (un-routed) behaviour.
"""

from typing import Optional

import structlog

from ..events.bus import Event
from ..events.types import ScheduledEvent, WebhookEvent
from .registry import AgentDefinition, AgentRegistry

logger = structlog.get_logger()


class AgentRouter:
    """Select an agent for an incoming event."""

    def __init__(self, registry: AgentRegistry) -> None:
        self.registry = registry

    def classify(self, event: Event) -> Optional[AgentDefinition]:
        """Return the best-matching enabled agent, or ``None``."""

        # 1. ScheduledEvent may carry an explicit agent_slug
        if isinstance(event, ScheduledEvent):
            agent = self._from_scheduled(event)
            if agent:
                return agent

        # 2. WebhookEvent — try routing rules (provider + event_type)
        if isinstance(event, WebhookEvent):
            agent = self._from_webhook_rules(event)
            if agent:
                return agent

        # 3. No match
        logger.debug(
            "No agent matched for event",
            event_type=event.event_type,
            event_id=event.id,
        )
        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _from_scheduled(self, event: ScheduledEvent) -> Optional[AgentDefinition]:
        """Check if the scheduled event specifies an agent slug via skill_name."""
        # Convention: if skill_name starts with "agent:" treat the rest as slug.
        if event.skill_name and event.skill_name.startswith("agent:"):
            slug = event.skill_name.removeprefix("agent:").strip()
            agent = self.registry.get_by_slug(slug)
            if agent and agent.enabled:
                return agent
            logger.warning("Scheduled event references unknown/disabled agent", slug=slug)
        return None

    def _from_webhook_rules(self, event: WebhookEvent) -> Optional[AgentDefinition]:
        """Walk routing rules and find the first match."""
        for rule in self.registry.routing_rules:
            if rule.provider and rule.provider != event.provider:
                continue
            if rule.event_type and rule.event_type != event.event_type_name:
                continue
            agent = self.registry.get_by_slug(rule.agent)
            if agent and agent.enabled:
                logger.info(
                    "Routed webhook to agent",
                    provider=event.provider,
                    event_type=event.event_type_name,
                    agent=agent.slug,
                )
                return agent
        return None
