"""Event handlers that bridge the event bus to Claude and Telegram.

AgentHandler: translates events into ClaudeIntegration.run_command() calls.
NotificationHandler: subscribes to AgentResponseEvent and delivers to Telegram.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from ..agents.registry import AgentDefinition
from ..agents.router import AgentRouter
from ..claude.facade import ClaudeIntegration
from .bus import Event, EventBus
from .types import AgentResponseEvent, ScheduledEvent, WebhookEvent

logger = structlog.get_logger()


class AgentHandler:
    """Translates incoming events into Claude agent executions.

    Webhook and scheduled events are converted into prompts and sent
    to ClaudeIntegration.run_command(). The response is published
    back as an AgentResponseEvent for delivery.

    When an AgentRouter is configured, events are classified first
    and the matched agent's system prompt is prepended automatically.
    """

    def __init__(
        self,
        event_bus: EventBus,
        claude_integration: ClaudeIntegration,
        default_working_directory: Path,
        default_user_id: int = 0,
        agent_router: Optional[AgentRouter] = None,
        project_base_dir: Optional[Path] = None,
    ) -> None:
        self.event_bus = event_bus
        self.claude = claude_integration
        self.default_working_directory = default_working_directory
        self.default_user_id = default_user_id
        self.agent_router = agent_router
        self.project_base_dir = project_base_dir or default_working_directory

    def register(self) -> None:
        """Subscribe to events that need agent processing."""
        self.event_bus.subscribe(WebhookEvent, self.handle_webhook)
        self.event_bus.subscribe(ScheduledEvent, self.handle_scheduled)

    async def handle_webhook(self, event: Event) -> None:
        """Process a webhook event through Claude."""
        if not isinstance(event, WebhookEvent):
            return

        agent = self._classify(event)

        logger.info(
            "Processing webhook event through agent",
            provider=event.provider,
            event_type=event.event_type_name,
            delivery_id=event.delivery_id,
            agent=agent.slug if agent else None,
        )

        prompt = self._build_webhook_prompt(event)
        prompt = self._prepend_agent_prompt(prompt, agent)
        working_dir = self._agent_working_dir(agent)

        try:
            response = await self.claude.run_command(
                prompt=prompt,
                working_directory=working_dir,
                user_id=self.default_user_id,
            )

            if response.content:
                await self.event_bus.publish(
                    AgentResponseEvent(
                        chat_id=0,
                        text=response.content,
                        originating_event_id=event.id,
                    )
                )
        except Exception:
            logger.exception(
                "Agent execution failed for webhook event",
                provider=event.provider,
                event_id=event.id,
            )

    async def handle_scheduled(self, event: Event) -> None:
        """Process a scheduled event through Claude."""
        if not isinstance(event, ScheduledEvent):
            return

        agent = self._classify(event)

        logger.info(
            "Processing scheduled event through agent",
            job_id=event.job_id,
            job_name=event.job_name,
            agent=agent.slug if agent else None,
        )

        prompt = event.prompt
        if event.skill_name and not (
            event.skill_name and event.skill_name.startswith("agent:")
        ):
            prompt = (
                f"/{event.skill_name}\n\n{prompt}" if prompt else f"/{event.skill_name}"
            )

        prompt = self._prepend_agent_prompt(prompt, agent)
        working_dir = (
            self._agent_working_dir(agent)
            if agent
            else (event.working_directory or self.default_working_directory)
        )

        try:
            response = await self.claude.run_command(
                prompt=prompt,
                working_directory=working_dir,
                user_id=self.default_user_id,
            )

            if response.content:
                for chat_id in event.target_chat_ids:
                    await self.event_bus.publish(
                        AgentResponseEvent(
                            chat_id=chat_id,
                            text=response.content,
                            originating_event_id=event.id,
                        )
                    )

                if not event.target_chat_ids:
                    await self.event_bus.publish(
                        AgentResponseEvent(
                            chat_id=0,
                            text=response.content,
                            originating_event_id=event.id,
                        )
                    )
        except Exception:
            logger.exception(
                "Agent execution failed for scheduled event",
                job_id=event.job_id,
                event_id=event.id,
            )

    # ------------------------------------------------------------------
    # Agent routing helpers
    # ------------------------------------------------------------------

    def _classify(self, event: Event) -> Optional[AgentDefinition]:
        """Ask the router for the best agent (or None)."""
        if self.agent_router:
            return self.agent_router.classify(event)
        return None

    def _prepend_agent_prompt(
        self, prompt: str, agent: Optional[AgentDefinition]
    ) -> str:
        """Prepend agent system prompt if an agent matched."""
        if not agent:
            return prompt
        try:
            system_prompt = agent.load_prompt(self.project_base_dir)
            return f"{system_prompt}\n\n---\n\n{prompt}"
        except FileNotFoundError:
            logger.warning(
                "Agent prompt file missing, using plain prompt",
                agent=agent.slug,
                prompt_file=str(agent.prompt_file),
            )
            return prompt

    def _agent_working_dir(self, agent: Optional[AgentDefinition]) -> Path:
        """Resolve working directory for the agent."""
        if agent and agent.working_directory:
            return self.project_base_dir / agent.working_directory
        return self.default_working_directory

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_webhook_prompt(self, event: WebhookEvent) -> str:
        """Build a Claude prompt from a webhook event."""
        payload_summary = self._summarize_payload(event.payload)

        return (
            f"A {event.provider} webhook event occurred.\n"
            f"Event type: {event.event_type_name}\n"
            f"Payload summary:\n{payload_summary}\n\n"
            f"Analyze this event and provide a concise summary. "
            f"Highlight anything that needs my attention."
        )

    def _summarize_payload(self, payload: Dict[str, Any], max_depth: int = 2) -> str:
        """Create a readable summary of a webhook payload."""
        lines: List[str] = []
        self._flatten_dict(payload, lines, max_depth=max_depth)
        # Cap at 2000 chars to keep prompt reasonable
        summary = "\n".join(lines)
        if len(summary) > 2000:
            summary = summary[:2000] + "\n... (truncated)"
        return summary

    def _flatten_dict(
        self,
        data: Any,
        lines: list,
        prefix: str = "",
        depth: int = 0,
        max_depth: int = 2,
    ) -> None:
        """Flatten a nested dict into key: value lines."""
        if depth >= max_depth:
            lines.append(f"{prefix}: ...")
            return

        if isinstance(data, dict):
            for key, value in data.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, (dict, list)):
                    self._flatten_dict(value, lines, full_key, depth + 1, max_depth)
                else:
                    val_str = str(value)
                    if len(val_str) > 200:
                        val_str = val_str[:200] + "..."
                    lines.append(f"{full_key}: {val_str}")
        elif isinstance(data, list):
            lines.append(f"{prefix}: [{len(data)} items]")
            for i, item in enumerate(data[:3]):  # Show first 3 items
                self._flatten_dict(item, lines, f"{prefix}[{i}]", depth + 1, max_depth)
        else:
            lines.append(f"{prefix}: {data}")
