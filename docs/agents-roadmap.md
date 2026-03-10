# Multi-Agent System Roadmap

## Current Phase: 3 — VIDEO + SALES departments
## Status: DONE (pending deploy + e2e test)

---

## Phase 1: Framework + 2 agents (CONTENT + R&D)

Goal: build the routing infrastructure so that adding a new agent = 1 YAML entry + 1 prompt file.

- [x] `config/agents.yaml` — agent registry (slug, name, department, prompt_file, working_directory, tools_allowlist)
- [x] `src/agents/__init__.py`
- [x] `src/agents/registry.py` — `AgentDefinition` dataclass + `load_agent_registry()`
- [x] `src/agents/router.py` — `AgentRouter.classify(event) -> agent_slug` with dept-label routing + fallback
- [x] Extend `AgentHandler` in `src/events/handlers.py` — inject AgentRouter, select prompt per agent before `run_command()`
- [x] `prompts/deaify-text.md` — system prompt for text humanization agent
- [x] `prompts/competitor-analysis.md` — system prompt for competitor research agent
- [x] Tests for AgentRouter (`tests/unit/test_agent_router.py`) — 15 tests passing
- [x] Wire AgentRouter into bot startup (`src/main.py`) — loads `config/agents.yaml` at init, injects into AgentHandler
- [ ] End-to-end: webhook -> route -> execute -> notify

## Phase 2: ANALYTICS department (3 agents)

Goal: automated insights delivered on schedule.

- [x] `prompts/cc-analytics.md` — Claude cost analytics agent (weekly report from cost_tracking table)
- [x] `prompts/chat-trends.md` — chat trend analysis agent
- [x] `prompts/blog-ideas.md` — content idea generation agent
- [x] Add `department` column to `cost_tracking` table (migration 8) + `CostTrackingModel.department` field
- [x] `get_costs_by_department()` method in `CostTrackingRepository`
- [x] 3 agents registered in `config/agents.yaml` (cc-analytics, chat-trends, blog-ideas)
- [ ] Cron jobs in scheduler for weekly analytics (requires deploy + configuration)

## Phase 3: VIDEO + SALES departments (3-4 agents)

Goal: content production and distribution automation.

- [x] `prompts/youtube-script.md` — video script generation agent (hook, sections, CTA, SEO tags)
- [x] `prompts/tg-broadcast.md` — Telegram broadcast agent (A/B variants, send time, inline buttons)
- [x] `prompts/deeplink-creator.md` — UTM link creator for YouTube/Telegram/Instagram cross-tracking
- [x] 3 agents registered in `config/agents.yaml` (youtube-script, tg-broadcast, deeplink-creator)
- [ ] Telegram command shortcuts for these agents (requires deploy + wiring)

## Phase 4: PRODUCT + supporting services (later)

Goal: shared knowledge layer across all departments. Requires separate architecture.

- [ ] Knowledge base design (storage schema, indexing, retrieval API)
- [ ] `src/agents/knowledge_base.py` — KB service
- [ ] Cohorts / knowledge graph (entity relationships)
- [ ] `prompts/product-name-normalizer.md` — depends on populated KB

---

## Architecture Decisions

Record key decisions here so they don't get revisited.

| # | Decision | Rationale | Date |
|---|----------|-----------|------|
| 1 | Agent config in YAML, not DB | Simple, version-controlled, no migration needed | 2026-03-10 |
| 2 | Prompts in `prompts/*.md` files | Easy to edit, git-tracked, readable | 2026-03-10 |
| 3 | Router in EventBus layer, not middleware | Events already typed, handler pattern fits | 2026-03-10 |
| 4 | Phase 4 deferred | KB needs separate architecture, not just prompts | 2026-03-10 |

## Notes

- 70-80% of infrastructure already exists (EventBus, Scheduler, Webhooks, Notifications, cost_tracking)
- Main work per agent = prompt engineering + YAML config entry
- Each new agent after Phase 1 should take ~15-30 minutes to add
