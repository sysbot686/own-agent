from __future__ import annotations

from own_agent.tools.types import ToolSpec


def plan(plan_text: str = "", **kwargs) -> str:
    return "Plan saved."


PLAN_SPEC = ToolSpec(
    name="plan",
    description="Create or update a plan for the current task. Break down complex tasks into steps. Mark progress with - [x] done / - [ ] pending.",
    parameters={
        "type": "object",
        "properties": {
            "plan_text": {
                "type": "string",
                "description": "The plan in markdown checklist format.",
            },
        },
        "required": ["plan_text"],
    },
    categories=("reasoning",),
)
