from __future__ import annotations

import re

from own_agent.tools.types import ToolSpec


def think(thought: str = "", **kwargs) -> str:
    return "Thought recorded."


SPEC = ToolSpec(
    name="think",
    description="Record a reasoning step. Use this to think through complex tasks before taking action.",
    parameters={
        "type": "object",
        "properties": {
            "thought": {
                "type": "string",
                "description": "Your reasoning for the next action.",
            },
        },
        "required": ["thought"],
    },
    categories=("reasoning",),
)
