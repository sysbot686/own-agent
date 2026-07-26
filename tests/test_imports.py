"""Import smoke tests."""


def test_import_core():
    import own_agent.cli
    import own_agent.config
    import own_agent.providers
    import own_agent.tools
    import own_agent.session
    import own_agent.context
    import own_agent.permissions
    import own_agent.agent


def test_import_providers():
    from own_agent.providers.openai_compatible import OpenAICompatibleProvider
    from own_agent.providers.anthropic import AnthropicProvider
    from own_agent.providers.types import ChatMessage, TokenUsage


def test_import_tools():
    from own_agent.tools import ToolRegistry, register_all_tools, ToolResult, ToolSpec
    from own_agent.tools.context import ExecutionContext


def test_import_agent():
    from own_agent.agent.loop import Agent
    from own_agent.agent.types import AgentConfig
