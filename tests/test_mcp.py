"""Tests for MCP client."""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

import pytest

from own_agent.mcp.client import McpClient, McpServerConfig
from own_agent.tools.types import ToolSpec


def _build_mock_server_script() -> str:
    return """\
import json, sys

def handle(line):
    msg = json.loads(line)
    req_id = msg.get("id")
    method = msg.get("method")
    params = msg.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {
                "protocolVersion": "2024-11-20",
                "capabilities": {"tools": {}},
            },
        }
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "echo",
                        "description": "Echo back the input",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                    }
                ],
            },
        }
    elif method == "tools/call":
        name = params.get("name", "")
        args = params.get("arguments", {})
        if name == "echo":
            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": args.get("text", "")}],
                },
            }
        return {
            "jsonrpc": "2.0", "id": req_id,
            "error": {"code": -32601, "message": f"Tool not found: {name}"},
        }
    else:
        return {
            "jsonrpc": "2.0", "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    sys.stdout.write(json.dumps(handle(line)) + "\\n")
    sys.stdout.flush()
"""


@pytest.fixture
def mock_server_script():
    tmp = tempfile.mkdtemp()
    path = Path(tmp) / "mock_mcp_server.py"
    path.write_text(_build_mock_server_script(), encoding="utf-8")
    return str(path)


@pytest.mark.asyncio
async def test_mcp_initialize(mock_server_script):
    config = McpServerConfig(command=sys.executable, args=[mock_server_script])
    client = McpClient(config)
    await client.start()
    try:
        result = await client.initialize()
        assert "protocolVersion" in result
        assert result["protocolVersion"] == "2024-11-20"
        assert "capabilities" in result
    finally:
        await client.stop()


@pytest.mark.asyncio
async def test_mcp_list_tools(mock_server_script):
    config = McpServerConfig(command=sys.executable, args=[mock_server_script])
    client = McpClient(config)
    await client.start()
    try:
        await client.initialize()
        tools = await client.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "echo"
        assert tools[0].description == "Echo back the input"
        assert "text" in tools[0].parameters.get("properties", {})
    finally:
        await client.stop()


@pytest.mark.asyncio
async def test_mcp_call_tool(mock_server_script):
    config = McpServerConfig(command=sys.executable, args=[mock_server_script])
    client = McpClient(config)
    await client.start()
    try:
        await client.initialize()
        result = await client.call_tool("echo", {"text": "hello mcp"})
        assert result == "hello mcp"
    finally:
        await client.stop()


@pytest.mark.asyncio
async def test_mcp_connected(mock_server_script):
    config = McpServerConfig(command=sys.executable, args=[mock_server_script])
    client = McpClient(config)
    assert not client.connected
    await client.start()
    assert client.connected
    await client.stop()
    assert not client.connected


@pytest.mark.asyncio
async def test_mcp_not_connected_error():
    config = McpServerConfig(command="nonexistent-command")
    client = McpClient(config)
    with pytest.raises(Exception):
        await client.list_tools()


@pytest.mark.asyncio
async def test_mcp_stop_cleanup(mock_server_script):
    config = McpServerConfig(command=sys.executable, args=[mock_server_script])
    client = McpClient(config)
    await client.start()
    await client.stop()
    # Stopping twice should be safe
    await client.stop()
