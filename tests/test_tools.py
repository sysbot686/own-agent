"""Tests for tools module."""

from own_agent.tools import ToolRegistry, register_all_tools, ToolResult, ExecutionContext


def test_all_tools_register():
    reg = ToolRegistry()
    register_all_tools(reg)
    names = [s.name for s in reg.list_specs()]
    expected = {"think", "ls", "view", "write", "edit", "grep", "glob", "shell", "python_exec"}
    assert expected.issubset(names), f"missing tools: {expected - set(names)}"


def test_think_tool():
    reg = ToolRegistry()
    register_all_tools(reg)
    r = reg.call("think", thought="testing")
    assert isinstance(r, ToolResult)
    assert r.success
    assert "recorded" in r.output


def test_think_no_args():
    reg = ToolRegistry()
    register_all_tools(reg)
    r = reg.call("think")
    assert isinstance(r, ToolResult)
    assert r.success


def test_ls_with_ctx():
    import tempfile
    tmp = tempfile.mkdtemp()
    ctx = ExecutionContext(cwd=tmp, workspace_root=tmp)
    reg = ToolRegistry()
    register_all_tools(reg)
    r = reg.call("ls", ctx=ctx)
    assert isinstance(r, ToolResult)
    assert r.success


def test_write_and_view():
    import tempfile, os
    tmp = tempfile.mkdtemp()
    ctx = ExecutionContext(cwd=tmp, workspace_root=tmp)
    reg = ToolRegistry()
    register_all_tools(reg)

    fpath = os.path.join(tmp, "test.txt")
    r = reg.call("write", ctx=ctx, file_path=fpath, content="hello\nworld")
    assert isinstance(r, ToolResult) and r.success

    r = reg.call("view", ctx=ctx, file_path=fpath)
    assert isinstance(r, ToolResult) and r.success
    assert "hello" in r.output


def test_write_outside_workspace():
    import tempfile
    tmp = tempfile.mkdtemp()
    ctx = ExecutionContext(cwd=tmp, workspace_root=tmp)
    reg = ToolRegistry()
    register_all_tools(reg)
    r = reg.call("write", ctx=ctx, file_path="/outside.txt", content="test")
    assert isinstance(r, ToolResult) and not r.success
    assert "outside workspace" in (r.error or "")


def test_edit_tool():
    import tempfile, os
    tmp = tempfile.mkdtemp()
    ctx = ExecutionContext(cwd=tmp, workspace_root=tmp)
    reg = ToolRegistry()
    register_all_tools(reg)

    fpath = os.path.join(tmp, "test.txt")
    reg.call("write", ctx=ctx, file_path=fpath, content="hello world")
    r = reg.call("edit", ctx=ctx, file_path=fpath, old_string="world", new_string="there")
    assert isinstance(r, ToolResult) and r.success

    r = reg.call("view", ctx=ctx, file_path=fpath)
    assert "there" in r.output


def test_grep_tool():
    import tempfile, os
    tmp = tempfile.mkdtemp()
    ctx = ExecutionContext(cwd=tmp, workspace_root=tmp)
    reg = ToolRegistry()
    register_all_tools(reg)

    fpath = os.path.join(tmp, "test.py")
    reg.call("write", ctx=ctx, file_path=fpath, content="def foo():\n    pass")

    r = reg.call("grep", ctx=ctx, pattern="def foo", path=tmp)
    assert isinstance(r, ToolResult) and r.success
    assert "test.py" in r.output


def test_unknown_tool():
    reg = ToolRegistry()
    register_all_tools(reg)
    r = reg.call("nonexistent")
    assert isinstance(r, ToolResult) and not r.success


def test_shell_blocked():
    from own_agent.tools.shell import shell
    r = shell(command="rm -rf /", description="test")
    assert r.startswith("Blocked")


def test_python_exec():
    from own_agent.tools.shell import python_exec
    r = python_exec(code='print("hello")')
    assert "hello" in r

    r = python_exec(code='_result = 42')
    assert "42" in r


def test_python_exec_restricted():
    from own_agent.tools.shell import python_exec
    r = python_exec(code='__import__("os")')
    assert "not allowed" in r


def test_python_exec_restricted_import():
    from own_agent.tools.shell import python_exec
    r = python_exec(code='import os')
    assert "not allowed" in r


def test_replace_all_in_schema():
    reg = ToolRegistry()
    register_all_tools(reg)
    for s in reg.list_specs():
        if s.name == "edit":
            assert "replace_all" in s.parameters.get("properties", {})
            return
    assert False, "edit tool not found"
