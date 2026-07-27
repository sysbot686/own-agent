"""Tests for permissions module."""

from own_agent.permissions import PermissionManager, PermissionMode


def test_bypass():
    pm = PermissionManager(mode=PermissionMode.BYPASS)
    r = pm.request("write", "write dangerous_file.txt")
    assert r.approved
    assert r.reason == "bypass mode"


def test_lenient_non_sensitive():
    pm = PermissionManager(mode=PermissionMode.LENIENT)
    r = pm.request("ls", "list directory")
    assert r.approved
    assert "lenient" in r.reason


def test_lenient_sensitive_no_callback():
    pm = PermissionManager(mode=PermissionMode.LENIENT)
    r = pm.request("write", "write test.txt")
    assert not r.approved
    assert "no approval" in r.reason


def test_standard_no_callback_non_sensitive():
    pm = PermissionManager(mode=PermissionMode.STANDARD)
    r = pm.request("anything", "details")
    assert r.approved
    assert "non-sensitive" in r.reason


def test_standard_no_callback_sensitive():
    pm = PermissionManager(mode=PermissionMode.STANDARD)
    r = pm.request("write", "write test.txt")
    assert not r.approved
    assert "no approval" in r.reason


def test_standard_with_callback():
    def cb(action, details, mode):
        return True
    pm = PermissionManager(mode=PermissionMode.STANDARD, on_request=cb)
    r = pm.request("write", "write test.txt")
    assert r.approved


def test_aggressive_sensitive():
    def cb(action, details, mode):
        return True
    pm = PermissionManager(mode=PermissionMode.AGGRESSIVE, on_request=cb)
    r = pm.request("shell", "run command")
    assert r.approved


def test_aggressive_non_sensitive_needs_callback():
    pm = PermissionManager(mode=PermissionMode.AGGRESSIVE)
    r = pm.request("ls", "list dir")
    assert not r.approved
    assert "no approval" in r.reason


def test_needs_approval():
    assert PermissionManager(mode=PermissionMode.STANDARD).needs_approval
    assert PermissionManager(mode=PermissionMode.AGGRESSIVE).needs_approval
    assert not PermissionManager(mode=PermissionMode.BYPASS).needs_approval
    assert not PermissionManager(mode=PermissionMode.LENIENT).needs_approval


def test_make_callback():
    def cb(action, details, mode):
        return True
    pm = PermissionManager(mode=PermissionMode.STANDARD, on_request=cb)
    callback = pm.make_callback()
    assert callback("write", "test.txt")
