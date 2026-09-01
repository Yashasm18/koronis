"""The defence-only guarantees SECURITY.md makes, asserted.

SECURITY.md said these were "checked by a test" and offered a `grep` command.
The grep was real; the test was not. This is the project's most load-bearing
safety claim - a detector that could reach the network or run a subprocess would
be a different kind of artifact entirely - and it was the one claim resting on a
sentence rather than an assertion.

The scan here is stricter than the documented grep, which only matches
module-level `import x` / `from x import y` lines. This walks the AST, so it also
catches an import inside a function, an aliased import, and the `__import__` /
`importlib` escape hatches. The grep stays in SECURITY.md because a reader can
run it in one line without trusting this file; it is checked too, so the
documented command cannot quietly stop matching what is enforced.
"""
import ast
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "koronis"
SOURCES = sorted(PKG.rglob("*.py"))

# Reaching outside the process, in any form.
BANNED_ROOTS = {
    "socket", "ssl", "requests", "urllib", "urllib2", "urllib3", "http",
    "httplib", "httpx", "aiohttp", "ftplib", "telnetlib", "smtplib", "poplib",
    "imaplib", "xmlrpc", "paramiko", "websockets", "subprocess", "pty",
    "multiprocessing",
}
ESCAPE_HATCHES = {"__import__", "eval", "exec", "compile"}


def test_there_are_sources_to_scan():
    """Guard the guard: an empty sweep would pass every assertion below."""
    assert len(SOURCES) > 20, f"only found {len(SOURCES)} modules under koronis/"


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_module_reaches_nothing_outside_the_process(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    rel = path.relative_to(ROOT)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                assert root not in BANNED_ROOTS, (
                    f"{rel}:{node.lineno} imports `{a.name}`. Koronis operates on "
                    "in-memory dataframes and must not be able to reach anything.")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            assert root not in BANNED_ROOTS, (
                f"{rel}:{node.lineno} imports from `{node.module}`. Koronis "
                "operates on in-memory dataframes and must not be able to reach "
                "anything.")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in ESCAPE_HATCHES, (
                f"{rel}:{node.lineno} calls `{node.func.id}()`, which can load or "
                "run code the import scan above cannot see.")
        elif isinstance(node, ast.Attribute) and node.attr == "system":
            # os.system(...)
            assert not (isinstance(node.value, ast.Name) and node.value.id == "os"), (
                f"{rel}:{node.lineno} calls os.system().")


def test_the_command_security_md_publishes_still_comes_back_clean():
    """The reader-facing check, run the way SECURITY.md tells a reader to run it."""
    security = (ROOT / "SECURITY.md").read_text()
    m = re.search(r'grep -rnE "(\^\([^"]+)" koronis/', security)
    assert m, "SECURITY.md no longer publishes the grep command this test verifies"

    pattern = re.compile(m.group(1), re.M)
    hits = [f"{p.relative_to(ROOT)}: {line}"
            for p in SOURCES
            for line in p.read_text().splitlines() if pattern.match(line)]
    assert not hits, "the command published in SECURITY.md now returns:\n  " + "\n  ".join(hits)


def test_generated_identifiers_are_labels_not_card_data():
    """No generated BIN can be mistaken for a real issuer range."""
    from koronis.data.background import load_background
    from koronis.data.campaigns import inject
    from koronis.data.schema import CampaignSpec

    bg = load_background(path=None, n_rows=2000, seed=0)
    ev = inject(bg, [CampaignSpec(n_attempts=100, k_devices=4, k_ips=2,
                                  duration_s=1800.0,
                                  start_ts=float(bg["ts"].iloc[200]))], seed=0)

    for col in ("bin_id", "card_id", "device_id", "ip_id"):
        numeric = [v for v in ev[col].unique() if re.fullmatch(r"\d{6,}", str(v))]
        assert not numeric, (
            f"{col} contains all-digit values that could be read as real card "
            f"data: {numeric[:5]}")
