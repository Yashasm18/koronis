"""The command line has to survive being typed at wrongly.

Defect 31: dispatch was a bare dict lookup, so `--help` and any typo exited on a
raw `KeyError` traceback. Every experiment ran correctly -- the failure was only
at the front door, which is exactly where a reader who has just cloned the repo
arrives first. Nothing in the suite covered the entry point, because every test
imported the functions directly and never went through `__main__`.

The last test here is the one that keeps rotting away: the docs name commands in
copy-pasteable lines, and a renamed experiment would leave those lines pointing
at nothing while every other test still passed.
"""
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def run(*args):
    return subprocess.run([sys.executable, "-m", "koronis.cli", *args],
                          capture_output=True, text=True, cwd=ROOT, timeout=120)


def test_help_exits_cleanly_and_lists_the_experiments():
    for flag in ("-h", "--help", "help"):
        out = run(flag)
        assert out.returncode == 0, f"{flag} exited {out.returncode}: {out.stderr[:300]}"
        assert "usage:" in out.stdout
        assert "ablation" in out.stdout and "frontier" in out.stdout
        assert "Traceback" not in out.stderr


def test_an_unknown_experiment_is_refused_by_name_not_by_traceback():
    out = run("no_such_experiment")
    assert out.returncode == 2, f"expected exit 2, got {out.returncode}"
    assert "no_such_experiment" in out.stderr
    assert "usage:" in out.stderr
    assert "Traceback" not in out.stderr, "the entry point still raises rather than reports"


def test_every_command_the_docs_tell_a_reader_to_run_exists():
    docs = [ROOT / "README.md", ROOT / "CONTRIBUTING.md"] + list((ROOT / "docs").glob("*.md"))
    named = set()
    for d in docs:
        named |= set(re.findall(r"koronis\.cli\s+([a-z_]+)", d.read_text(encoding="utf-8")))
    assert named, "no CLI commands are documented anywhere"
    listed = set(run("--help").stdout.split("experiments:")[1].split())
    missing = sorted(named - listed)
    assert not missing, f"documented but not dispatchable: {missing}"
