"""Keep internal planning material outside public documentation."""

from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def public_documents():
    if (ROOT / ".git").exists():
        names = subprocess.check_output(
            [
                "git",
                "-C",
                str(ROOT),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "--",
                "*.md",
            ],
            text=True,
        ).splitlines()
        return sorted({ROOT / name for name in names if (ROOT / name).is_file()})
    paths = set(ROOT.glob("*.md"))
    for directory in ("docs", "packaging", ".github", ".claude"):
        paths.update((ROOT / directory).rglob("*.md"))
    return sorted(
        p
        for p in paths
        if "llm_wiki" not in p.parts
        and p.name not in {"AGENTS.md", "PUBLISH_PREPARATION_PLAN.md"}
    )


def test_public_documents_do_not_contain_internal_plans_or_task_ids():
    for path in public_documents():
        relative = path.relative_to(ROOT).as_posix()
        assert not relative.startswith(
            (".publication/", "docs/llm_wiki/", ".claude/")
        ), relative
        assert not re.search(
            r"(?i)(?:backlog|implementation[-_ ]|feasibility|spike|/adr/|phase[-_ ]?\d|^design\.md|_PLAN\.md)",
            relative,
        ), relative
        text = path.read_text(encoding="utf-8")
        assert not re.search(
            r"\b(?:MATTE|VIS|WIN|MIT|PUB|REL|CC)-\d+(?:\.\d+)*", text
        ), relative
        assert not re.search(
            r"(?im)^#{1,6}\s+.*(?:\bphase\s+\d|implementation (?:plan|review)|delivery roadmap|backlog)",
            text,
        ), relative


def test_public_markdown_links_resolve_to_present_files():
    for path in public_documents():
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^\s)]+)\)", text):
            if "://" in target or target.startswith(("#", "mailto:")):
                continue
            file_part = target.split("#", 1)[0]
            assert (path.parent / file_part).exists(), (
                str(path.relative_to(ROOT)),
                target,
            )


def test_distributed_configuration_help_has_no_internal_task_ids():
    for name in (
        "config/default.yaml",
        "config/avatar.yaml",
        "src/custback/default.yaml",
        "src/custback/avatar/avatar.yaml",
    ):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert not re.search(r"\b(?:MATTE|VIS|WIN|MIT|PUB)-\d", text), name
