from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Skill:
    name: str
    description: str = ""
    content: str = ""
    commands: list[str] = field(default_factory=list)


class SkillLoader:
    def __init__(self, skill_dirs: list[Path] | None = None) -> None:
        self._dirs = [Path(d) for d in skill_dirs] if skill_dirs else []

    def list_skills(self) -> list[Skill]:
        skills: list[Skill] = []
        for sd in self._dirs:
            if not sd.exists():
                continue
            for f in sd.iterdir():
                if f.suffix in (".md", ".txt"):
                    skills.append(self._load_skill(f))
        return skills

    def load(self, name: str) -> Skill | None:
        for sd in self._dirs:
            if not sd.exists():
                continue
            for f in sd.iterdir():
                if f.stem == name and f.suffix in (".md", ".txt"):
                    return self._load_skill(f)
        return None

    @staticmethod
    def _load_skill(path: Path) -> Skill:
        text = path.read_text(encoding="utf-8")
        name = path.stem
        lines = text.splitlines()
        description = ""
        commands: list[str] = []
        in_cmds = False
        for line in lines:
            if line.startswith("# ") and not description:
                description = line[2:].strip()
            elif line.lower().startswith("commands:") or line.lower().startswith("aliases:"):
                in_cmds = True
            elif in_cmds and line.startswith("- "):
                commands.append(line[2:].strip())
            elif in_cmds and not line.startswith(" "):
                in_cmds = False
        return Skill(name=name, description=description, content=text, commands=commands)
