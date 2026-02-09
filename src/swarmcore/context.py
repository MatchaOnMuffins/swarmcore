from __future__ import annotations


class SharedContext:
    """Shared key-value store passed between agents in a swarm run."""

    def __init__(self) -> None:
        self._full: dict[str, str] = {}
        self._summaries: dict[str, str] = {}

    def set(self, key: str, value: str, *, summary: str | None = None) -> None:
        self._full[key] = value
        self._summaries[key] = summary if summary is not None else value

    def get(self, key: str) -> str | None:
        return self._full.get(key)

    def get_summary(self, key: str) -> str | None:
        return self._summaries.get(key)

    def format_for_prompt(self, *, expand: set[str] | None = None) -> str:
        """Render context entries as markdown sections.

        When *expand* is provided, keys in the set show full output;
        all other keys show summaries. When *expand* is ``None``
        (the default), every key shows its full output for backward
        compatibility.
        """
        if not self._full:
            return ""
        sections = []
        for name in self._full:
            if expand is None or name in expand:
                content = self._full[name]
                sections.append(f"## {name}\n{content}")
            else:
                content = self._summaries[name]
                sections.append(f"## {name} (summary)\n{content}")
        return "\n\n".join(sections)

    def to_dict(self) -> dict[str, str]:
        return dict(self._full)

    def __repr__(self) -> str:
        return f"SharedContext({self._full!r})"
