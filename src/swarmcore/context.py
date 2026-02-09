from __future__ import annotations


class SharedContext:
    """Shared key-value store passed between agents in a swarm run."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    def set(self, key: str, value: str) -> None:
        self._data[key] = value

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def format_for_prompt(self) -> str:
        """Render all context entries as markdown sections."""
        if not self._data:
            return ""
        sections = []
        for name, content in self._data.items():
            sections.append(f"## {name}\n{content}")
        return "\n\n".join(sections)

    def to_dict(self) -> dict[str, str]:
        return dict(self._data)

    def __repr__(self) -> str:
        return f"SharedContext({self._data!r})"
