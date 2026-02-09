from __future__ import annotations

import re


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

    def keys(self) -> list[str]:
        """Return all entry keys in insertion order."""
        return list(self._full.keys())

    def search(self, pattern: str) -> dict[str, list[str]]:
        """Search across all full entries, returning matching lines by agent name.

        Falls back to substring matching if *pattern* is not a valid regex.
        """
        try:
            regex = re.compile(pattern)
        except re.error:
            regex = re.compile(re.escape(pattern))

        results: dict[str, list[str]] = {}
        for key, value in self._full.items():
            matches = [line for line in value.splitlines() if regex.search(line)]
            if matches:
                results[key] = matches
        return results

    def entries(self) -> list[tuple[str, str, str, int]]:
        """Return ``(key, summary, full, char_count)`` tuples for all entries."""
        return [
            (key, self._summaries[key], self._full[key], len(self._full[key]))
            for key in self._full
        ]

    def to_dict(self) -> dict[str, str]:
        return dict(self._full)

    def __repr__(self) -> str:
        return f"SharedContext({self._full!r})"
