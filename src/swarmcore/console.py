from __future__ import annotations

import sys
from typing import TextIO

from swarmcore.hooks import Event, EventType, Hooks

# ANSI escape sequences
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


class ConsoleReporter:
    """Hook handler that prints formatted, colored execution progress to the terminal.

    Drop-in replacement for the hand-rolled ``_handle_event`` pattern
    commonly copy-pasted across example scripts.

    Parameters
    ----------
    color:
        Enable ANSI color output. Set to ``False`` for plain text.
    verbose:
        When ``True``, shows tool call arguments and result previews.
        When ``False``, shows just tool name and duration.
    file:
        Output stream. Defaults to ``sys.stderr`` to keep ``stdout``
        clean for piping ``result.output``.
    """

    def __init__(
        self,
        *,
        color: bool = True,
        verbose: bool = False,
        file: TextIO | None = None,
    ) -> None:
        self._color = color
        self._verbose = verbose
        self._file = file or sys.stderr

    # -- Color helpers ------------------------------------------------

    def _c(self, code: str, text: str) -> str:
        if not self._color:
            return text
        return f"{code}{text}{_RESET}"

    def _bold(self, text: str) -> str:
        return self._c(_BOLD, text)

    def _dim(self, text: str) -> str:
        return self._c(_DIM, text)

    def _green(self, text: str) -> str:
        return self._c(_GREEN, text)

    def _yellow(self, text: str) -> str:
        return self._c(_YELLOW, text)

    def _red(self, text: str) -> str:
        return self._c(_RED, text)

    # -- Output helpers -----------------------------------------------

    def _print(self, text: str = "", *, end: str = "\n", flush: bool = False) -> None:
        print(text, end=end, flush=flush, file=self._file)

    def _section(self, title: str) -> None:
        rule = "\u2500" * 70
        self._print(f"\n{self._bold(rule)}")
        self._print(f"  {self._bold(title)}")
        self._print(self._bold(rule))

    # -- Event dispatch -----------------------------------------------

    def __call__(self, event: Event) -> None:
        d = event.data
        match event.type:
            case EventType.SWARM_START:
                pass  # section headers come from STEP_START
            case EventType.SWARM_END:
                dur = d.get("duration_seconds", "?")
                count = d.get("agent_count", "?")
                total_cost = d.get("total_cost", 0.0)
                cost_part = ""
                if total_cost:
                    cost_part = ", $" + f"{total_cost:.4f}"
                self._print(
                    "\n  "
                    + self._bold("Done")
                    + " \u2014 "
                    + str(count)
                    + " agents, "
                    + str(dur)
                    + "s total"
                    + cost_part
                )
            case EventType.STEP_START:
                agents = d.get("agents", [])
                par = d.get("parallel", False)
                label = " | ".join(agents) if par else (agents[0] if agents else "?")
                suffix = " (parallel)" if par else ""
                self._section(f"Step {d.get('step_index', 0) + 1}: {label}{suffix}")
            case EventType.STEP_END:
                pass
            case EventType.AGENT_START:
                agent = d.get("agent", "?")
                self._print(f"\n  {self._bold('▶ ' + agent)}")
            case EventType.LLM_CALL_START:
                idx = d.get("call_index", "?")
                self._print(
                    f"  {self._dim('LLM call ' + str(idx) + '...')}",
                    end="",
                    flush=True,
                )
            case EventType.LLM_CALL_END:
                dur = d.get("duration_seconds", "?")
                tok = d.get("total_tokens", "?")
                suffix = ""
                if d.get("finish_reason") == "tool_calls":
                    suffix = " → " + self._yellow("tool_calls")
                self._print(f" {dur}s, {tok} tok{suffix}")
            case EventType.TOOL_CALL_START:
                tool = d.get("tool", "?")
                if self._verbose:
                    args = ", ".join(
                        f'{k}="{v}"' for k, v in d.get("arguments", {}).items()
                    )
                    self._print("  │ " + self._red("⚡ " + tool + "(" + args + ")"))
                else:
                    self._print("  │ " + self._red("⚡ " + tool))
            case EventType.TOOL_CALL_END:
                dur = d.get("duration_seconds", "?")
                self._print("  │   " + self._dim("→ returned (" + str(dur) + "s)"))
            case EventType.AGENT_END:
                name = d.get("agent", "?")
                dur = d.get("duration_seconds", "?")
                cost = d.get("cost", 0.0)
                cost_suffix = ""
                if cost:
                    cost_suffix = " $" + f"{cost:.4f}"
                self._print(
                    "  "
                    + self._green(
                        "✓ " + name + " done (" + str(dur) + "s)" + cost_suffix
                    )
                )
            case EventType.AGENT_ERROR:
                name = d.get("agent", "?")
                err = d.get("error", "unknown")
                self._print("  " + self._red("✗ " + name + " error: " + str(err)))


def console_hooks(
    *,
    color: bool = True,
    verbose: bool = False,
    file: TextIO | None = None,
) -> Hooks:
    """Return a :class:`Hooks` instance with a :class:`ConsoleReporter` registered.

    One-liner replacement for the common pattern of creating ``Hooks()``,
    instantiating a reporter, and calling ``hooks.on_all(reporter)``.
    """
    hooks = Hooks()
    hooks.on_all(ConsoleReporter(color=color, verbose=verbose, file=file))
    return hooks
