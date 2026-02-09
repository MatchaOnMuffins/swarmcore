from __future__ import annotations

import inspect
import json
import time
from typing import TYPE_CHECKING, Any, Callable, cast, get_type_hints

import litellm
from litellm.types.utils import Choices, ModelResponse, Usage

from swarmcore.context import SharedContext
from swarmcore.exceptions import AgentError
from swarmcore.hooks import Event, EventType, Hooks
from swarmcore.models import AgentResult, LLMCallRecord, TokenUsage, ToolCallRecord

if TYPE_CHECKING:
    from swarmcore.flow import Flow

_PYTHON_TO_JSON_SCHEMA: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
}


def _function_to_tool_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """Convert a Python function to an OpenAI function-calling tool schema."""
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    doc = inspect.getdoc(func) or ""

    description = doc.split("\n")[0].strip() if doc else func.__name__

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        param_type = hints.get(param_name, str)
        json_type = _PYTHON_TO_JSON_SCHEMA.get(param_type, "string")
        properties[param_name] = {"type": json_type}

        for line in doc.split("\n"):
            stripped = line.strip()
            if stripped.startswith(f"{param_name}:") or stripped.startswith(
                f"{param_name} :"
            ):
                properties[param_name]["description"] = stripped.split(":", 1)[
                    1
                ].strip()
                break

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


class Agent:
    def __init__(
        self,
        name: str,
        instructions: str,
        model: str = "anthropic/claude-opus-4-6",
        tools: list[Callable[..., Any]] | None = None,
    ) -> None:
        self.name = name
        self.instructions = instructions
        self.model = model

        self._tools: dict[str, Callable[..., Any]] = {}
        self._tool_schemas: list[dict[str, Any]] = []

        if tools:
            for func in tools:
                self._tools[func.__name__] = func
                self._tool_schemas.append(_function_to_tool_schema(func))

    def __rshift__(self, other: Agent | Flow) -> Flow:
        from swarmcore.flow import Flow as _Flow

        if isinstance(other, Agent):
            return _Flow([self, other])
        if isinstance(other, _Flow):
            steps: list[Agent | list[Agent]] = [self, *other._steps]
            return _Flow(steps)
        return NotImplemented

    def __or__(self, other: Agent | Flow) -> Flow:
        from swarmcore.flow import Flow as _Flow

        if isinstance(other, Agent):
            return _Flow([[self, other]])
        if isinstance(other, _Flow):
            agents: list[Agent] = []
            for step in other._steps:
                if isinstance(step, list):
                    agents.extend(step)
                else:
                    agents.append(step)
            return _Flow([[self] + agents])
        return NotImplemented

    async def run(
        self,
        task: str,
        context: SharedContext,
        *,
        hooks: Hooks | None = None,
        structured_output: bool = False,
        expand: set[str] | None = None,
        extra_tools: list[Callable[..., Any]] | None = None,
    ) -> AgentResult:
        """Execute the agent on a task with shared context."""
        start = time.monotonic()
        total_usage = TokenUsage()
        llm_call_records: list[LLMCallRecord] = []
        tool_call_records: list[ToolCallRecord] = []
        call_index = 0

        # Build run-local tool registry (agent tools + any extras)
        run_tools: dict[str, Callable[..., Any]] = dict(self._tools)
        run_schemas: list[dict[str, Any]] = list(self._tool_schemas)
        if extra_tools:
            for func in extra_tools:
                run_tools[func.__name__] = func
                run_schemas.append(_function_to_tool_schema(func))

        if hooks and hooks.is_active:
            await hooks.emit(
                Event(EventType.AGENT_START, {"agent": self.name, "task": task})
            )

        system_content = self.instructions
        context_str = context.format_for_prompt(expand=expand)
        if context_str:
            system_content += "\n\n# Context from prior agents\n" + context_str
        if structured_output:
            system_content += (
                "\n\n# Output format\n"
                "Begin your response with a 2-3 sentence executive summary "
                "wrapped in <summary>...</summary> tags, then provide your "
                "full detailed response below the tags."
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": task},
        ]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if run_schemas:
            kwargs["tools"] = run_schemas

        try:
            while True:
                if hooks and hooks.is_active:
                    await hooks.emit(
                        Event(
                            EventType.LLM_CALL_START,
                            {"agent": self.name, "call_index": call_index},
                        )
                    )

                llm_start = time.monotonic()
                response = cast(ModelResponse, await litellm.acompletion(**kwargs))
                llm_duration = round(time.monotonic() - llm_start, 3)

                usage = cast(Usage | None, getattr(response, "usage", None))
                call_usage = TokenUsage()
                if usage:
                    call_usage.prompt_tokens = usage.prompt_tokens or 0
                    call_usage.completion_tokens = usage.completion_tokens or 0
                    call_usage.total_tokens = usage.total_tokens or 0
                    total_usage.prompt_tokens += call_usage.prompt_tokens
                    total_usage.completion_tokens += call_usage.completion_tokens
                    total_usage.total_tokens += call_usage.total_tokens

                choice = cast(Choices, response.choices[0])
                message = choice.message
                finish_reason = getattr(choice, "finish_reason", None) or ""

                tool_names_requested: list[str] = []
                if message.tool_calls:
                    tool_names_requested = [
                        str(tc.function.name) for tc in message.tool_calls
                    ]

                llm_record = LLMCallRecord(
                    call_index=call_index,
                    token_usage=call_usage,
                    duration_seconds=llm_duration,
                    tool_calls_requested=tool_names_requested,
                    finish_reason=finish_reason,
                )
                llm_call_records.append(llm_record)

                if hooks and hooks.is_active:
                    await hooks.emit(
                        Event(
                            EventType.LLM_CALL_END,
                            {
                                "agent": self.name,
                                "call_index": call_index,
                                "finish_reason": finish_reason,
                                "duration_seconds": llm_duration,
                            },
                        )
                    )

                call_index += 1

                if not message.tool_calls:
                    output = message.content or ""
                    break

                # Manually construct assistant message dict for compatibility
                msg_dict: dict[str, Any] = {
                    "role": "assistant",
                    "content": message.content,
                }
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ]
                messages.append(msg_dict)

                for tool_call in message.tool_calls:
                    fn_name = tool_call.function.name
                    fn_args = json.loads(tool_call.function.arguments)

                    if fn_name not in run_tools:
                        raise AgentError(
                            self.name, f"Model called unknown tool: {fn_name}"
                        )

                    if hooks and hooks.is_active:
                        await hooks.emit(
                            Event(
                                EventType.TOOL_CALL_START,
                                {
                                    "agent": self.name,
                                    "tool": fn_name,
                                    "arguments": fn_args,
                                },
                            )
                        )

                    tool_start = time.monotonic()
                    result = run_tools[fn_name](**fn_args)
                    if inspect.isawaitable(result):
                        result = await result
                    tool_duration = round(time.monotonic() - tool_start, 3)

                    result_str = str(result)
                    tool_record = ToolCallRecord(
                        tool_name=fn_name,
                        arguments=fn_args,
                        result=result_str[:1000],
                        duration_seconds=tool_duration,
                    )
                    tool_call_records.append(tool_record)

                    if hooks and hooks.is_active:
                        await hooks.emit(
                            Event(
                                EventType.TOOL_CALL_END,
                                {
                                    "agent": self.name,
                                    "tool": fn_name,
                                    "duration_seconds": tool_duration,
                                },
                            )
                        )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_str,
                        }
                    )

                kwargs["messages"] = messages

        except AgentError:
            if hooks and hooks.is_active:
                await hooks.emit(
                    Event(
                        EventType.AGENT_ERROR,
                        {"agent": self.name, "error": str(AgentError)},
                    )
                )
            raise
        except Exception as e:
            if hooks and hooks.is_active:
                await hooks.emit(
                    Event(
                        EventType.AGENT_ERROR,
                        {"agent": self.name, "error": str(e)},
                    )
                )
            raise AgentError(self.name, str(e)) from e

        duration = time.monotonic() - start

        agent_result = AgentResult(
            agent_name=self.name,
            input_task=task,
            output=output,
            model=self.model,
            duration_seconds=round(duration, 3),
            token_usage=total_usage,
            llm_calls=llm_call_records,
            tool_calls=tool_call_records,
            llm_call_count=len(llm_call_records),
            tool_call_count=len(tool_call_records),
        )

        if hooks and hooks.is_active:
            await hooks.emit(
                Event(
                    EventType.AGENT_END,
                    {
                        "agent": self.name,
                        "duration_seconds": agent_result.duration_seconds,
                    },
                )
            )

        return agent_result
