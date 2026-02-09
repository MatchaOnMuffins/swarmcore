from __future__ import annotations

import inspect
import json
import time
from typing import Any, Callable, get_type_hints

import litellm

from swarmcore.context import SharedContext
from swarmcore.exceptions import AgentError
from swarmcore.models import AgentResult, TokenUsage

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
                properties[param_name]["description"] = (
                    stripped.split(":", 1)[1].strip()
                )
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

    async def run(self, task: str, context: SharedContext) -> AgentResult:
        """Execute the agent on a task with shared context."""
        start = time.monotonic()
        total_usage = TokenUsage()

        system_content = self.instructions
        context_str = context.format_for_prompt()
        if context_str:
            system_content += "\n\n# Context from prior agents\n" + context_str

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": task},
        ]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if self._tool_schemas:
            kwargs["tools"] = self._tool_schemas

        try:
            while True:
                response = await litellm.acompletion(**kwargs)

                usage = response.usage
                if usage:
                    total_usage.prompt_tokens += usage.prompt_tokens or 0
                    total_usage.completion_tokens += usage.completion_tokens or 0
                    total_usage.total_tokens += usage.total_tokens or 0

                choice = response.choices[0]
                message = choice.message

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

                    if fn_name not in self._tools:
                        raise AgentError(
                            self.name, f"Model called unknown tool: {fn_name}"
                        )

                    result = self._tools[fn_name](**fn_args)
                    if inspect.isawaitable(result):
                        result = await result

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(result),
                        }
                    )

                kwargs["messages"] = messages

        except AgentError:
            raise
        except Exception as e:
            raise AgentError(self.name, str(e)) from e

        duration = time.monotonic() - start

        return AgentResult(
            agent_name=self.name,
            input_task=task,
            output=output,
            model=self.model,
            duration_seconds=round(duration, 3),
            token_usage=total_usage,
        )
