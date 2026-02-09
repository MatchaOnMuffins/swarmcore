class SwarmError(Exception):
    """Base exception for all swarm-level errors."""


class AgentError(SwarmError):
    """Raised when an individual agent fails during execution."""

    def __init__(self, agent_name: str, message: str) -> None:
        self.agent_name = agent_name
        super().__init__(f"Agent '{agent_name}': {message}")
