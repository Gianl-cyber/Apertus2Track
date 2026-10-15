from dataclasses import dataclass


@dataclass
class Verdict:
    """Answer of a guardrail rule when it decides not to plan."""

    decision: str  # "refuse" or "ask_user"
    message: str   # reason or question shown to the user
    rule: str      # name of the rule that fired, for logging
