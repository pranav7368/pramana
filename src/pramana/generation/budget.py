"""Per-request provider-call budget, shared by generation and verification."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from time import monotonic

from pramana.generation.base import ProviderError


@dataclass
class CallBudget:
    remaining_calls: int
    deadline: float
    seconds: float


_active: ContextVar[CallBudget | None] = ContextVar("pramana_call_budget", default=None)


@contextmanager
def request_budget(max_calls: int, seconds: float):
    token = _active.set(CallBudget(max_calls, monotonic() + seconds, seconds))
    try:
        yield
    finally:
        _active.reset(token)


def remaining_seconds(default: float) -> float:
    """Bound non-HTTP waits without spending a provider call."""
    budget = _active.get()
    if budget is None:
        return default
    remaining = budget.deadline - monotonic()
    if remaining <= 0 or budget.remaining_calls <= 0:
        raise ProviderError("Request provider-call budget exhausted", provider="budget")
    # Deadline arithmetic can round slightly above the configured duration.
    # Never advertise a timeout longer than that original request allowance.
    return min(default, budget.seconds, remaining)


def provider_timeout(default: float) -> float:
    budget = _active.get()
    if budget is None:
        return default
    remaining = remaining_seconds(default)
    budget.remaining_calls -= 1
    return min(default, remaining)
