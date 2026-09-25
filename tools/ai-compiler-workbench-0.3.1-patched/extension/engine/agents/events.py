"""Backward-compatibility shim: re-exports bus from core.events."""
from core.events import bus, EventBus
__all__ = ["bus", "EventBus"]
