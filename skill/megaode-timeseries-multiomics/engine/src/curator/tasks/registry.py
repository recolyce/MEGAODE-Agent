"""Extensible task registry. Add a module and call register_task."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.curator.loader import BioMasterDataset

TaskFn = Callable[..., list[Any]]
TASKS: dict[str, TaskFn] = {}


def register_task(name: str) -> Callable[[TaskFn], TaskFn]:
    def deco(fn: TaskFn) -> TaskFn:
        TASKS[name] = fn
        return fn

    return deco


def get_task(name: str) -> TaskFn:
    if name not in TASKS:
        raise KeyError(f"Unknown task {name!r}. Registered: {sorted(TASKS)}")
    return TASKS[name]


def available_tasks() -> list[str]:
    return sorted(TASKS)
