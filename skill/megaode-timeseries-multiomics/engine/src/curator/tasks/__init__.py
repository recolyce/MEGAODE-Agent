from src.curator.tasks import last_interval as _last_interval  # noqa: F401
from src.curator.tasks.registry import TASKS, get_task, register_task

__all__ = ["TASKS", "get_task", "register_task"]
