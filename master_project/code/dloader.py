from master_project.code.common.state import State, StateSpace
from master_project.code.common.task import Task, TaskSpace


class DataLoader:
    """Simple Wrapper for centralized data loading and initialisation."""

    def __init__(
        self,
        state_space: StateSpace,
        task_space: TaskSpace,
        verbose: bool = False,
    ):
        self._states = State.from_json_list(state_space)
        self._states.sort(key=lambda s: s.id)
        self._tasks = Task.from_json_list(task_space)
        self._tasks.sort(key=lambda t: t.id)
        for task in self._tasks:
            task.initialize_task_parameters(self._states, verbose)
            task.initialize_overrides(self._states, verbose)

    @property
    def states(self) -> list[State]:
        return self._states

    @property
    def tasks(self) -> list[Task]:
        return self._tasks
