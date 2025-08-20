import importlib
from tapas_gmm.master_project.state import State, StateSpace
from tapas_gmm.master_project.task import Task, TaskSpace


def load_dependency(dep_config):
    module_name, class_name = dep_config["class"].rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    args = [
        load_dependency(arg) if isinstance(arg, dict) and "class" in arg else arg
        for arg in dep_config.get("args", [])
    ]
    kwargs = {
        k: load_dependency(v) if isinstance(v, dict) and "class" in v else v
        for k, v in dep_config.get("kwargs", {}).items()
    }
    return cls(*args, **kwargs)


class DataLoader:
    """Simple Wrapper for centralized data loading and initialisation."""

    def __init__(
        self,
        state_space: StateSpace,
        task_space: TaskSpace,
        verbose: bool = False,
    ):
        self._states = State.from_json_list(state_space)
        self._tasks = Task.from_json_list(task_space)
        for task in self._tasks:
            task.initialize_task_parameters(self._states, verbose)
            task.initialize_overrides(self._states, verbose)

    @property
    def states(self) -> list[State]:
        return self._states

    @property
    def tasks(self) -> list[Task]:
        return self._tasks
