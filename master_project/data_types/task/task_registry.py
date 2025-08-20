from master_project.data_types.task.task import Task


class TaskRegistry:
    _registry: dict[str, Task] = {}

    @classmethod
    def register_type(cls, value: StateType):
        """Decorator to register state types"""

        def decorator(state_class):
            cls._registry[value] = state_class
            return state_class

        return decorator

    @classmethod
    def get(cls, name):
        return cls._registry.get(name)

    @classmethod
    def all(cls):
        return cls._registry.copy()
