from master_project.registry.state_type import StateType


class StateTypeRegistry:
    _registry: dict[str, StateType] = {}

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
