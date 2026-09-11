import abc
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypeVar, cast

from heca.misc import logger

C = TypeVar("C", bound="Configurable")
R = TypeVar("R", bound="Registerable")
P = TypeVar("P", bound="Persistable")


def find_repo_root(start: Path) -> Path:
    for p in (start, *start.parents):
        if (p / "pyproject.toml").exists():
            return p
    raise RuntimeError("Could not find repository root")


class Configurable(abc.ABC):
    root: ClassVar[Path] = find_repo_root(Path(__file__).resolve()) / "data"
    _config_registry: ClassVar[dict[type, type]] = {}

    @dataclass(kw_only=True)
    class Config:
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._config_registry = {}
        cls._instances = {}
        # Register cls.Config -> cls into every Registerable ancestor's registry,
        # so A.get(B.Config(...)) resolves B even when called on A.
        own_config = cls.__dict__.get("Config")
        if own_config is None:
            # raise TypeError(f"{cls.__name__} must define a nested Config dataclass.")
            return

        for base in cls.__mro__[1:]:
            if issubclass(base, Configurable):
                base._config_registry[own_config] = cls

    def __init__(self, cfg: "Configurable.Config"):
        self.cfg = cfg

    @classmethod
    def get(cls: type[C], cfg: "Configurable.Config") -> C:
        target_cls = cls._config_registry.get(type(cfg), cls)
        return target_cls(cfg)

    @classmethod
    def base_dir(cls, cfg: "Configurable.Config", folder: str) -> Path:
        """
        cls.root / folder
        """
        path = cls.root / folder
        path.mkdir(parents=True, exist_ok=True)
        return path


class Registerable(Configurable):
    _instances: ClassVar[dict[tuple[type, str], "Registerable"]] = {}

    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        label: str

    def __init__(self, cfg: "Registerable.Config"):
        super().__init__(cfg)
        self.cfg = cfg

    @classmethod
    def _key(cls: type[R], cfg: "Registerable.Config") -> tuple[type, str]:
        return (type(cfg), cfg.label)

    @classmethod
    def get(cls: type[R], cfg: "Registerable.Config") -> R:
        # Resolve the concrete subclass from the config type; fall back to cls
        # itself if this is already a concrete class being called directly.
        target_cls = cls._config_registry.get(type(cfg), cls)
        key = cls._key(cfg)
        if key not in cls._instances:
            cls._instances[key] = target_cls(cfg)
        return cast(R, cls._instances[key])

    @classmethod
    def instance_dir(cls, cfg: "Registerable.Config", folder: str) -> Path:
        """
        cls.root / folder / cfg.label
        """
        path = cls.base_dir(cfg, folder) / cfg.label
        path.mkdir(parents=True, exist_ok=True)
        return path


class Persistable(Registerable, abc.ABC):
    _persisted_instances: ClassVar[dict[tuple[type, str], "Persistable"]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._persisted_instances = {}

    @dataclass(kw_only=True)
    class Config(Registerable.Config):
        load_tag: str | None = None
        tag: str
        folder: str

    def __init__(self, cfg: "Persistable.Config"):
        super().__init__(cfg)
        self.cfg = cfg

    @classmethod
    def _key(cls: type[R], cfg: "Persistable.Config") -> tuple[type, str]:
        return (type(cfg), cfg.label + cfg.tag)

    @classmethod
    def get(cls: type[P], cfg: "Persistable.Config", auto_load: bool = True) -> P:
        target_cls = cls._config_registry.get(type(cfg), cls)
        key = cls._key(cfg)
        # print(key)
        if key not in cls._persisted_instances:
            # print("new")
            instance = cast(P, target_cls(cfg))
            if auto_load:
                instance.load()
            cls._persisted_instances[key] = instance
        return cast(P, cls._persisted_instances[key])

    @classmethod
    def load_dir(cls, cfg: "Persistable.Config") -> Path:
        """
        cls.root / cfg.folder / cfg.label / cfg.tag
        """
        tag = cfg.load_tag or cfg.tag
        path = cls.instance_dir(cfg, cfg.folder) / tag
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def save_dir(cls, cfg: "Persistable.Config") -> Path:
        """
        cls.root / cfg.folder / cfg.label / cfg.tag
        """
        path = cls.instance_dir(cfg, cfg.folder) / cfg.tag
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save(self):
        path = self.save_dir(self.cfg)
        logger.info(f"Saving {type(self)} to {path}")
        self._save(path)

    def load(self):
        path = self.load_dir(self.cfg)
        logger.info(f"Loading {type(self)} from {path}")
        self._load(path)

    @abc.abstractmethod
    def _save(self, path: Path) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def _load(self, path: Path) -> bool:
        raise NotImplementedError()


# root / folder / label / tag
# data / agents / tapas /
