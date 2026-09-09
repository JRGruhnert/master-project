from dataclasses import dataclass
from enum import Enum
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from heca.misc import logger
from heca.data.entity import Entity
from heca.observation.td_entity import TDEntity
from heca.properties.states.area import AreaStateConfig
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D


@dataclass
class ObjectDeltaPoint:
    position_delta: float
    rotation_delta: float
    state: int
    cluster: int


@dataclass
class EntityPoint:
    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float
    rw: float
    state: int


@dataclass
class EntityData:
    entity: Entity
    current: EntityPoint
    goal: EntityPoint
    different: bool
    solved: bool


class Entity3DMode(Enum):
    DIFFERENCE = "diff"
    COLORS = "colors"
    DELTAS = "deltas"


@dataclass
class Style3DConfig:
    solved: str = "tab:green"
    unsolved: str = "black"
    true: str = "tab:yellow"
    false: str = "black"
    red: str = "tab:red"
    blue: str = "tab:blue"
    pink: str = "tab:pink"
    default: str = "tab:orange"


STYLE_DICT = {
    "solved": "tab:green",
    "unsolved": "black",
    "true": "tab:yellow",
    "false": "black",
    "red": "tab:red",
    "blue": "tab:blue",
    "pink": "tab:pink",
    "default": "tab:orange",
}


@dataclass
class Entity3DHelperConfig:
    mode: Entity3DMode = Entity3DMode.DIFFERENCE
    style: Style3DConfig = Style3DConfig()


class Entity3DHelper:
    def __init__(self, config: Entity3DHelperConfig):
        self.config = config
        self.entities: list[EntityData] = []

        if self.config.mode == Entity3DMode.DIFFERENCE:
            self.fig = plt.figure(figsize=(16, 10))
            self.axes = {
                "different": self.fig.add_subplot(121, projection="3d"),
                "same": self.fig.add_subplot(122, projection="3d"),
            }
        elif self.config.mode == Entity3DMode.COLORS:
            self.fig = plt.figure(figsize=(20, 10))
            self.axes = {
                "red": self.fig.add_subplot(131, projection="3d"),
                "blue": self.fig.add_subplot(132, projection="3d"),
                "pink": self.fig.add_subplot(133, projection="3d"),
            }
        elif self.config.mode == Entity3DMode.DELTAS:
            self.fig = plt.figure(figsize=(16, 10))
            self.axes = {
                "precon": self.fig.add_subplot(121, projection="3d"),
                "postcon": self.fig.add_subplot(122, projection="3d"),
            }
        else:
            logger.error(f"Unsupported mode: {self.config.mode}")
            raise ValueError(f"Unsupported mode: {self.config.mode}")

    def finish(self):
        result_handles = None
        if self.config.mode == Entity3DMode.DIFFERENCE:
            sd_count = sum(e.solved and e.different for e in self.entities)
            ss_count = sum(e.solved and not e.different for e in self.entities)
            d_count = sum(e.different for e in self.entities)
            s_count = len(self.entities) - d_count
            self.axes["different"].set_title(f"Different ({sd_count}/{d_count})")
            self.axes["same"].set_title(f"Same ({ss_count}/{s_count})")
            result_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="white",
                    markeredgecolor=self.config.style.solved,
                    markeredgewidth=1.5,
                    markersize=8,
                    label="Solved",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="white",
                    markeredgecolor=self.config.style.unsolved,
                    markeredgewidth=1.5,
                    markersize=8,
                    label="Unsolved",
                ),
            ]

        elif self.config.mode == Entity3DMode.COLORS:
            self.axes["red"].set_title(f"Red")
            self.axes["blue"].set_title(f"Blue")
            self.axes["pink"].set_title(f"Pink")
            result_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="white",
                    markeredgecolor="tab:red",
                    markeredgewidth=1.5,
                    markersize=8,
                    label="Red",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="white",
                    markeredgecolor="tab:blue",
                    markeredgewidth=1.5,
                    markersize=8,
                    label="Blue",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="white",
                    markeredgecolor="tab:pink",
                    markeredgewidth=1.5,
                    markersize=8,
                    label="Pink",
                ),
            ]
        elif self.config.mode == Entity3DMode.DELTAS:
            self.axes["precon"].set_title(f"Preconditions")
            self.axes["postcon"].set_title(f"Postconditions")
        else:
            raise ValueError(f"Unsupported mode: {self.config.mode}")

        for ax in self.axes.values():
            if result_handles is None:
                continue
            a = ax.legend(
                handles=result_handles,
                title="Result",
                loc="upper right",
                fontsize="small",
            )
            ax.add_artist(a)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
        logger.debug(f"Finished plotting entities {len(self.entities)}")

    def set_entity(
        self,
        entity: Entity,
        current: TDEntity,
        goal: TDEntity,
        different: bool,
        solved: bool,
    ):
        self.entities.append(
            EntityData(
                entity=entity,
                current=self.get_entity_point(current),
                goal=self.get_entity_point(goal),
                different=different,
                solved=solved,
            )
        )

    def get_entity_point(self, entity: TDEntity) -> EntityPoint:
        return EntityPoint(
            x=entity.position[0].item(),
            y=entity.position[1].item(),
            z=entity.position[2].item(),
            rx=entity.rotation[0].item(),
            ry=entity.rotation[1].item(),
            rz=entity.rotation[2].item(),
            rw=entity.rotation[3].item(),
            state=int(entity.state[0].item()),
        )

    def get_color(self, solved: bool) -> str:
        return self.config.style.solved if solved else self.config.style.unsolved

    def get_axis(self, ed: EntityData) -> Axes | None:
        if self.config.mode == Entity3DMode.DIFFERENCE:
            return self.axes["different"] if ed.different else self.axes["same"]
        elif self.config.mode == Entity3DMode.COLORS:
            return self.axes.get(ed.entity.cfg.label, None)
        else:
            raise ValueError(f"Unsupported mode: {self.config.mode}")

    def show_entities(self):
        for ed in self.entities:
            ec = self.get_color(ed.solved)
            if self.config.mode == Entity3DMode.DELTAS:
                ax_current = self.axes["precon"]
                ax_goal = self.axes["postcon"]
            else:
                ax_current = self.get_axis(ed)
                ax_goal = self.get_axis(ed)

            if ax_current and ax_goal:
                self.show_location(
                    ax_current,
                    ed.current,
                    STYLE_DICT.get(ed.entity.cfg.label, STYLE_DICT["default"]),
                )
                self.show_location(
                    ax_goal,
                    ed.goal,
                    STYLE_DICT.get(ed.entity.cfg.label, STYLE_DICT["default"]),
                )
                self.show_rotation(ax_current, ed.current, ec)
                self.show_rotation(ax_goal, ed.goal, ec)

    def show_location(self, axis: Axes, point: EntityPoint, color: str):
        axis.scatter(point.x, point.y, point.z, c=color)

    def show_rotation(self, axis: Axes, point: EntityPoint, color: str):
        dir = R.from_quat([point.rx, point.ry, point.rz, point.rw]).apply([0.01, 0, 0])
        axis.quiver(
            point.x,
            point.y,
            point.z,
            dir[0],
            dir[1],
            dir[2],
            length=1,
            color=color,
            arrow_length_ratio=0.1,
        )

    def show_edges(self):
        for ed in self.entities:
            p1 = ed.current
            p2 = ed.goal
            solved = ed.solved
            ax = self.get_axis(ed)
            if ax is not None:
                ax.plot(
                    [p1.x, p2.x],
                    [p1.y, p2.y],
                    [p1.z, p2.z],
                    color=self.get_color(solved),
                )

    def show_areas(self, area: AreaStateConfig):
        for axis in self.axes.values():
            for surface in area.eval_surfaces.values():
                (x0, y0, z0), (x1, y1, z1) = surface
                corners = np.array(
                    [
                        [x0, y0, z0],
                        [x1, y0, z0],
                        [x1, y1, z0],
                        [x0, y1, z0],
                        [x0, y0, z1],
                        [x1, y0, z1],
                        [x1, y1, z1],
                        [x0, y1, z1],
                    ]
                )
                faces = [
                    [corners[0], corners[1], corners[2], corners[3]],  # bottom
                    [corners[4], corners[5], corners[6], corners[7]],  # top
                    [corners[0], corners[1], corners[5], corners[4]],  # front
                    [corners[2], corners[3], corners[7], corners[6]],  # back
                    [corners[1], corners[2], corners[6], corners[5]],  # right
                    [corners[3], corners[0], corners[4], corners[7]],  # left
                ]
                poly = Poly3DCollection(
                    faces, color="grey", alpha=0.2, linewidths=0.5, edgecolors="k"
                )
                axis.add_collection3d(poly)
