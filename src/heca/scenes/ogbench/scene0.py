from dataclasses import dataclass
from textwrap import dedent

import numpy as np

from heca.data.free import FreeEntity
from heca.data.prismatic import PrismaticEntity
from heca.data.static import StaticEntity
from heca.scenes.ogbench.scene import OGScene
from heca.data.data import DCEntity, DCScene
from heca.data.entity import Entity


class OGScene0(OGScene):
    @dataclass(kw_only=True)
    class Config(OGScene.Config):
        tag: str = "scene0"
        vis: bool = False

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    @property
    def description(self) -> str:
        return dedent("""
            Scene Description

            The image shows a virtual robot manipulation environment.

            Fixed properties:
            - The robot arm is always semi-transparent and purple.
            - The sliding window is always located on the right side of the scene.
            - The drawer is always located on the left side of the scene.
            - Two buttons are always located near the front-center of the scene:
            - a left button,
            - a right button.

            Objects:
            - A semi-transparent purple robot arm.
            - A white sliding window with a white handle.
                The window can be opened by pulling it toward the camera.
            - A white drawer with a white handle.
            - A red cube.

            Variable properties:
            - The window may be open or closed.
            - The drawer may be open or closed.
            - The red cube may be:
                - on the floor,
                - inside the drawer, or
                - outside the camera's field of view.
            - Each button may be either red or white.

            Button color indicates state:
            - Red = locked
            - White = unlocked
        """)

    @property
    def entities(self) -> dict[str, Entity]:
        ents = {
            "drawer0": PrismaticEntity.Config(
                question="What describes the drawer the best?",
                answers=["It is open", "It is closed"],
            ),
            "window0": PrismaticEntity.Config(
                question="What describes the sliding window the best?",
                answers=[
                    "it is open and therefore moved to the front",
                    "it is closed and therefore moved to the back",
                ],
            ),
            "button0": StaticEntity.Config(
                question="What is the color of the left button?",
                answers=["white", "red"],
                n_states=2,
            ),
            "button1": StaticEntity.Config(
                question="What is the color of the right button?",
                answers=["white", "red"],
                n_states=2,
            ),
            "cube0": FreeEntity.Config(
                question="Where is the red cube in the scene?",
                answers=[
                    "inside the drawer",
                    "on the floor",
                    "unknown, cause it is not visible",
                ],
            ),
        }
        return {l: Entity.get(e) for l, e in ents.items()}
