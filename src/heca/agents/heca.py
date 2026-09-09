from dataclasses import dataclass
from typing import Sequence

import torch

from heca.experts.expert import ExpertModel
from heca.graphs.graph import Graph, SubgoalMode
from heca.graphs.edge_set import ResidualMode
from heca.learning.learner import Learner
from heca.misc import logger
from heca.misc.interrupt import stop_requested
from heca.data.data import DCScene
from heca.misc.base import Configurable
from heca.scenes.scene import Scene, SceneFeedback


class Heca(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        agents: Sequence[ExpertModel.Config]
        learner: Learner.Config
        smode: SubgoalMode
        visualize: bool
        inference: bool
        virtual: bool
        reload: bool
        use_gt: bool

        fit_rotation: bool = True
        residual_mode: ResidualMode = ResidualMode.CONSTANT

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.learner = Learner.get(self.cfg.learner)
        if self.cfg.inference:
            self.learner.eval()

        self._episode = 0
        self._x: DCScene | None = None
        self._y: DCScene | None = None
        self.scene = Scene.get(self.cfg.agents[0].scene)

        for a in self.cfg.agents:
            expert = ExpertModel.get(a, auto_load=False)
            expert.use_gt(self.cfg.use_gt)
            expert.load()

            if self.cfg.reload:
                expert.force_recompute()

            if self.cfg.virtual:

                expert.virtual()

        self.graph = Graph.generate(
            list(self.cfg.agents),
            smode=cfg.smode,
            residual_mode=cfg.residual_mode,
        )
        self.graph.plot(path=self.scene.save_dir(self.scene.cfg))
        if self.cfg.smode in (SubgoalMode.CHAIN, SubgoalMode.BOTH):
            self.graph.plot_connections(path=self.scene.save_dir(self.scene.cfg))
        self.graph.log()

    def step(
        self, x: DCScene, new_ep: bool = False
    ) -> tuple[DCScene, SceneFeedback, bool]:
        if new_ep:
            self._episode += 1
        self.graph.set_start(x)
        data = self.graph.export()
        option = self.learner.predict(data, new_ep)
        a, s = self.graph.select(option)
        z, fb = ExpertModel.get(a).act(x, s)
        if logger.TRACE:
            self._trace_step(x, data, option, a, s, z, fb)
        if stop_requested():
            return z, fb, False
        lock = self.learner.update(fb)
        return z, fb, lock

    def _trace_step(
        self,
        x: DCScene,
        data,
        option: int,
        a: ExpertModel.Config,
        s: DCScene,
        z: DCScene,
        fb: SceneFeedback,
    ):
        def _indent(text: str, prefix: str = "    ") -> str:
            return "\n".join(prefix + line for line in text.splitlines())

        lines = [
            "===== step trace " f"client={self.learner.cfg.tag} ",
            "start:",
            _indent(str(x)),
            "goal:",
            _indent(str(self.graph.goal)),
        ]

        net = (
            self.learner.inference_net
            if self.learner.train_mode
            else self.learner.network
        )
        with torch.inference_mode():
            logits = net.actor(data).detach().cpu().squeeze(0)
        probs = torch.softmax(logits, dim=-1)

        lines.append("options (logits/probs):")
        for i, key in enumerate(self.graph.export_keys):
            agent = self.graph.ns_option.items[
                self.graph.ns_option.get_index(key)
            ].model.tag
            lines.append(
                f"    [{i}] key={key:<45} agent={agent:<28} "
                f"logit={logits[i]:+.3f} prob={probs[i]:.3f}"
            )
        lines.append(
            f"    selected: idx={option} key={self.graph.export_keys[option]} "
            f"agent={a.tag}"
        )

        lines.append("subgoal (given to expert):")
        lines.append(_indent(str(s)))
        lines.append("result (expert outcome):")
        lines.append(_indent(str(z)))
        lines.append(
            f"final feedback: terminal={fb.terminal} reward={fb.reward:.4f} "
            f"truncated={fb.truncated}"
        )
        logger.trace("\n".join(lines))

    def act(self, x: DCScene, y: DCScene) -> tuple[DCScene, SceneFeedback]:
        self.graph.set_goal(y)
        z, fb, lock = self.step(x, True)
        while not (fb.truncated or fb.terminal or lock):
            z, fb, lock = self.step(z)
        return z, fb

    def sample(self) -> tuple[DCScene, DCScene]:
        (x, ix), (y, iy) = self.scene.sample_task()
        logger.debug("New Episode")
        return x, y

    def tick(self) -> bool:
        if self._x is None or self._y is None:
            self._x, self._y = self.sample()
            self.graph.set_goal(self._y)
            new_ep = True
        else:
            new_ep = False

        z, fb, lock = self.step(self._x, new_ep)
        self._x = z

        if fb.truncated or fb.terminal or lock:
            self._x = None
            self._y = None

        return lock
