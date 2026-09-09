from pathlib import Path
import h5py
import joblib
from matplotlib import pyplot as plt
import torch
import numpy as np
from dataclasses import dataclass, field
from functools import cached_property
from tensordict import TensorDict
from tapas_gmm_modified.utils.robot_trajectory import RobotTrajectory
from tapas_gmm_modified.policy.gmm import GMMPolicy, GMMPolicyConfig
from tapas_gmm_modified.utils.observation import SceneObservation, dict_to_tensordict
from tapas_gmm_modified.policy.models.tpgmm import (
    ModelType,
    FittingStage,
    InitStrategy,
    TPGMMConfig,
    AutoTPGMMConfig,
    AutoTPGMM,
    Demos,
    FrameSelectionConfig,
    DemoSegmentationConfig,
    CascadeConfig,
)

from heca.experts.expert import ExpertModel
from heca.data.pair import ConPair
from heca.data.data import DCScene
from heca.data.prismatic import PrismaticEntity
from heca.misc import logger
from heca.misc.interrupt import stop_requested
from heca.misc.hardware import device
from heca.scenes.scene import Scene, SceneFeedback
from heca.utils.quaternion import Quaternion

import riepybdlib.mappings as _rbd_mappings

_orig_quat_log_e = _rbd_mappings.quat_log_e


def _hemisphere_quat_log_e(g, reg=1e-6, arccos_func=_rbd_mappings.arccos_cont):
    if isinstance(g, list):
        g = [(-q if q.q0 < 0 else q) for q in g]
    elif g.q0 < 0:
        g = -g
    return _orig_quat_log_e(g, reg, arccos_func)


def _hemisphere_quat_log_e_star(g, reg=1e-6):
    return _hemisphere_quat_log_e(g, reg, _rbd_mappings.arccos_star)


_rbd_mappings.quat_log_e = _hemisphere_quat_log_e
_rbd_mappings.quat_log_e_star = _hemisphere_quat_log_e_star
# --------------------------------------------------------------------------------


class TapasExpert(ExpertModel):
    @dataclass(kw_only=True)
    class Config(ExpertModel.Config):
        folder: str = "tapas"
        policy: GMMPolicyConfig = field(
            default_factory=lambda: GMMPolicyConfig(
                suffix="release",
                model=AutoTPGMMConfig(
                    tpgmm=TPGMMConfig(
                        n_components=20,
                        add_time_component=True,
                        add_action_component=False,
                        position_only=False,
                        add_gripper_action=True,
                        reg_shrink=1e-3,  # 1e-2
                        reg_diag=1e-3,  # 2e-4
                        reg_diag_gripper=2e-2,
                        reg_em_finish_shrink=1e-2,
                        reg_em_finish_diag=2e-4,
                        reg_em_finish_diag_gripper=2e-2,
                        reg_init_diag=5e-4,  # 5
                        heal_time_variance=True,
                    ),
                    frame_selection=FrameSelectionConfig(
                        init_strategy=InitStrategy.TIME_BASED,
                        fitting_actions=(FittingStage.INIT,),
                        rel_score_threshold=0.0,
                    ),
                    demos_segmentation=DemoSegmentationConfig(
                        distance_based=False,
                        velocity_based=True,
                        repeat_final_step=10,
                        repeat_first_step=10,
                        components_prop_to_len=True,  # True,
                        velocity_threshold=0.007,
                    ),
                ),
                time_based=True,
                predict_dx_in_xdx_models=True,
                binary_gripper_action=True,
                binary_gripper_closed_threshold=0.5,
                dbg_prediction=False,
                force_overwrite_checkpoint_config=True,
                time_scale=1.0,
                postprocess_prediction=True,
                invert_prediction_batch=False,
                return_full_batch=True,
                batch_predict_in_t_models=True,
            ),
        )
        repeat_actions: int = 0
        gt_frames: list[list[str]] | None = None
        segment_ids: list[int] | None = None
        max_con_comps: int = 10
        fix_bimodal: bool = False
        snap_ee_actions: bool = True
        pos_only: bool = False
        use_ee_frames: bool = False
        max_demos: int = 20

        def __post_init__(self):
            if self.gt_frames is not None:
                flat = [n for seg in self.gt_frames for n in seg]
                if flat and all(isinstance(n, str) for n in flat):
                    segs = self.gt_frames
                    if not self.use_ee_frames:
                        segs = [
                            [n for n in seg if n not in ("ee_init", "ee_target")]
                            for seg in segs
                        ]
                        if any(len(seg) == 0 for seg in segs):
                            raise ValueError(
                                f"{self.tag}: use_ee_frames=False strips ee_init/"
                                "ee_target, but a gt_frames segment would end up "
                                "with no remaining frame anchor."
                            )
                    scene = Scene.get(self.scene, auto_load=False)
                    labels = list(scene.entities.keys())
                    frame_names = (
                        ["ee_init"]
                        + labels
                        + [f"{l}_target" for l in labels]
                        + ["ee_target"]
                    )
                    name_to_idx = {name: i for i, name in enumerate(frame_names)}
                    gt_frames = [[name_to_idx[n] for n in seg] for seg in segs]
                    self.policy.model.frame_selection.gt_frames = gt_frames
            self.policy.model.tpgmm.position_only = self.pos_only

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        temp = GMMPolicy(self.cfg.policy)
        assert isinstance(temp, GMMPolicy), "Policy model must be a GMMPolicy."
        self.policy = temp.to(device)

    # Safety cap on predicted plan length. A degenerate HMM-cascade can yield a
    # near-zero time step and produce a plan of ~1e5-1e6 actions, which would
    # otherwise freeze the action-replay loop in _act (see "Product did not
    # converge" warnings). Normal plans are ~40-200 steps.
    MAX_PLAN_STEPS: int = 2000

    def make_batch_prediction(
        self, x: SceneObservation  # type: ignore
    ) -> RobotTrajectory | None:
        # prds, _ = self.policy.predict(x)
        try:
            prds, _ = self.policy.predict(x)  # type: ignore
            if prds is not None and len(prds.points) > self.MAX_PLAN_STEPS:
                logger.warning(
                    f"{self.cfg.tag}: predicted plan of {len(prds.points)} "
                    f"steps (> {self.MAX_PLAN_STEPS}); aborting this option."
                )
                return None
            return prds  # type: ignore
        except Exception as e:
            logger.debug(f"Error: {e}")
            return None

    def make_prediction(self, x: SceneObservation) -> tuple[np.ndarray | None, bool]:  # type: ignore
        # prds, _ = self.policy.predict(x)
        try:
            prds, info = self.policy.predict(x)  # type: ignore
            return prds, info["done"]  # type: ignore
        except Exception as e:
            logger.debug(f"Error: {e}")
            return None, True

    def _act(self, x, y):
        gen = self._get_prediction(x, y)
        z, fb = x, None
        try:
            action = next(gen)
        except StopIteration:
            action = None
        while action is not None:
            if stop_requested():
                return x, SceneFeedback(reward=0.0, terminal=True, truncated=True)
            tdscene, tdimage, fb = self.scene.step(action)
            z = self.make_scene(tdscene, tdimage)
            try:
                action = gen.send(z)  # online: feed reached scene back
            except StopIteration:
                action = None
        if fb is None:  # nothing executed: prediction error
            return x, SceneFeedback(reward=0.0, terminal=True, truncated=False)
        return z, fb

    def _get_prediction(self, x, y):
        self.policy.reset_episode()
        quat0 = x.extras["ee_pose"][3:7] if self.cfg.pos_only else None

        def full_action(ee, gripper):
            if self.cfg.pos_only:
                return np.concatenate((ee, quat0, gripper))  # type: ignore
            return np.concatenate((ee, gripper))

        if self.cfg.policy.return_full_batch:
            xt = self.tapas_td(x, y)
            predictions = self.make_batch_prediction(xt)
            if predictions is None:
                yield None
                return
            while not predictions.is_finished:
                pred = predictions.step()
                yield full_action(pred.ee, pred.gripper)
        else:
            scene = x
            while True:
                xt = self.tapas_td(scene, y)
                action, done = self.make_prediction(xt)
                if done:
                    return  # finished
                if action is None:
                    yield None  # error
                    return
                scene = yield action  # consumer steps + sends back
                if scene is None:
                    return

    def _load(self, path: Path):
        filepath = self.policy_path(path)
        if filepath.exists():
            self.policy.from_disk(str(filepath))
            self.policy = self.policy.to(device)
            logger.info(f"Loading tapas policy from: {filepath}")
        else:
            logger.warning(f"No tapas policy found at given path: {filepath}")

    def eval(self):
        self.policy.eval()

    def policy_path(self, path: Path) -> Path:
        if self._use_gt:
            file_name = "policy_gt.pt"
        else:
            file_name = "policy_img.pt"
        return path / file_name

    def _save(self, path: Path):
        filepath = self.policy_path(path)
        logger.info(f"Saving tapas policy to: {filepath}")
        self.model.to_disk(str(filepath))

    @cached_property
    def model(self) -> AutoTPGMM:
        temp = self.policy.model
        assert isinstance(temp, AutoTPGMM)
        return temp

    @cached_property
    def demos(self) -> Demos:
        temp = self.model._demos
        assert isinstance(temp, Demos)
        return temp

    def tapas_td(self, dc_obs: DCScene, dc_goal: DCScene) -> TensorDict:
        poses = {}
        for l, entity in self.scene.entities.items():
            p = dc_obs[l].tpose
            if entity.cfg.canonicalize_rot:
                q = p[3:7].numpy().astype(float)
                yaw = np.arctan2(
                    2.0 * (q[0] * q[3] + q[1] * q[2]),
                    1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2),
                )
                p[3:7] = torch.tensor(
                    [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=p.dtype
                )
            poses[l] = p

        for l in self.scene.entities:
            poses[f"{l}_target"] = dc_goal[l].tpose

        poses["ee_target"] = torch.tensor(dc_goal.extras["ee_pose"])
        object_poses = dict_to_tensordict(poses)

        states = {l: dc_obs[l].tste for l in self.scene.entities}

        for l in self.scene.entities:
            states[f"{l}_target"] = dc_goal[l].tste
        states["ee_target"] = torch.tensor(dc_goal.extras["gripper_state"])
        object_states = dict_to_tensordict(states)

        action = torch.Tensor(dc_obs.extras["action"])
        reward = torch.Tensor(dc_obs.extras["reward"])
        joint_pos = torch.Tensor(dc_obs.extras["joint_pos"])
        joint_vel = torch.Tensor(dc_obs.extras["joint_vel"])
        ee_pose = torch.tensor(dc_obs.extras["ee_pose"])
        gripper_state = torch.Tensor(dc_obs.extras["gripper_state"])

        return SceneObservation(
            feedback=reward,
            action=action,
            cameras=None,  # multicam_obs,
            ee_pose=ee_pose,
            gripper_state=gripper_state,
            object_poses=object_poses,
            object_states=object_states,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            batch_size=torch.Size([]),
        )

    @cached_property
    def tps(self) -> set[str]:
        labels = set()
        for idx, key in enumerate(self.demos.frame_names):
            if idx in self.model._used_frames and key in self.scene.entities:
                labels.add(key)
        logger.debug(f"{self.cfg.tag} entities: {labels}")
        return labels

    @cached_property
    def conditions(self) -> ConPair:
        cache_path = self.conditions_cache_path
        if cache_path.exists() and not self._force_recompute:
            logger.debug(f"Loading cached conditions from {cache_path}")
            return joblib.load(cache_path)
        else:
            self.fit_conditions()
        return joblib.load(cache_path)

    @property
    def conditions_cache_path(self) -> Path:
        name = "condition-gt.joblib" if self._use_gt else "condition-vis.joblib"
        return self.load_dir(self.cfg) / name

    def fit_stage1(self, demos: Demos):
        liks, avg_logliks = self.model.fit_trajectories(
            demos,
            init_strategy=InitStrategy.TIME_BASED,
            fitting_actions=(FittingStage.INIT,),
        )
        logger.info(f"stage1 avg_logliks={avg_logliks}")
        return liks, avg_logliks

    def fit_stage2(self, demos: Demos):
        liks, avg_logliks = self.model.fit_trajectories(
            demos,
            fitting_actions=(FittingStage.EM_HMM,),
        )
        logger.info(f"stage2 avg_logliks={avg_logliks}")
        return liks, avg_logliks

    def plot_stage1(self):
        self.model.plot_model(
            scatter=True,
            annotate_gaussians=True,
            annotate_trajs=True,
            mean_as_base=False,
        )

    def plot_stage2(self):
        self.model.plot_model(
            scatter=True,
            annotate_gaussians=True,
            annotate_trajs=True,
            mean_as_base=False,
            time_based=True,
        )

    def plot_demo_velocities(
        self, selections: list[int] | None = None, max_demos: int | None = None
    ) -> list[int]:

        if selections is None:
            ids = self._all_demo_ids()
        else:
            ids = list(selections)
        if max_demos is not None:
            ids = ids[:max_demos]
        _, accepted = self._scan_demos(ids, plot=True)
        return accepted

    def load_demos(self) -> Demos:
        ids = self._all_demo_ids()
        observations, _ = self._scan_demos(ids, plot=False)
        if len(observations) < self.cfg.max_demos:
            raise ValueError(
                f"{self.cfg.tag}: only {len(observations)}/{len(ids)} demos "
                f"match the expected {self._expected_velocity_pauses()} "
                f"velocity pause(s); need >= {self.cfg.max_demos}."
            )
        return Demos(
            observations,
            meta_data={"tag": self.cfg.tag + self.cfg.label},
            add_init_ee_pose_as_frame=True,
            add_world_frame=False,
            frames_from_keypoints=False,
            kp_indeces=None,
            enforce_z_up=False,
            modulo_object_z_rotation=False,
            make_quats_continuous=True,
        )  # type: ignore

    def _scan_demos(
        self, ids: list[int], plot: bool
    ) -> tuple[list[SceneObservation], list[int]]:  # type: ignore

        seg_cfg = self.cfg.policy.model.demos_segmentation
        expected = self._expected_velocity_pauses()
        out_dir = self.save_dir(self.cfg) / "plots" / "vel"
        out_dir.mkdir(parents=True, exist_ok=True)

        path = self.load_dir(self.cfg)
        kept: list[SceneObservation] = []  # type: ignore
        accepted_ids: list[int] = []
        with h5py.File(path / "demos.h5", "r") as demos_file:
            for demo_id in ids:
                demos_scenes, demos_images = self.scene.load_dataset(
                    demos_file,
                    selections=[demo_id],
                    with_images=not self._use_gt,
                )
                demo_scenes = demos_scenes[0]
                if self._use_gt:
                    td = self.dcscenes_to_tdtapas(demo_scenes)
                else:
                    demo_extracted: list[DCScene] = []
                    for idx, td_img in enumerate(demos_images[0]):
                        extracted = self.from_image(td_img)
                        extr_scene = DCScene(extracted, demo_scenes[idx].extras)
                        demo_extracted.append(extr_scene)
                    td = self.dcscenes_to_tdtapas(demo_extracted)

                vel = np.linalg.norm(td.action[:, :3].numpy().astype(float), axis=1)
                boundaries = self._velocity_segments(vel, seg_cfg)
                pause_ok = expected is None or len(boundaries) == expected
                if pause_ok:
                    accepted_ids.append(demo_id)

                opening_old = td.action[:, -1].numpy().astype(float).copy()
                if self.cfg.snap_ee_actions:
                    self._snap_gripper_per_segment(td)
                opening_new = td.action[:, -1].numpy().astype(float)

                if plot:
                    prefix = "accepted" if pause_ok else "discarded"
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(
                        np.arange(len(vel)),
                        vel,
                        linewidth=0.8,
                        c="gray",
                        label="velocity",
                    )
                    ax.axhline(
                        y=seg_cfg.velocity_threshold,
                        color="r",
                        label="threshold",
                        linewidth=0.8,
                    )
                    for idx, m in enumerate(boundaries):
                        if idx == 0:
                            ax.axvline(x=m, color="g", linestyle="--", label="segment")
                        else:
                            ax.axvline(x=m, color="g", linestyle="--")
                    ax.plot(
                        np.arange(len(opening_old)),
                        opening_old,
                        c="tab:purple",
                        linewidth=0.8,
                        linestyle="--",
                        label="gripper (raw)",
                    )
                    if self.cfg.snap_ee_actions:
                        ax.plot(
                            np.arange(len(opening_new)),
                            opening_new,
                            c="tab:blue",
                            linewidth=0.8,
                            label="gripper (snapped)",
                        )
                    ax.set_title(f"{prefix} — {self.cfg.tag} demo {demo_id}")
                    ax.set_xlabel("timestep")
                    ax.set_ylabel("velocity (position magnitude)")
                    ax.legend(fontsize=7)
                    fig.savefig(
                        out_dir / f"{prefix}_demo_{demo_id}.png",
                        dpi=150,
                        bbox_inches="tight",
                    )
                    plt.close(fig)

                if pause_ok:
                    kept.append(td)
                    if len(kept) >= self.cfg.max_demos:
                        break

        if plot:
            logger.info(
                f"Saved velocity plots to {out_dir} ({len(accepted_ids)} accepted)"
            )
        return kept, accepted_ids

    def _expected_velocity_pauses(self) -> int | None:
        gt = self.cfg.policy.model.frame_selection.gt_frames
        return None if gt is None else len(gt) - 1

    def _all_demo_ids(self) -> list[int]:
        path = self.load_dir(self.cfg) / "demos.h5"
        with h5py.File(path, "r") as f:
            demo = np.asarray(f["demo"][:])  # type: ignore
        if len(demo) == 0:
            return []
        return list(range(int(np.max(demo)) + 1))

    @staticmethod
    def _velocity_segments(
        vel: np.ndarray, seg_cfg: DemoSegmentationConfig
    ) -> list[int]:
        stop_indeces = np.argwhere(np.abs(vel) < seg_cfg.velocity_threshold).flatten()
        if len(stop_indeces) == 0:
            return []
        idx_diff = np.diff(stop_indeces)
        split_idx = np.argwhere(idx_diff > seg_cfg.max_idx_distance).flatten() + 1
        if len(split_idx) == 0:
            stop_segmented = [stop_indeces]
        else:
            stop_segmented = np.split(stop_indeces, split_idx)
        filtered = [c for c in stop_segmented if len(c) >= seg_cfg.min_len]
        segment_mean = [int(c.mean()) for c in filtered]
        return [
            m
            for m in segment_mean
            if m > seg_cfg.min_end_distance and m < len(vel) - seg_cfg.min_end_distance
        ]

    def _snap_gripper_per_segment(self, stacked: SceneObservation):  # type: ignore
        seg_cfg = self.cfg.policy.model.demos_segmentation
        opening = stacked.action[:, -1].numpy().astype(float)

        vel = np.linalg.norm(stacked.action[:, :3].numpy().astype(float), axis=1)
        boundaries = self._velocity_segments(vel, seg_cfg)
        edges = [0] + boundaries + [len(opening)]

        starts = np.asarray(edges[:-1])
        lengths = np.diff(edges)
        stepped = np.repeat(opening[starts], lengths)

        stacked.action[:, -1] = torch.tensor(stepped)

    def dcscenes_to_tdtapas(self, scenes: list[DCScene]) -> TensorDict:
        if self.cfg.fix_bimodal:
            self._fold_bimodal_yaw(scenes)
        obs: list[TensorDict] = []
        td_goal = scenes[-1]
        for td_scene in scenes:
            td_obs = td_scene
            td = self.tapas_td(td_obs, td_goal)
            obs.append(td)
        stacked_obs = TensorDict.stack(obs, dim=0)
        assert isinstance(stacked_obs, SceneObservation)
        return stacked_obs  # type: ignore

    def _target_label(self) -> str | None:
        label = None
        for candidate in self.scene.entities:
            if self.cfg.tag.startswith(candidate + "_") and (
                label is None or len(candidate) > len(label)
            ):
                label = candidate
        return label

    @staticmethod
    def _quat_yaw(q: np.ndarray) -> float:
        q = Quaternion.normalize(np.asarray(q, dtype=float))
        return float(
            np.arctan2(
                2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)
            )
        )

    def _fold_bimodal_yaw(self, scenes: list[DCScene]) -> None:
        label = self._target_label()
        if label is None:
            return

        q_z180 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)  # 180 deg about z
        rel = np.array(
            [
                self._quat_yaw(s.extras["ee_pose"][3:7]) - self._quat_yaw(s[label].rot)
                for s in scenes
            ]
        )
        rel = (rel + np.pi) % (2.0 * np.pi) - np.pi
        if isinstance(self.scene.entities[label], PrismaticEntity):
            # Prismatic targets keep a fixed world orientation, so the two grasp
            # orientations sit at rel ~ 0 and rel ~ +-180, exactly on the
            # sin(circular_mean) == 0 boundary of the test below, which makes the
            # fold decision degenerate. Use the median instead (|med| > 90 deg).
            fold = abs(float(np.median(rel))) > np.pi / 2.0
        else:
            mean_rel = float(np.arctan2(np.sin(rel).mean(), np.cos(rel).mean()))
            fold = np.sin(mean_rel) > 0.0
        if fold:
            for s in scenes:
                q = np.asarray(s.extras["ee_pose"][3:7], dtype=float)
                s.extras["ee_pose"][3:7] = Quaternion.mul(q_z180, q)

    def fit_conditions(self):
        logger.info(
            f"[{self.cfg.scene.tag}] Fitting conditions ({self._use_gt}): {self.cfg.tag}"
        )
        path = self.load_dir(self.cfg)
        cache_path = self.conditions_cache_path
        demos_file = h5py.File(path / f"demos.h5", "r")
        demos_scenes, demos_images = self.scene.load_dataset(
            demos_file,
            only_conditions=True,
            with_images=not self._use_gt,
        )

        pre_data: dict[str, np.ndarray] = {}
        post_data: dict[str, np.ndarray] = {}
        if self._use_gt:
            start_scenes = [demo[0] for demo in demos_scenes]
            end_scenes = [demo[-1] for demo in demos_scenes]
        else:
            start_scenes = [self.from_image(demo[0]) for demo in demos_images]
            end_scenes = [self.from_image(demo[-1]) for demo in demos_images]

        for key in self.entities:
            pre_data[key] = np.stack([s[key].value for s in start_scenes])
            post_data[key] = np.stack([s[key].value for s in end_scenes])

        pair = ConPair.make(self.cfg.tag, pre_data, post_data, self.entities)
        pair.plot(path)
        joblib.dump(pair, cache_path)
        logger.info(f"Saved conditions to {cache_path}")
