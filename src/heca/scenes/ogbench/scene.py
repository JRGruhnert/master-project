from collections import defaultdict
from dataclasses import dataclass
import atexit
import time
from typing import Any, cast

from gym import Env
import h5py
import numpy as np
import ogbench
import torch
from ogbench.manipspace.envs.scene_env_base import SceneEnvBase

from heca.data.data import DCEntity, DCScene, TDImage
from heca.misc import logger
from heca.scenes.scene import Scene, SceneFeedback


class OGScene(Scene):
    @dataclass(kw_only=True)
    class Config(Scene.Config):
        label: str = "ogbench"
        vis: bool
        tag: str
        viewer: bool = False
        frame_time: float = 0.05

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.env_id = "vi-" + cfg.tag if cfg.vis else "gt-" + cfg.tag
        self._env = self._make_env(self.env_id)
        self._last_ee_pose = None
        self._last_ee_yaw = None
        self._viewer_launched = False
        self._meta_xyz_center = np.array([0.425, 0.0, 0.0], dtype=np.float32)
        self._meta_xyz_scaler = np.array([10.0], dtype=np.float32)

    def _update_meta(self, obs: dict):
        """Parse the workspace center/scaler from an env obs dict."""
        if "meta_xyz_center" in obs and "meta_xyz_scaler" in obs:
            self._meta_xyz_center = np.asarray(obs["meta_xyz_center"], dtype=np.float32)
            self._meta_xyz_scaler = np.asarray(obs["meta_xyz_scaler"], dtype=np.float32)

    def normalize_position(self, pos) -> np.ndarray:
        """Map a world-frame position into the scene's normalized frame."""
        return (
            np.asarray(pos, dtype=np.float32) - self._meta_xyz_center
        ) * self._meta_xyz_scaler

    def unnormalize_position(self, pos) -> np.ndarray:
        """Inverse of ``normalize_position`` (raw = norm / scaler + center)."""
        return (
            np.asarray(pos, dtype=np.float32) / self._meta_xyz_scaler
        ) + self._meta_xyz_center

    def _make_env(self, env_id: str) -> Env:
        return cast(
            Env,
            ogbench.make_env_and_datasets(
                dataset_name=env_id,
                env_only=True,
                mode="randomized",
                dataset_only=False,
                # control_timestep=0.5,
            ),
        )

    @property
    def env(self) -> SceneEnvBase:
        return cast(SceneEnvBase, self._env.unwrapped)

    def _sync_viewer(self):
        """Launch (once) and sync the passive viewer after a step/reset."""
        if not self.cfg.viewer:
            return
        if not self._viewer_launched:
            self.env.launch_passive_viewer()
            self._viewer_launched = True
            atexit.register(self.close_viewer)
        self.env.sync_passive_viewer()
        if self.cfg.frame_time > 0:
            time.sleep(self.cfg.frame_time)

    def close_viewer(self):
        """Close the passive viewer if it was launched (idempotent)."""
        if not self._viewer_launched:
            return
        try:
            self.env.close_passive_viewer()
        except Exception as e:
            logger.warning(f"Failed to close passive viewer: {e}")
        finally:
            self._viewer_launched = False

    def close(self):
        self.close_viewer()
        self._env.close()

    def to_td_image(self, obs: dict) -> TDImage:
        image_dict = obs["image"]
        if not isinstance(image_dict, dict):
            empty = torch.empty(0)
            return TDImage(
                rgb=empty.clone(),
                d=empty.clone(),
                mask=torch.empty(0, dtype=torch.uint8),
                extr=empty.clone(),
                intr=empty.clone(),
            )

        rgb = image_dict["rgb"].transpose((2, 0, 1)) / 255
        depth = image_dict["depth"]
        mask = image_dict["mask"]
        extr = image_dict["extrinsics"]
        intr = image_dict["intrinsics"]

        return TDImage(
            rgb=torch.Tensor(rgb),
            d=torch.Tensor(depth),
            mask=torch.Tensor(mask).to(torch.uint8),
            extr=torch.Tensor(extr),
            intr=torch.Tensor(intr),
        )

    def to_np_image(self, obs: dict) -> np.ndarray:
        image_dict = obs["image"]
        if not isinstance(image_dict, dict):
            return np.zeros((0,), dtype=np.uint8)
        return image_dict["rgb"]

    def get_extras(self, obs: dict) -> dict[str, Any]:
        pos = obs["proprio_effector_pos"]
        rot = obs["proprio_effector_quat"]
        ste = obs["proprio_gripper_opening"]
        rot = np.array(rot, dtype=np.float32)
        ee_pos = self.normalize_position(pos)
        ee_pose = np.concatenate((ee_pos, rot))
        if "actions" in obs.keys():  # is demo
            yaw = obs["proprio_effector_yaw"].item()
            if self._last_ee_pose is not None:
                pos_delta = ee_pos - self._last_ee_pose
                yaw_delta = self._wrap_angle(yaw - self._last_ee_yaw)
            else:
                pos_delta = np.zeros_like(ee_pos)
                yaw_delta = 0.0
            self._last_ee_pose = ee_pos
            self._last_ee_yaw = yaw
            axis_angle = np.array([0, 0, yaw_delta], dtype=np.float32)
            action = np.concatenate([pos_delta, axis_angle, ste])
            reward = obs["success"]
        else:
            yaw = obs["proprio_effector_yaw"].item()
            action = np.concatenate([pos, np.array([0, 0, yaw]), ste])
            reward = np.array([0])
        return {
            "action": action,
            "reward": reward,
            "ee_pose": ee_pose,
            "gripper_state": np.atleast_1d(obs["proprio_gripper_state"]),
            "joint_pos": obs["proprio_joint_pos"],
            "joint_vel": obs["proprio_joint_vel"],
        }

    def to_internal(self, obs: Any, info: dict[str, Any]) -> Any:
        goal = info.pop("goal", None)
        goal_rendered = info.pop("goal_rendered", None)
        info["image"] = obs
        if goal is not None:
            goal["image"] = goal_rendered
        return info, goal

    def _sample_task(
        self,
    ) -> tuple[
        tuple[DCScene, TDImage],
        tuple[DCScene, TDImage],
    ]:
        ob, info = self._env.reset(options={"render_goal": True})
        obs, goal = self.to_internal(ob, info)
        self._update_meta(obs)
        self.last_pos = obs["proprio_effector_pos"]
        self.last_rot = obs["proprio_effector_yaw"]
        self.last_ste = obs["proprio_gripper_opening"]
        s_scene, s_image, _ = self.from_internal(obs)
        g_scene, g_image, _ = self.from_internal(goal)
        self._sync_viewer()
        return (s_scene, s_image), (g_scene, g_image)

    def get_ee_dc(self, obs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos = obs["proprio_effector_pos"]
        yaw = obs["proprio_effector_yaw"].item()
        rot = self.yaw_to_quat(yaw)
        ste_idx = np.atleast_1d(obs["proprio_gripper_state"])
        return pos, rot, ste_idx

    def yaw_to_quat(self, yaw: float) -> np.ndarray:
        half_yaw = yaw / 2
        return np.array([np.cos(half_yaw), 0, 0, np.sin(half_yaw)], dtype=np.float32)

    def quat_to_yaw(self, quat: np.ndarray) -> float:
        # quat is in (w, x, y, z) format.
        siny_cosp = 2 * (quat[0] * quat[3] + quat[1] * quat[2])
        cosy_cosp = 1 - 2 * (quat[2] ** 2 + quat[3] ** 2)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return yaw

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def to_internal_action(self, action: np.ndarray) -> np.ndarray:
        pos = self.unnormalize_position(action[:3])
        quat = action[3:7]
        yaw = self.quat_to_yaw(quat)
        state = action[7]
        result = np.concatenate([pos, [yaw], [state]], axis=0)
        return result

    def _step(self, action: np.ndarray) -> tuple[Any, SceneFeedback]:
        action = self.to_internal_action(action)
        ob, reward, terminated, truncated, info = self.env.step(action, False, True)  # type: ignore
        obs, _ = self.to_internal(ob, info)
        self._sync_viewer()
        assert isinstance(reward, float)
        return obs, SceneFeedback(
            terminal=terminated, reward=reward, truncated=truncated
        )

    def _step_virt(
        self,
        x: DCScene,
        y: DCScene,
        elabels: list[str],
        rand_noise: float,
        fail_noise: float,
    ) -> tuple[Any, SceneFeedback]:
        subgoal: dict[str, Any] = {}
        for label in elabels:
            subgoal.update(
                self.entities[label].env_state_value(
                    label, y, unnormalize_pos=self.unnormalize_position
                )
            )
        ob, reward, terminated, truncated, info = self.env.step_scene(
            subgoal, rand_noise, fail_noise
        )
        obs, _ = self.to_internal(ob, info)
        self._sync_viewer()
        assert isinstance(reward, float)
        return obs, SceneFeedback(
            terminal=terminated, reward=reward, truncated=truncated
        )

    def load_dataset(
        self,
        file: h5py.File,
        selections: list[int] | None = None,
        only_conditions: bool = False,
        with_images: bool = True,
    ) -> tuple[list[list[DCScene]], list[list[TDImage]]]:
        demo_indices: np.ndarray = file["demo"][:]  # type: ignore

        change_points = np.where(np.diff(demo_indices) != 0)[0] + 1
        starts = np.concatenate([[0], change_points])
        ends = np.concatenate([change_points, [len(demo_indices)]])

        image_keys = {"rgb", "depth", "mask", "extrinsics", "intrinsics"}

        segments_scene: list[list[DCScene]] = []
        segments_image: list[list[TDImage]] = []
        if selections is None:
            selections = list(range(len(starts)))

        for episode_idx in selections:
            start = starts[episode_idx]
            end = ends[episode_idx]
            self._last_ee_pose = None
            self._last_ee_yaw = None

            segment_scene: list[DCScene] = []
            segment_image: list[TDImage] = []
            if only_conditions:
                indices = [start, end - 1]
            else:
                indices = range(start, end)
            for i in indices:
                ob = {
                    key: file[key][i]  # type: ignore
                    for key in file.keys()
                    if key not in image_keys | {"demo"}
                }

                if with_images:
                    image = dict(
                        rgb=file["rgb"][i],  # type: ignore
                        depth=file["depth"][i],  # type: ignore
                        mask=file["mask"][i],  # type: ignore
                        extrinsics=file["extrinsics"][i],  # type: ignore
                        intrinsics=file["intrinsics"][i],  # type: ignore
                    )
                    obs, _ = self.to_internal(image, ob)
                    dc_scene, td_image, _ = self.from_internal(obs)
                    segment_image.append(td_image)
                else:
                    dc_scene = self.to_dc_scene(ob)
                segment_scene.append(dc_scene)
            segments_scene.append(segment_scene)
            segments_image.append(segment_image)

        return segments_scene, segments_image

    def to_dc_scene(self, obs: dict) -> DCScene:
        self._update_meta(obs)
        dc_entities: dict[str, DCEntity] = {}
        for label, entity in self.entities.items():
            dc_entities[label] = entity.value_from_gt(
                label, obs, normalize_pos=self.normalize_position
            )
        extras = self.get_extras(obs)
        return DCScene(dc_entities, extras=extras)

    def demo_auto_extract(self):
        scene_path = Scene.load_dir(self.cfg)
        with h5py.File(scene_path / f"{self.env_id}.h5", "r") as f:
            done_ds = f["oracle_done"]
            assert isinstance(done_ds, h5py.Dataset)
            done = np.asarray(done_ds)[:, 0]

            success_ds = f["oracle_success"]
            assert isinstance(success_ds, h5py.Dataset)
            success = np.asarray(success_ds)[:, 0]

            task_ds = f["privileged_target_task"]
            assert isinstance(task_ds, h5py.Dataset)
            task = np.asarray(task_ds, dtype=str)

            start_ds = f["oracle_start"]
            assert isinstance(start_ds, h5py.Dataset)
            start = np.asarray(start_ds)[:, 0]

            start_idxs = np.where(start == 1.0)[0]

            agent_data = defaultdict(list)
            discarded = defaultdict(int)

            for i in range(len(start_idxs) - 1):
                s = start_idxs[i]
                e = start_idxs[i + 1] - 1  # inclusive end / oracle_done boundary
                if s >= e:
                    continue

                label = str(task[e])
                agent_key = self.entities[label].make_agent_key(label, f, s, e)

                if success[e] != 1.0:
                    discarded[agent_key] += 1
                    continue

                agent_data[agent_key].append((s, e))

            all_keys = list(f.keys())
            for agent_key, segments in agent_data.items():
                out_path = scene_path / "experts" / agent_key / f"demos.h5"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with h5py.File(out_path, "w") as out:
                    demo_id = 0
                    for s, e in segments:
                        demo_ids = np.full(e - s + 1, demo_id, dtype=np.int32)
                        for key in all_keys:
                            ds = f[key]
                            if not isinstance(ds, h5py.Dataset):
                                continue
                            data = np.asarray(ds[s : e + 1])
                            if key == "demo":
                                continue  # demo ids are written below
                            if key not in out:
                                maxshape = (None,) + data.shape[1:]
                                out.create_dataset(
                                    key,
                                    data=data,
                                    maxshape=maxshape,
                                    chunks=(1,) + data.shape[1:],
                                    compression="gzip",
                                    compression_opts=4,
                                )
                            else:
                                ds = out[key]
                                assert isinstance(ds, h5py.Dataset)
                                n = ds.shape[0]
                                ds.resize(n + len(data), axis=0)
                                ds[n : n + len(data)] = data
                        # Add demo_id dataset
                        if "demo" not in out:
                            out.create_dataset(
                                "demo", data=demo_ids, maxshape=(None,), chunks=(1,)
                            )
                        else:
                            ds = out["demo"]
                            assert isinstance(ds, h5py.Dataset)
                            n = ds.shape[0]
                            ds.resize(n + len(demo_ids), axis=0)
                            ds[n : n + len(demo_ids)] = demo_ids
                        demo_id += 1

                print(
                    f"  {agent_key}: kept={len(segments)}, "
                    f"discarded={discarded.get(agent_key, 0)}, demo_path={out_path}"
                )

            # Print discarded-only keys (if any).
            for agent_key, count in discarded.items():
                if agent_key not in agent_data:
                    print(f"  {agent_key}: kept=0, discarded={count}")
