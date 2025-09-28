import json
import pathlib
import re
from loguru import logger
import numpy as np
import torch
from calvin_env.envs.observation import CalvinObservation
from hrl.observation.calvin_tapas import MasterCalvinObs
from hrl.skill.skill import Skill
from hrl.state.state import State
from tapas_gmm.utils.select_gpu import device
from tapas_gmm.policy import import_policy
from tapas_gmm.policy.gmm import GMMPolicy, GMMPolicyConfig
from tapas_gmm.policy.models.tpgmm import (
    AutoTPGMM,
    AutoTPGMMConfig,
    ModelType,
    TPGMMConfig,
)
from tapas_gmm.utils.observation import (
    CameraOrder,
    SceneObservation,
    SingleCamObservation,
    dict_to_tensordict,
    empty_batchsize,
)


class Tapas(Skill):
    @classmethod
    def from_json_list(cls, skill_space: str, relative_path: str) -> list["Skill"]:
        """Convert a SkillSpace to a list of Skill objects by reading from skills.json"""
        # Load the skills.json file
        path = pathlib.Path(relative_path)

        if not path.exists():
            raise FileNotFoundError(f"Skills JSON file not found at {path}")

        with open(path, "r") as f:
            data: dict = json.load(f)

        # Filter skills based on the requested skill space
        filtered = []
        for skill_key, skill_value in data.items():
            # Check if this skill belongs to the requested space
            skill_space_list = skill_value.get("space")
            if skill_space_list is None:
                raise ValueError(f"Skill {skill_key} does not have a 'space' defined.")

            if skill_space in skill_space_list:
                skill = cls(name=skill_key, **skill_value)
                filtered.append(skill)

        return filtered

    def __init__(
        self,
        name: str,
        policy_name: str,
        id: int,
        states: list[State],
        reversed: bool,
        override_keys: list[str],
    ):
        super().__init__(name, id)
        self._reversed = reversed
        self._override_keys = override_keys
        self.policy_name = policy_name
        self._states = states
        self._overrides: dict[str, np.ndarray] = {}
        self._policy: GMMPolicy = self._load_policy()
        self._initialize_conditions()
        self._initialize_overrides()
        self.prepare()

    def _policy_checkpoint_name(self) -> pathlib.Path:
        return (
            pathlib.Path("data")
            / self.name
            / ("demos" + "_" + "gmm" + "_policy" + "-release")
        ).with_suffix(".pt")

    def _get_config(self) -> GMMPolicyConfig:
        """
        Get the configuration for the OpenDrawer policy.
        """
        return GMMPolicyConfig(
            suffix="release",
            model=AutoTPGMMConfig(
                tpgmm=TPGMMConfig(
                    n_components=20,
                    model_type=ModelType.HMM,
                    use_riemann=True,
                    add_time_component=True,
                    add_action_component=False,
                    position_only=False,
                    add_gripper_action=True,
                    reg_shrink=1e-2,
                    reg_diag=2e-4,
                    reg_diag_gripper=2e-2,
                    reg_em_finish_shrink=1e-2,
                    reg_em_finish_diag=2e-4,
                    reg_em_finish_diag_gripper=2e-2,
                    trans_cov_mask_t_pos_corr=False,
                    em_steps=1,
                    fix_first_component=True,
                    fix_last_component=True,
                    reg_init_diag=5e-4,  # 5
                    heal_time_variance=False,
                ),
            ),
            time_based=True,
            predict_dx_in_xdx_models=False,
            binary_gripper_action=True,
            binary_gripper_closed_threshold=0.95,
            dbg_prediction=False,
            # the kinematics model in RLBench is just to unreliable -> leads to mistakes
            topp_in_t_models=False,
            force_overwrite_checkpoint_config=True,  # TODO:  otherwise it doesnt work
            time_scale=1.0,
            # ---- Changing often ----
            postprocess_prediction=False,  # TODO:  abs quaternions if False else delta quaternions
            return_full_batch=True,
            batch_predict_in_t_models=True,  # Change if visualization is needed
            invert_prediction_batch=self._reversed,
        )

    def _load_policy(self) -> GMMPolicy:
        PolicyClass = import_policy(self.policy_name)
        config = self._get_config()
        policy: GMMPolicy = PolicyClass(config).to(device)

        file_name = self._policy_checkpoint_name()  # type: ignore
        logger.info("Loading policy checkpoint from {}", file_name)
        policy.from_disk(file_name)
        policy.eval()
        return policy

    def _initialize_conditions(self):
        """
        Initialize the task parameters based on the active states.
        """
        tpgmm: AutoTPGMM = self._policy.model
        # Taskparameters of the AutoTPGMM model
        tapas_tp: set[str] = set()
        for _, segment in enumerate(tpgmm.segment_frames):
            for _, frame_idx in enumerate(segment):
                pos_str, rot_str = tpgmm.frame_mapping[frame_idx]
                tapas_tp.add(pos_str)
                tapas_tp.add(rot_str)
        # TODO: Currently assumes tapas tps are euler and quaternion
        # My whole code does not generalize to other Task Parameterized models and state types
        for state in self._states:
            pre_value = state.retrieve_precon(
                tpgmm.start_values[state.name],
                tpgmm.end_values[state.name],
                self._reversed,
                True if state.name in tapas_tp else False,
            )
            post_value = state.retrieve_precon(
                tpgmm.start_values[state.name],
                tpgmm.end_values[state.name],
                not self._reversed,
                True if state.name in tapas_tp else False,
            )
            if pre_value is not None:
                self.precons[state.name] = pre_value
            if post_value is not None:
                self.postcons[state.name] = post_value

    def _initialize_overrides(self):
        """
        Initialize the task parameters based on the active states.
        """
        # TODO: Its a copy of initialize_task_parameters but only override states get loaded and also in reverse
        # So basically normal since reversed is True
        tpgmm: AutoTPGMM = self._policy.model
        for state in self._states:
            if state.name in self._override_keys:
                value = state.retrieve_precon(
                    tpgmm.start_values[state.name],
                    tpgmm.end_values[state.name],
                    not self._reversed,  # NOTE: We want the opposite of the reverse trajectory
                    True,  # NOTE: All States are True here
                )
                if value is None:
                    raise ValueError(
                        f"Failed to create override for state {state.name}. This should not happen."
                    )
                self._overrides[state.name] = value.numpy()

    def prepare(self, predict_as_batch: bool = True, control_duration: int = -1):
        super().__init__(predict_as_batch, control_duration)
        self._policy.reset_episode()

    def predict(
        self,
        current: MasterCalvinObs,
        goal: MasterCalvinObs,
    ) -> np.ndarray:
        self.current_step += 1
        if self.predict_as_batch:  # We currently only support batch prediction
            if self.current_step == 0:
                # Batch prediction for the given observation
                # NOTE: Could use control_duration later to enforce certain length
                try:
                    self.predictions, _ = self._policy.predict(
                        self.to_skill_format(current, goal)
                    )
                except Exception as e:
                    logger.error(f"Error during batch prediction: {e}")
                    return None
            if len(self.predictions) == 0 or self.current_step >= len(self.predictions):
                return None
            return self._to_action(self.predictions[self.current_step])
        else:
            # TODO: DID NOT TEST YET
            raise NotImplementedError("Non-batch prediction not implemented yet.")
            # prediction, _ = self._policy.predict(self.to_format(current, goal))
            # return self._to_action(prediction)

    def _to_action(self, prediction) -> np.ndarray:
        return np.concatenate(
            (
                prediction.ee,
                prediction.gripper,
            )
        )

    def to_skill_format(self, obs: CalvinObservation, goal: MasterCalvinObs = None) -> SceneObservation:  # type: ignore
        """
        Convert the observation from the environment to a SceneObservation. This format is used for TAPAS.

        Returns
        -------
        SceneObservation
            The observation in common format as SceneObservation.
        """
        if obs.action is None:
            action = None
        else:
            action = torch.Tensor(obs.action)

        if obs.reward is None:
            reward = torch.Tensor([0.0])
        else:
            reward = torch.Tensor([obs.reward])
        joint_pos = torch.Tensor(obs.joint_pos)
        joint_vel = torch.Tensor(obs.joint_vel)
        ee_pose = torch.Tensor(obs.ee_pose)
        ee_state = torch.Tensor([obs.ee_state])
        camera_obs = {}
        for cam in obs.camera_names:
            rgb = obs.rgb[cam].transpose((2, 0, 1)) / 255
            mask = obs.mask[cam].astype(int)

            camera_obs[cam] = SingleCamObservation(
                **{
                    "rgb": torch.Tensor(rgb),
                    "depth": torch.Tensor(obs.depth[cam]),
                    "mask": torch.Tensor(mask).to(torch.uint8),
                    "extr": torch.Tensor(obs.extr[cam]),
                    "intr": torch.Tensor(obs.intr[cam]),
                },
                batch_size=empty_batchsize,
            )

        multicam_obs = dict_to_tensordict(
            {"_order": CameraOrder._create(obs.camera_names)} | camera_obs
        )
        object_poses_dict = obs.object_poses
        object_states_dict = obs.object_states
        if goal is not None and self._reversed:
            # NOTE: This is only a hack to make reversed tapas models work
            # TODO: Update this when possible
            # logger.debug(f"Overriding Tapas Task {task.name}")
            for state_name, state_value in self._overrides.items():
                match_position = re.search(r"(.+?)_(?:position)", state_name)
                match_rotation = re.search(r"(.+?)_(?:rotation)", state_name)
                match_scalar = re.search(r"(.+?)_(?:scalar)", state_name)
                if state_name == "ee_position":
                    ee_pose = torch.cat(
                        [
                            torch.Tensor(state_value),
                            ee_pose[3:],
                        ]
                    )
                elif state_name == "ee_rotation":
                    ee_pose = torch.cat(
                        [
                            ee_pose[:3],
                            torch.Tensor(state_value),
                        ]
                    )
                elif state_name == "ee_scalar":
                    ee_state = torch.Tensor(state_value)

                # TODO: Evaluate if goal state is correct here
                elif match_position:
                    temp_pos = self.states[0].area_tapas_override(
                        goal.states[f"{match_position.group(1)}_position"],
                        self.spawn_surfaces,
                    )
                    object_poses_dict[match_position.group(1)] = np.concatenate(
                        [
                            temp_pos.numpy(),
                            object_poses_dict[match_position.group(1)][3:],
                        ]
                    )
                elif match_rotation:
                    object_poses_dict[match_rotation.group(1)] = np.concatenate(
                        [
                            object_poses_dict[match_rotation.group(1)][:3],
                            goal.states[f"{match_rotation.group(1)}_rotation"].numpy(),
                        ]
                    )
                elif match_scalar:
                    object_states_dict[match_scalar.group(1)] = goal.states[
                        f"{match_scalar.group(1)}_scalar"
                    ].numpy()
                else:
                    raise ValueError(f"Unknown state name: {state_name}")

        object_poses = dict_to_tensordict(
            {
                name: torch.Tensor(pose)
                for name, pose in sorted(object_poses_dict.items())
            },
        )
        object_states = dict_to_tensordict(
            {
                name: torch.Tensor([state])
                for name, state in sorted(object_states_dict.items())
            },
        )

        return SceneObservation(
            feedback=reward,
            action=action,
            cameras=multicam_obs,
            ee_pose=ee_pose,
            gripper_state=ee_state,
            object_poses=object_poses,
            object_states=object_states,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            batch_size=empty_batchsize,
        )
