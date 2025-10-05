import pathlib
import re
from loguru import logger
import numpy as np
import torch
from calvin_env.envs.observation import CalvinEnvObservation
from hrl.common.tapas_state import TapasState
from hrl.env.calvin.calvin_observation import CalvinObservation
from hrl.common.skill import Skill
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


class TapasSkill(Skill):
    def __init__(
        self,
        name: str,
        id: int,
        reversed: bool,
        overrides: list[str],
    ):
        super().__init__(name, id)
        self.reversed = reversed
        self.overrides = overrides
        self.policy_name = "gmm"
        self.overrides_dict: dict[str, np.ndarray] = {}
        self.policy: GMMPolicy = self._load_policy()
        self.first_prediction = True
        self.predictions: list = []

    def _policy_checkpoint_name(self) -> pathlib.Path:
        return (
            pathlib.Path("data")
            / "skills"
            / "tapas"
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
            invert_prediction_batch=self.reversed,
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

    def initialize_conditions(self, states: list[TapasState]):
        """
        Initialize the task parameters based on the active states.
        """
        tpgmm: AutoTPGMM = self.policy.model
        # Taskparameters of the AutoTPGMM model
        tapas_tp: set[str] = set()
        for _, segment in enumerate(tpgmm.segment_frames):
            for _, frame_idx in enumerate(segment):
                pos_str, rot_str = tpgmm.frame_mapping[frame_idx]
                tapas_tp.add(pos_str)
                tapas_tp.add(rot_str)
        # TODO: Currently assumes tapas tps are euler and quaternion
        # My whole code does not generalize to other Task Parameterized models and state types
        for state in states:
            pre_value = state.make_additional_tps(
                tpgmm.start_values[state.name],
                tpgmm.end_values[state.name],
                self.reversed,
                True if state.name in tapas_tp else False,
            )
            post_value = state.make_additional_tps(
                tpgmm.start_values[state.name],
                tpgmm.end_values[state.name],
                not self.reversed,
                True if state.name in tapas_tp else False,
            )
            if pre_value is not None:
                self.precons[state.name] = pre_value
            if post_value is not None:
                self.postcons[state.name] = post_value

    def initialize_overrides(self, states: list[TapasState]):
        """
        Initialize the task parameters based on the active states.
        """
        # NOTE: Its a copy of initialize_task_parameters but only override states get loaded and also in reverse
        # So basically normal since reversed is True
        tpgmm: AutoTPGMM = self.policy.model
        for state in states:
            if state.name in self.overrides:
                value = state.make_additional_tps(
                    tpgmm.start_values[state.name],
                    tpgmm.end_values[state.name],
                    not self.reversed,  # NOTE: We want the opposite of the reverse trajectory
                    True,  # NOTE: All States are True here
                )
                if value is None:
                    raise ValueError(
                        f"Failed to create override for state {state.name}. This should not happen."
                    )
                self.overrides_dict[state.name] = value.numpy()

    def reset(self, env, predict_as_batch: bool = True, control_duration: int = -1):
        super().reset(predict_as_batch, control_duration)
        self.policy.reset_episode(env)
        self.first_prediction = True
        self.predictions = []

    def predict(
        self,
        current: CalvinEnvObservation,
        goal: CalvinObservation,
        states: list[TapasState],
    ) -> np.ndarray:
        if self.predict_as_batch:
            if self.first_prediction:
                # NOTE: Could use control_duration later to enforce certain length
                try:
                    self.predictions, _ = self.policy.predict(
                        self.to_skill_format(current, goal, states)
                    )
                except FloatingPointError as e:
                    logger.error(f"Numerical error in GMM prediction: {e}")
                    # Return a safe default action (e.g., no movement)
                    return None  # TODO: I think its just cause of a bad robot position
                except Exception as e:
                    logger.error(f"Error in skill prediction: {e}")
                    return None
                self.first_prediction = False
            if self.predictions.is_finished:
                return None
            return self._to_action(self.predictions.step())
        else:
            # NOTE: DID NOT TEST YET but is theoretical available in Tapas
            raise NotImplementedError("Non-batch prediction not implemented yet.")

    def _to_action(self, prediction) -> np.ndarray:
        return np.concatenate(
            (
                prediction.ee,
                prediction.gripper,
            )
        )

    def to_skill_format(self, obs: CalvinEnvObservation, goal: CalvinObservation = None, states: list[TapasState] = None) -> SceneObservation:  # type: ignore
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
        if goal is not None and self.reversed:
            states_dict = {state.name: state for state in states} if states else {}
            # NOTE: This is only a hack to make reversed tapas models work
            # TODO: Update this when possible
            # logger.debug(f"Overriding Tapas Task {task.name}")
            for state_name, state_value in self.overrides_dict.items():
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

                elif match_position:
                    position_state_name = f"{match_position.group(1)}_position"
                    if position_state_name not in states_dict:
                        temp_state = states_dict[position_state_name]
                        temp_pos = temp_state.area_tapas_override(
                            goal.top_level_observation[position_state_name],
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
                            goal.top_level_observation[
                                f"{match_rotation.group(1)}_rotation"
                            ].numpy(),
                        ]
                    )
                elif match_scalar:
                    object_states_dict[match_scalar.group(1)] = (
                        goal.top_level_observation[
                            f"{match_scalar.group(1)}_scalar"
                        ].numpy()
                    )
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
                name: (
                    torch.tensor([state]) if np.isscalar(state) else torch.tensor(state)
                )
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
