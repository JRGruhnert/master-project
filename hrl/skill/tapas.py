from enum import Enum
import json
import pathlib
import pprint
from loguru import logger
import numpy as np
import torch
from hrl.observation.observation import MPObservation
from hrl.skill import skill
from hrl.skill.skill import Skill
from hrl.state.state import State
from tapas_gmm.policy.policy import Policy
from tapas_gmm.utils.select_gpu import device
from tapas_gmm.policy import import_policy
from tapas_gmm.policy.gmm import GMMPolicy, GMMPolicyConfig
from tapas_gmm.policy.models.tpgmm import (
    AutoTPGMM,
    AutoTPGMMConfig,
    ModelType,
    TPGMMConfig,
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
        id: int,
        states: list[State],
        reversed: bool,
        override_keys: list[str],
    ):
        super().__init__(name, id)
        self._reversed = reversed
        self._override_keys = override_keys
        self._overrides: dict[str, np.ndarray] = {}
        self._policy: GMMPolicy = self._load_policy()
        self.initialize_conditions(states)
        self.initialize_overrides(states)


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

    def initialize_conditions(self, states: list[State], verbose: bool = False):
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
        for state in states:
            value = state.retrieve_precon(
                tpgmm.start_values[state.name],
                tpgmm.end_values[state.name],
                self._reversed,
                True if state.name in tapas_tp else False,
            )
            anti_value = state.retrieve_precon(
                tpgmm.start_values[state.name],
                tpgmm.end_values[state.name],
                not self._reversed,
                True if state.name in tapas_tp else False,
            )
            if value is not None:
                self.precons[state.name] = value
                self.postcons[state.name] = anti_value

        self.prepare()
        if verbose:
            logger.info(f"Initialized task parameters for {self.name}:")
            logger.info(
                "\n" + pprint.pformat(dict(self.precons), indent=2, width=80)
            )

    def initialize_overrides(self, states: list[State], verbose: bool = False):
        """
        Initialize the task parameters based on the active states.
        """
        # TODO: Its a copy of initialize_task_parameters but only override states get loaded and also in reverse
        # So basically normal since reversed is True
        tpgmm: AutoTPGMM = self._policy.model
        for state in states:
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
        if verbose:
            logger.info(f"Initialized overrides for {self.name}:")
            logger.info(
                "\n" + pprint.pformat(dict(self._overrides), indent=2, width=80)
            )

    def prepare(self, predict_as_batch: bool = True, control_duration: int = -1):
        super().__init__(predict_as_batch, control_duration)
        self._policy.reset_episode()

    def predict(self, current, states):
        if self._predict_as_batch:
            if self._current_step == 0:
                # Batch prediction for the given observation
                try:
                    predictions, _ = self.policy.predict(
                        self.make_tapas_format(self.current_calvin, skill, self.goal)
                    )
            except Exception as e:
                logger.error(f"Error during batch prediction: {e}")
                return None
        else:

        prediction, _ = self.policy.predict(
            self.make_tapas_format(self.current_calvin, skill, self.goal)
        )
        #TODO Gripper state
        return np.concatenate(
                (
                    prediction.ee,
                    prediction.gripper,
                )
            )
