# src/integrations/taco/calvin_transforms.py
from tacorl.utils.transforms import TransformManager


class CalvinTransformManager(TransformManager):
    def __call__(self, seq_dict, transf_type="train"):
        # Your custom preprocessing + key renaming
        processed_dict = {}

        # Process and rename CALVIN observations
        if "rgb_obs" in seq_dict:
            # Extract nested RGB and give it a new key
            rgb_data = seq_dict["rgb_obs"]
            if isinstance(rgb_data, dict) and "rgb_static" in rgb_data:
                processed_dict["processed_rgb"] = self.process_rgb(
                    rgb_data["rgb_static"]
                )
                processed_dict["gripper_rgb"] = rgb_data.get("rgb_gripper", None)

        if "robot_obs" in seq_dict:
            # Process robot state and rename
            processed_dict["robot_state"] = self.process_robot_obs(
                seq_dict["robot_obs"]
            )
            processed_dict["ee_pose"] = self.extract_ee_pose(seq_dict["robot_obs"])

        if "rel_actions_world" in seq_dict:
            # Keep actions but maybe rename
            processed_dict["actions"] = seq_dict["rel_actions_world"]

        # You can add completely new keys too!
        processed_dict["combined_obs"] = self.combine_modalities(processed_dict)

        return processed_dict

    def process_rgb(self, rgb_data):
        # Your RGB preprocessing
        # Normalize, resize, extract features, etc.
        return rgb_data  # placeholder

    def process_robot_obs(self, robot_obs):
        # Your robot state preprocessing
        return robot_obs  # placeholder
