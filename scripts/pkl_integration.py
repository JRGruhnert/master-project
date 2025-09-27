"""
Integration between Pkl configurations and existing State classes.
This bridges the gap between configuration and runtime objects.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import your existing classes
from master_project.data_types.state import (
    State,
    StateDistance,
    StateSuccess,
    StateSpace,
)
import torch


class PklStateFactory:
    """Factory to create State objects from Pkl configurations."""

    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)

    def load_pkl_as_json(self, pkl_file: str) -> Dict[str, Any]:
        """Load and evaluate a Pkl file, returning JSON data."""
        pkl_path = self.config_dir / f"{pkl_file}.pkl"

        try:
            result = subprocess.run(
                ["pkl", "eval", "-f", "json", str(pkl_path)],
                capture_output=True,
                text=True,
                check=True,
            )
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error evaluating {pkl_path}: {e.stderr}")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from {pkl_path}: {e}")
            return {}

    def create_state_from_config(
        self, state_name: str, state_config: Dict[str, Any]
    ) -> Optional[State]:
        """Create a State object from configuration dictionary."""
        try:
            # Map configuration to State parameters
            state_type = StateDistance(state_config["type"])
            success_mode = StateSuccess(state_config["success_mode"])
            lower_bound = torch.tensor(state_config["lower_bound"], dtype=torch.float32)
            upper_bound = torch.tensor(state_config["upper_bound"], dtype=torch.float32)

            # Create State using your existing factory method
            return State._create_state_by_type(
                name=state_name,
                state_type=state_type,
                success_mode=success_mode,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )
        except Exception as e:
            print(f"Error creating state {state_name}: {e}")
            return None

    def load_states_for_space(self, state_space: StateSpace) -> List[State]:
        """Load all states for a given StateSpace from Pkl configuration."""
        config_data = self.load_pkl_as_json("complete_states")

        if not config_data:
            return []

        states = []
        state_configs = config_data.get("states", {})
        space_mapping = config_data.get("state_spaces", {})

        # Get state names for this space
        state_names = space_mapping.get(state_space.value, [])

        for state_name in state_names:
            if state_name in state_configs:
                state_obj = self.create_state_from_config(
                    state_name, state_configs[state_name]
                )
                if state_obj:
                    states.append(state_obj)

        return states

    def load_all_states(self) -> Dict[str, State]:
        """Load all states from configuration."""
        config_data = self.load_pkl_as_json("complete_states")

        if not config_data:
            return {}

        states = {}
        state_configs = config_data.get("states", {})

        for state_name, state_config in state_configs.items():
            state_obj = self.create_state_from_config(state_name, state_config)
            if state_obj:
                states[state_name] = state_obj

        return states

    def validate_configuration(self, config_name: str = "complete_states") -> bool:
        """Validate that the Pkl configuration can create valid State objects."""
        config_data = self.load_pkl_as_json(config_name)

        if not config_data:
            return False

        state_configs = config_data.get("states", {})
        valid_count = 0
        total_count = len(state_configs)

        for state_name, state_config in state_configs.items():
            try:
                # Validate configuration structure
                required_fields = ["type", "success_mode", "lower_bound", "upper_bound"]
                if not all(field in state_config for field in required_fields):
                    print(f"❌ State {state_name} missing required fields")
                    continue

                # Validate enum values
                StateDistance(state_config["type"])
                StateSuccess(state_config["success_mode"])

                # Validate bounds
                lower = state_config["lower_bound"]
                upper = state_config["upper_bound"]
                if not isinstance(lower, list) or not isinstance(upper, list):
                    print(f"❌ State {state_name} bounds must be lists")
                    continue

                if len(lower) != len(upper):
                    print(f"❌ State {state_name} bounds must have same length")
                    continue

                valid_count += 1
                print(f"✅ State {state_name} is valid")

            except Exception as e:
                print(f"❌ State {state_name} validation failed: {e}")

        success_rate = valid_count / total_count if total_count > 0 else 0
        print(
            f"\n📊 Validation Results: {valid_count}/{total_count} states valid ({success_rate:.1%})"
        )

        return success_rate == 1.0


def main():
    """Example usage of Pkl configuration integration."""
    factory = PklStateFactory()

    print("🔍 Validating Pkl configuration...")
    if factory.validate_configuration():
        print("✅ All configurations are valid")
    else:
        print("❌ Some configurations have errors")
        return

    print("\n📋 Loading states for different spaces...")

    # Load states for each space
    for space in StateSpace:
        states = factory.load_states_for_space(space)
        print(f"  {space.value}: {len(states)} states")
        for state in states:
            print(f"    - {state.name} ({state.state_type.value})")

    print("\n🏗️  Creating State objects from Pkl...")
    all_states = factory.load_all_states()
    print(f"Created {len(all_states)} State objects:")

    for name, state in all_states.items():
        print(f"  - {name}: {state.state_type.value} ({state.success_mode.value})")

    print("\n✅ Integration complete!")


if __name__ == "__main__":
    main()
