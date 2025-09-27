"""
Configuration loader and manager for the master project.
Uses auto-generated classes from Pkl configurations.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Type, TypeVar, Optional
from dataclasses import asdict

# Import generated configurations (will be available after running generate_configs.py)
try:
    from generated import *
except ImportError:
    print(
        "⚠️  Generated configurations not found. Run 'python scripts/generate_configs.py' first."
    )


T = TypeVar("T")


class ConfigManager:
    """Manages configuration loading and validation."""

    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self._configs = {}

    def load_pkl_config(self, config_name: str, config_class: Type[T]) -> T:
        """Load a configuration from Pkl file and instantiate the class."""
        pkl_file = self.config_dir / f"{config_name}.pkl"

        if not pkl_file.exists():
            raise FileNotFoundError(f"Configuration file {pkl_file} not found")

        # Use Pkl CLI to evaluate the configuration
        import subprocess

        result = subprocess.run(
            ["pkl", "eval", "-f", "json", str(pkl_file)],
            capture_output=True,
            text=True,
            check=True,
        )

        config_data = json.loads(result.stdout)
        return self._instantiate_config(config_data, config_class)

    def _instantiate_config(self, data: Dict[str, Any], config_class: Type[T]) -> T:
        """Instantiate a configuration class from dictionary data."""
        if hasattr(config_class, "from_dict"):
            return config_class.from_dict(data)
        else:
            # Fallback to direct instantiation
            return config_class(**data)

    def save_config(self, config_obj: Any, output_path: str, format: str = "json"):
        """Save a configuration object to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(output_file, "w") as f:
                json.dump(asdict(config_obj), f, indent=2)
        elif format == "pickle":
            with open(output_file, "wb") as f:
                pickle.dump(config_obj, f)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def load_config_from_json(self, json_path: str, config_class: Type[T]) -> T:
        """Load configuration from JSON file."""
        with open(json_path, "r") as f:
            data = json.load(f)
        return self._instantiate_config(data, config_class)


# Example usage functions
def load_state_configs() -> Dict[str, Any]:
    """Load all state configurations."""
    manager = ConfigManager()
    try:
        return manager.load_pkl_config("states", dict)
    except Exception as e:
        print(f"Error loading state configs: {e}")
        return {}


def load_network_config(network_name: str):
    """Load a specific network configuration."""
    manager = ConfigManager()
    try:
        configs = manager.load_pkl_config("networks", dict)
        return configs.get("networks", {}).get(network_name)
    except Exception as e:
        print(f"Error loading network config for {network_name}: {e}")
        return None


def create_training_config(**overrides):
    """Create a training configuration with optional overrides."""
    manager = ConfigManager()
    try:
        base_config = manager.load_pkl_config("networks", dict)
        training_config = base_config.get("training", {})
        training_config.update(overrides)
        return training_config
    except Exception as e:
        print(f"Error creating training config: {e}")
        return None


# CLI interface for configuration management
def main():
    """CLI interface for configuration management."""
    import argparse

    parser = argparse.ArgumentParser(description="Configuration Manager")
    parser.add_argument(
        "action", choices=["generate", "validate", "convert"], help="Action to perform"
    )
    parser.add_argument("--config", help="Configuration file name")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument(
        "--format", choices=["json", "pickle"], default="json", help="Output format"
    )

    args = parser.parse_args()

    if args.action == "generate":
        from scripts.generate_configs import PklGenerator

        generator = PklGenerator()
        generator.generate_all_configs()

    elif args.action == "validate":
        # Validate configuration files
        manager = ConfigManager()
        config_files = list(Path("configs").glob("*.pkl"))
        for config_file in config_files:
            try:
                # Basic validation by attempting to evaluate
                import subprocess

                subprocess.run(
                    ["pkl", "eval", str(config_file)], check=True, capture_output=True
                )
                print(f"✅ {config_file.name} is valid")
            except subprocess.CalledProcessError as e:
                print(f"❌ {config_file.name} has errors: {e.stderr.decode()}")

    elif args.action == "convert":
        if not args.config or not args.output:
            print("❌ --config and --output are required for conversion")
            return

        manager = ConfigManager()
        try:
            # Load and convert configuration
            config_data = manager.load_pkl_config(args.config, dict)
            manager.save_config(config_data, args.output, args.format)
            print(f"✅ Converted {args.config}.pkl to {args.output}")
        except Exception as e:
            print(f"❌ Error converting configuration: {e}")


if __name__ == "__main__":
    main()
