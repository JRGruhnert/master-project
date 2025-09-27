#!/usr/bin/env python3
"""
Pkl Configuration Generator
Automatically generates Python classes from Apple Pkl configuration files.
"""

import subprocess
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Union
from dataclasses import dataclass, field
from enum import Enum


class PklGenerator:
    """Generate Python classes from Pkl configuration files."""

    def __init__(self, config_dir: str = "configs", output_dir: str = "generated"):
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def install_pkl(self):
        """Install Pkl CLI if not already installed."""
        try:
            subprocess.run(["pkl", "--version"], check=True, capture_output=True)
            print("✅ Pkl CLI already installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("📦 Installing Pkl CLI...")
            # Install Pkl CLI (adjust for your system)
            install_cmd = [
                "curl",
                "-L",
                "-o",
                "/tmp/pkl",
                "https://github.com/apple/pkl/releases/latest/download/pkl-linux-amd64",
            ]
            subprocess.run(install_cmd, check=True)
            subprocess.run(["chmod", "+x", "/tmp/pkl"], check=True)
            subprocess.run(["sudo", "mv", "/tmp/pkl", "/usr/local/bin/"], check=True)
            print("✅ Pkl CLI installed")

    def evaluate_pkl_file(self, pkl_file: Path) -> Dict[str, Any]:
        """Evaluate a Pkl file and return the result as a dictionary."""
        try:
            result = subprocess.run(
                ["pkl", "eval", "-f", "json", str(pkl_file)],
                capture_output=True,
                text=True,
                check=True,
            )
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error evaluating {pkl_file}: {e.stderr}")
            return {}
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing JSON from {pkl_file}: {e}")
            return {}

    def generate_dataclass_from_config(self, name: str, config: Dict[str, Any]) -> str:
        """Generate a Python dataclass from a configuration dictionary."""
        class_name = self._to_class_name(name)

        # Extract fields and their types
        fields = []
        imports = set(
            [
                "from dataclasses import dataclass, field",
                "from typing import Dict, List, Optional, Union",
            ]
        )

        for key, value in config.items():
            field_type, field_default = self._infer_type_and_default(key, value)
            if field_type.startswith("List") or field_type.startswith("Dict"):
                imports.add("from typing import Dict, List, Optional, Union")

            if field_default is None:
                fields.append(f"    {key}: {field_type}")
            else:
                fields.append(f"    {key}: {field_type} = {field_default}")

        # Generate the class
        class_code = f"""
{chr(10).join(sorted(imports))}

@dataclass
class {class_name}:
    \"\"\"Auto-generated configuration class for {name}.\"\"\"
{chr(10).join(fields)}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> '{class_name}':
        \"\"\"Create instance from dictionary.\"\"\"
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        \"\"\"Convert instance to dictionary.\"\"\"
        from dataclasses import asdict
        return asdict(self)
"""
        return class_code

    def generate_enum_from_values(self, name: str, values: List[str]) -> str:
        """Generate a Python enum from a list of values."""
        enum_name = self._to_class_name(name)
        enum_values = [
            f'    {val.upper().replace(" ", "_")} = "{val}"' for val in values
        ]

        return f"""
from enum import Enum

class {enum_name}(Enum):
    \"\"\"Auto-generated enum for {name}.\"\"\"
{chr(10).join(enum_values)}
"""

    def _to_class_name(self, name: str) -> str:
        """Convert a name to a valid Python class name."""
        return "".join(word.capitalize() for word in name.split("_"))

    def _infer_type_and_default(self, key: str, value: Any) -> tuple[str, str]:
        """Infer Python type annotation and default value from a value."""
        if isinstance(value, bool):
            return "bool", str(value)
        elif isinstance(value, int):
            return "int", str(value)
        elif isinstance(value, float):
            return "float", str(value)
        elif isinstance(value, str):
            return "str", f'"{value}"'
        elif isinstance(value, list):
            if not value:
                return "List[Any]", "field(default_factory=list)"
            elif all(isinstance(x, (int, float)) for x in value):
                elem_type = (
                    "float" if any(isinstance(x, float) for x in value) else "int"
                )
                return f"List[{elem_type}]", f"field(default_factory=lambda: {value})"
            elif all(isinstance(x, str) for x in value):
                return "List[str]", f"field(default_factory=lambda: {value})"
            else:
                return "List[Any]", f"field(default_factory=lambda: {value})"
        elif isinstance(value, dict):
            return "Dict[str, Any]", "field(default_factory=dict)"
        elif value is None:
            return "Optional[Any]", "None"
        else:
            return "Any", f'"{str(value)}"'

    def generate_all_configs(self):
        """Generate Python classes for all Pkl files in the config directory."""
        print(f"🔄 Generating Python classes from Pkl configs in {self.config_dir}")

        for pkl_file in self.config_dir.glob("*.pkl"):
            if pkl_file.name.startswith("base_"):
                continue  # Skip base templates

            print(f"📄 Processing {pkl_file.name}")
            config_data = self.evaluate_pkl_file(pkl_file)

            if not config_data:
                continue

            # Generate classes for each top-level configuration
            for section_name, section_data in config_data.items():
                if isinstance(section_data, dict):
                    # Single configuration object
                    if self._is_config_object(section_data):
                        class_code = self.generate_dataclass_from_config(
                            section_name, section_data
                        )
                        output_file = self.output_dir / f"{section_name}_config.py"
                        output_file.write_text(class_code)
                        print(f"  ✅ Generated {output_file.name}")

                    # Multiple configuration objects
                    else:
                        for config_name, config_data in section_data.items():
                            if isinstance(config_data, dict) and self._is_config_object(
                                config_data
                            ):
                                class_code = self.generate_dataclass_from_config(
                                    config_name, config_data
                                )
                                output_file = (
                                    self.output_dir / f"{config_name}_config.py"
                                )
                                output_file.write_text(class_code)
                                print(f"  ✅ Generated {output_file.name}")

        # Generate __init__.py for easy imports
        self._generate_init_file()
        print(f"🎉 All configurations generated in {self.output_dir}")

    def _is_config_object(self, data: Dict[str, Any]) -> bool:
        """Check if a dictionary represents a configuration object."""
        return isinstance(data, dict) and len(data) > 1 and "name" in data

    def _generate_init_file(self):
        """Generate __init__.py to make imports easier."""
        init_content = '''"""
Auto-generated configuration classes from Pkl files.
"""

'''

        # Add imports for all generated classes
        for py_file in self.output_dir.glob("*_config.py"):
            module_name = py_file.stem
            class_name = self._to_class_name(module_name.replace("_config", ""))
            init_content += f"from .{module_name} import {class_name}Config\n"

        init_content += "\n__all__ = [\n"
        for py_file in self.output_dir.glob("*_config.py"):
            class_name = self._to_class_name(py_file.stem.replace("_config", ""))
            init_content += f'    "{class_name}Config",\n'
        init_content += "]\n"

        init_file = self.output_dir / "__init__.py"
        init_file.write_text(init_content)


def main():
    """Main function to generate configurations."""
    generator = PklGenerator()

    # Install Pkl CLI if needed
    generator.install_pkl()

    # Generate all configurations
    generator.generate_all_configs()


if __name__ == "__main__":
    main()
