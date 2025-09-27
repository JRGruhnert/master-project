#!/bin/bash

# Setup script for Pkl configuration system

set -e

echo "🚀 Setting up Pkl configuration system..."

# Create directories
mkdir -p configs generated scripts

# Check if Pkl is installed
if ! command -v pkl &> /dev/null; then
    echo "📦 Installing Pkl CLI..."
    
    # Detect OS and install accordingly
    OS=$(uname -s)
    ARCH=$(uname -m)
    
    case $OS in
        Linux)
            if [ "$ARCH" = "x86_64" ]; then
                PKL_URL="https://github.com/apple/pkl/releases/latest/download/pkl-linux-amd64"
            else
                echo "❌ Unsupported architecture: $ARCH"
                exit 1
            fi
            ;;
        Darwin)
            if [ "$ARCH" = "x86_64" ]; then
                PKL_URL="https://github.com/apple/pkl/releases/latest/download/pkl-macos-amd64"
            elif [ "$ARCH" = "arm64" ]; then
                PKL_URL="https://github.com/apple/pkl/releases/latest/download/pkl-macos-aarch64"
            else
                echo "❌ Unsupported architecture: $ARCH"
                exit 1
            fi
            ;;
        *)
            echo "❌ Unsupported OS: $OS"
            exit 1
            ;;
    esac
    
    # Download and install
    curl -L -o /tmp/pkl "$PKL_URL"
    chmod +x /tmp/pkl
    sudo mv /tmp/pkl /usr/local/bin/pkl
    
    echo "✅ Pkl CLI installed successfully"
else
    echo "✅ Pkl CLI already installed"
fi

# Verify installation
pkl --version

# Generate Python classes from Pkl configurations
echo "🔄 Generating Python classes from Pkl configurations..."
python scripts/generate_configs.py

# Create example usage script
cat > example_usage.py << 'EOF'
#!/usr/bin/env python3
"""
Example usage of Pkl-generated configurations.
"""

from master_project.config_manager import ConfigManager, load_state_configs, load_network_config
from pathlib import Path

def main():
    """Demonstrate configuration usage."""
    
    print("🔍 Loading configurations...")
    
    # Load state configurations
    states = load_state_configs()
    print(f"📋 Loaded {len(states.get('states', {}))} state configurations")
    
    # Load a specific network configuration
    gnn_config = load_network_config("gnn1")
    if gnn_config:
        print(f"🧠 Loaded GNN1 config: {gnn_config.get('architecture', 'Unknown')}")
    
    # Example: Create a custom configuration
    manager = ConfigManager()
    
    # You can now use the generated classes like:
    # from generated import StateConfig, NetworkConfig, TrainingConfig
    
    print("✅ Configuration system ready!")
    print("\n📚 Usage examples:")
    print("  - Edit .pkl files in the configs/ directory")
    print("  - Run 'python scripts/generate_configs.py' to regenerate classes")
    print("  - Import and use generated classes from the 'generated' module")
    print("  - Use ConfigManager to load and validate configurations")

if __name__ == "__main__":
    main()
EOF

chmod +x example_usage.py

echo ""
echo "🎉 Pkl configuration system setup complete!"
echo ""
echo "📁 Created files:"
echo "  - configs/*.pkl (configuration templates)"
echo "  - scripts/generate_configs.py (class generator)"
echo "  - master_project/config_manager.py (configuration manager)"
echo "  - example_usage.py (usage examples)"
echo ""
echo "🚀 Next steps:"
echo "  1. Edit the .pkl files in configs/ to match your needs"
echo "  2. Run 'python scripts/generate_configs.py' to generate Python classes"
echo "  3. Use the generated classes in your code"
echo "  4. Run 'python example_usage.py' to see it in action"
echo ""
echo "📖 Pkl documentation: https://pkl-lang.org/
