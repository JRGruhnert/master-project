#!/bin/bash
# filepath: /home/jangruhnert/Documents/GitHub/master-project/install.sh

export HRL_ROOT=$(pwd)
mkdir -p "$HRL_ROOT/dependencies/"

echo "Installing HRL Master Project dependencies..."

# Install calvin_env_modified
export CALVIN_ENV_ROOT="$HRL_ROOT/dependencies/calvin_env_modified"
if [ ! -d "$CALVIN_ENV_ROOT" ] ; then
    echo "Cloning calvin_env_modified..."
    git clone --recursive https://github.com/JRGruhnert/calvin_env_modified.git $CALVIN_ENV_ROOT
    cd "$CALVIN_ENV_ROOT"
else
    echo "Updating calvin_env_modified..."
    cd "$CALVIN_ENV_ROOT"
    git pull --recurse-submodules
fi
pip install .


# Install riepybdlib
export RIEPYBDLIB_ROOT="$HRL_ROOT/dependencies/riepybdlib"
if [ ! -d "$RIEPYBDLIB_ROOT" ] ; then
    echo "Cloning riepybdlib..."
    git clone https://github.com/vonHartz/riepybdlib.git $RIEPYBDLIB_ROOT
    cd "$RIEPYBDLIB_ROOT"
else
    echo "Updating riepybdlib..."
    cd "$RIEPYBDLIB_ROOT"
    git pull --recurse-submodules
fi
pip install .

# Install TAPAS (with separation branch)
export TAPAS_ROOT="$HRL_ROOT/dependencies/tapas"
if [ ! -d "$TAPAS_ROOT" ] ; then
    echo "Cloning tapas..."
    git clone https://github.com/JRGruhnert/TapasCalvin.git $TAPAS_ROOT
    cd "$TAPAS_ROOT"
    git checkout seperation
    #pip install .  # Install after cloning
else
    echo "Updating tapas..."
    cd "$TAPAS_ROOT"
    git pull --recurse-submodules
fi
pip install .

# Install this project (HRL)
echo "Installing HRL project..."
cd $HRL_ROOT
pip install .

echo "Installation complete!"