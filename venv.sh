export HRL_ROOT=/home/jangruhnert/Documents/GitHub/master-project
export CALVIN_ENV_ROOT=$HRL_ROOT/dependencies/calvin_env_modified
export TACORL_ROOT=$HRL_ROOT/dependencies/tacorl
export TAPAS_ROOT=$HRL_ROOT/dependencies/tapas
export PYTHONPATH=$HRL_ROOT:$CALVIN_ENV_ROOT:$TACORL_ROOT:$TAPAS_ROOT:$PYTHONPATH

echo "Paths set!"
echo "HRL_ROOT: $HRL_ROOT"
echo "PYTHONPATH: $PYTHONPATH"