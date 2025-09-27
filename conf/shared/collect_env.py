from tapas_gmm.master_project.environment import MasterEnvConfig
from conf.env.calvin.env_collect_conf import calvin_env_config


collect_env = MasterEnvConfig(
    calvin_config=calvin_env_config,
    debug_vis=True,
)
