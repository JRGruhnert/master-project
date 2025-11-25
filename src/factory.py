from src.agents.agent import Agent, AgentConfig
from src.agents.human import HumanAgent, HumanAgentConfig
from src.agents.ppo.baseline import BaselineAgent, BaselineAgentConfig
from src.agents.ppo.gnn import GNNAgent, GNNAgentConfig
from src.agents.search_tree import SearchTreeAgent, SearchTreeAgentConfig
from src.environments.environment import Environment, EnvironmentConfig
from src.modules.buffer import Buffer
from src.modules.evaluators.dense import DenseEvaluator, DenseEvaluatorConfig
from src.modules.evaluators.dense2 import Dense2Evaluator, Dense2EvaluatorConfig
from src.modules.evaluators.evaluator import Evaluator, EvaluatorConfig
from src.modules.evaluators.sparse import SparseEvaluator, SparseEvaluatorConfig
from src.modules.storage import Storage
from src.experiments.experiment import Experiment, ExperimentConfig
from src.experiments.pepr import PePrExperiment, PePrConfig
from src.experiments.skill_check import SkillCheckExperiment, SkillCheckExperimentConfig
from src.environments.calvin import (
    CalvinEnvironment,
    CalvinEnvironmentConfig,
    CalvinEnvironmentConfig,
)


def select_experiment(
    config: ExperimentConfig,
    env: Environment,
    storage: Storage,
) -> Experiment:
    """Create experiment from config - simple factory function"""

    if isinstance(config, PePrConfig):
        return PePrExperiment(config, env, storage)
    elif isinstance(config, SkillCheckExperimentConfig):
        return SkillCheckExperiment(config, env, storage)
    else:
        raise ValueError(f"Unknown experiment type: {type(config)}")


def select_agent(
    config: AgentConfig,
    storage_module: Storage,
    buffer_module: Buffer,
) -> Agent:
    """Create agent from config - simple factory function"""

    if isinstance(config, BaselineAgentConfig):
        return BaselineAgent(config, storage_module, buffer_module)
    elif isinstance(config, GNNAgentConfig):
        return GNNAgent(config, storage_module, buffer_module)
    elif isinstance(config, SearchTreeAgentConfig):
        return SearchTreeAgent(config, storage_module, buffer_module)
    elif isinstance(config, HumanAgentConfig):
        return HumanAgent(config, storage_module, buffer_module)
    else:
        raise ValueError(f"Unknown agent type: {type(config)}")


def select_evaluator(
    config: EvaluatorConfig,
    storage: Storage,
) -> Evaluator:
    """Create reward module from config - simple factory function"""
    if isinstance(config, DenseEvaluatorConfig):
        return DenseEvaluator(config, storage)
    elif isinstance(config, Dense2EvaluatorConfig):
        return Dense2Evaluator(config, storage)
    elif isinstance(config, SparseEvaluatorConfig):
        return SparseEvaluator(config, storage)
    else:
        raise ValueError(f"Unknown evaluator type: {type(config)}")


def select_environment(
    config: EnvironmentConfig,
    evaluator: Evaluator,
    storage: Storage,
) -> Environment:
    """Create environment from config - simple factory function"""
    if isinstance(config, CalvinEnvironmentConfig):
        return CalvinEnvironment(
            config,
            evaluator,
            storage,
        )
    else:
        raise ValueError(f"Unknown environment type: {type(config)}")
