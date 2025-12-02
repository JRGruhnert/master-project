import wandb
from src.environments.calvin import CalvinEnvironmentConfig
from src.experiments.pepr import PePrConfig
from src.modules.buffer import BufferConfig
from src.modules.evaluators.dense2 import Dense2EvaluatorConfig
from src.modules.evaluators.sparse import SparseEvaluatorConfig
from src.modules.logger import LoggerConfig
from src.modules.storage import StorageConfig
from src.agents.ppo.baseline import BaselineAgentConfig
from src.agents.ppo.gnn import GNNAgentConfig

from scripts.train import TrainConfig, Trainer


def entry_point():
    run = wandb.init()
    # Use wandb.config to override TrainConfig
    if wandb.config["dense_evaluator"]:
        print("Using Dense2EvaluatorConfig from wandb config.")
        evaluator = Dense2EvaluatorConfig(
            success_reward=100.0,
            positive_step_reward=5.0,
            negative_step_reward=-1.0,
        )
    else:
        print("Using SparseEvaluatorConfig from wandb config.")
        evaluator = SparseEvaluatorConfig(
            success_reward=100.0,
            step_reward=-1.0,
        )

    if wandb.config["storage.network"] == "gnn":
        agent_type = GNNAgentConfig
    elif wandb.config["storage.network"] == "baseline":
        agent_type = BaselineAgentConfig
    else:
        raise ValueError(f"Unsupported agent type: {wandb.config['storage.network']}")

    config = TrainConfig(
        agent=agent_type(
            eval=wandb.config["agent.eval"],
            early_stop_patience=wandb.config["agent.early_stop_patience"],
            use_ema_for_early_stopping=wandb.config["agent.use_ema_for_early_stopping"],
            ema_smoothing_factor=wandb.config["agent.ema_smoothing_factor"],
            min_batches=wandb.config["agent.min_batches"],
            max_batches=wandb.config["agent.max_batches"],
            saving_freq=wandb.config["agent.saving_freq"],
            save_stats=wandb.config["agent.save_stats"],
            mini_batch_size=wandb.config["agent.mini_batch_size"],
            learning_epochs=wandb.config["agent.learning_epochs"],
            lr_annealing=wandb.config["agent.lr_annealing"],
            learning_rate=wandb.config["agent.learning_rate"],
            gamma=wandb.config["agent.gamma"],
            gae_lambda=wandb.config["agent.gae_lambda"],
            eps_clip=wandb.config["agent.eps_clip"],
            entropy_coef=wandb.config["agent.entropy_coef"],
            critic_coef=wandb.config["agent.critic_coef"],
            max_grad_norm=wandb.config["agent.max_grad_norm"],
            target_kl=wandb.config["agent.target_kl"],
            clip_value_loss=wandb.config["agent.clip_val_loss"],
        ),
        buffer=BufferConfig(
            steps=wandb.config["buffer.steps"],
        ),
        logger=LoggerConfig(),
        storage=StorageConfig(
            used_skills=wandb.config["storage.used_skills"],
            used_states=wandb.config["storage.used_states"],
            eval_states=wandb.config["storage.eval_states"],
            network=wandb.config["storage.network"],
        ),
        experiment=PePrConfig(
            p_empty=wandb.config["experiment.p_empty"],
            p_rand=wandb.config["experiment.p_rand"],
        ),
        environment=CalvinEnvironmentConfig(render=False),
        evaluator=evaluator,
    )

    trainer = Trainer(config, run)
    trainer.run()
    run.finish()


if __name__ == "__main__":
    entry_point()
