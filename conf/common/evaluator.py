from src.modules.evaluators.dense_best import DenseBestEvaluatorConfig
from src.modules.evaluators.sparse import SparseEvaluatorConfig


dense_evaluator = DenseBestEvaluatorConfig(
    success_reward=1.0,
    positive_step_reward=0.1,
    negative_step_reward=-0.01,
)

sparse_evaluator = SparseEvaluatorConfig(
    step_reward=-0.01,
    success_reward=1.0,
)
