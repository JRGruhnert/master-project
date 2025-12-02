from src.modules.evaluators.dense2 import Dense2EvaluatorConfig
from src.modules.evaluators.sparse import SparseEvaluatorConfig


dense_evaluator = Dense2EvaluatorConfig(
    success_reward=1.0,
    positive_step_reward=0.05,
    negative_step_reward=-0.01,
)

sparse_evaluator = SparseEvaluatorConfig(
    success_reward=1.0,
    step_reward=-0.01,
)
