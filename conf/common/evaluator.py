from src.modules.evaluators.dense2 import Dense2EvaluatorConfig
from src.modules.evaluators.sparse import SparseEvaluatorConfig


dense_evaluator = Dense2EvaluatorConfig(
    success_reward=100.0,
    positive_step_reward=5.0,
    negative_step_reward=-1.0,
)

sparse_evaluator = SparseEvaluatorConfig(
    success_reward=100.0,
    step_reward=-1.0,
)
