from src.modules.evaluators.dense import DenseEvaluatorConfig
from src.modules.evaluators.dense2 import Dense2EvaluatorConfig
from src.modules.evaluators.dense3 import Dense3EvaluatorConfig
from src.modules.evaluators.sparse import SparseEvaluatorConfig


dense1_evaluator = DenseEvaluatorConfig(
    success_reward=1.0,
    positive_step_reward=0.05,
    negative_step_reward=-0.01,
)

dense2_evaluator = Dense2EvaluatorConfig(
    success_reward=1.0,
    max_progress_reward=0.5,
    max_regress_penalty=0.25,
    step_penalty=-0.01,
)

dense3_evaluator = Dense3EvaluatorConfig(
    success_reward=25.0,
    max_progress_reward=1.0,
    step_penalty=-0.002,
    add_monotonic_reward=True,
)

sparse_evaluator = SparseEvaluatorConfig(
    success_reward=25.0,
    step_reward=-0.002,
)
