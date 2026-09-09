from dataclasses import dataclass

from scripts.runners.plots.training_plots.training_plot import (
    TrainingPlot,
    TrainingPlotConfig,
)


@dataclass
class ATrainingPlotConfig(TrainingPlotConfig):
    pass


class ATrainingPlot(TrainingPlot):
    def __init__(self, config: ATrainingPlotConfig):
        super().__init__(config)
        self.config = config

    def plot_content(self):
        for run in self.runs:
            print(run.metadata)
