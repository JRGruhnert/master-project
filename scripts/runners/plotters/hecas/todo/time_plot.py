import matplotlib.pyplot as plt
import numpy as np
from heca.misc.run_data import TrainingRunData
from heca.plots.helper.helper import *
from dataclasses import dataclass

from heca.misc.run_data import TrainingRunData
from heca.runners.plotters.hecas.heca import (
    HecaPlotter,
    HecaPlotterConfig,
)


@dataclass
class TTimePlotConfig(HecaPlotterConfig):
    name: str = "comparison_all_time"


class TTimePlot(HecaPlotter):
    def __init__(self, config: TTimePlotConfig):
        super().__init__(config)
        self.config = config
        self.sr: dict[str, list[float]] = {"gnn": [], "baseline": []}

    def __call__(self, runs: list[TrainingRunData]):
        for set in LIST_DOMAIN_SMALL:
            for nt in ["gnn", "baseline"]:
                if set in ["sr", "srp", "srpb"]:
                    run = self.get(
                        nt=nt,
                        mode=MODE_TRAIN,
                        origin="srpb",
                        dest=set,
                        pe=0.0,
                        pr=0.0,
                        runs=runs,
                    )
                else:
                    run = self.get(
                        nt=nt,
                        mode=MODE_TRAIN,
                        origin=set,
                        dest=set,
                        pe=0.0,
                        pr=0.0,
                        runs=runs,
                    )
                self.sr[nt].append(run.stats["run_stats"]["sr_until_80"])

        x = np.arange(len(LIST_DOMAIN_SMALL))
        width = 0.20

        fig, ax = plt.subplots(figsize=FIG_SIZE_HIGH)

        ax.bar(
            x - width / 2,
            self.sr[NT_MLP],
            width,
            label=MAP_LABEL[NT_MLP],
            color=MAP_COLOR[NT_MLP]["main"],
        )
        ax.bar(
            x + width / 2,
            self.sr[NT_GNN],
            width,
            label=MAP_LABEL[NT_GNN],
            color=MAP_COLOR[NT_GNN]["main"],
        )

        ax.set_xlabel(LABEL_SKILLSET)
        ax.set_ylabel(LABEL_EPOCH)
        # set_y_ticks(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(LIST_DOMAIN_SMALL)
        ax.legend(handles=LEGEND_WITHOUT_TREE_AND_EVAL)
        ax.set_title("Comparison of time until 80% Success Rate")
        self.save()
