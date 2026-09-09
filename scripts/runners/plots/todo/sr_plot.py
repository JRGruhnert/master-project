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
class TSuccessRatePlotConfig(HecaPlotterConfig):
    name: str = "comparison_all_sr"


class TSuccessRatePlot(HecaPlotter):
    def __init__(self, config: TSuccessRatePlotConfig):
        super().__init__(config)
        self.config = config
        self.sr: dict[str, dict[str, list[float]]] = {
            "gnn": {MODE_EVAL: [], MODE_TRAIN: []},
            "baseline": {MODE_EVAL: [], MODE_TRAIN: []},
            "tree": {MODE_EVAL: [], MODE_TRAIN: []},
        }

    def __call__(self, runs: list[TrainingRunData]):
        for set in LIST_DOMAIN:
            for nt in [NT_GNN, NT_MLP, NT_TREE]:
                if nt == NT_TREE:
                    try:
                        run = self.get(
                            nt=nt,
                            mode=MODE_EVAL,
                            origin=set,
                            dest=set,
                            pe=0.0,
                            pr=0.0,
                            runs=runs,
                        )
                    except ValueError:
                        run = None
                    if run is not None:
                        self.sr[nt][MODE_EVAL].append(run.stats["run_stats"]["max_sr"])
                    else:
                        self.sr[nt][MODE_EVAL].append(0.0)
                else:
                    if set in ["sr", "srp", "srpb"]:
                        run_t = self.get(
                            nt=nt,
                            mode=MODE_TRAIN,
                            origin="srpb",
                            dest=set,
                            pe=0.0,
                            pr=0.0,
                            runs=runs,
                        )
                        run_e = self.get(
                            nt=nt,
                            mode=MODE_EVAL,
                            origin="srpb",
                            dest=set,
                            pe=0.0,
                            pr=0.0,
                            runs=runs,
                        )

                    else:
                        run_t = self.get(
                            nt=nt,
                            mode=MODE_TRAIN,
                            origin=set,
                            dest=set,
                            pe=0.0,
                            pr=0.0,
                            runs=runs,
                        )
                        run_e = self.get(
                            nt=nt,
                            mode=MODE_EVAL,
                            origin=set,
                            dest=set,
                            pe=0.0,
                            pr=0.0,
                            runs=runs,
                        )
                    self.sr[nt][MODE_TRAIN].append(run_t.stats["run_stats"]["max_sr"])
                    self.sr[nt][MODE_EVAL].append(run_e.stats["run_stats"]["max_sr"])

        x = np.arange(len(LIST_DOMAIN))
        width = 0.1

        fig, ax = plt.subplots(figsize=FIG_SIZE_FLAT)

        ax.bar(
            x - 3 * width,
            self.sr[NT_MLP][MODE_TRAIN],
            width,
            label=MAP_LABEL[NT_MLP],
            color=MAP_COLOR[NT_MLP]["main"],
        )
        ax.bar(
            x - 2 * width,
            self.sr[NT_MLP][MODE_EVAL],
            width,
            edgecolor="black",
            linewidth=1.0,
            hatch=HATCH_PATTERN,
            label=MAP_LABEL[NT_MLP],
            color=MAP_COLOR[NT_MLP]["secondary"],
        )
        ax.bar(
            x - width / 2,
            self.sr[NT_GNN][MODE_TRAIN],
            width,
            label=MAP_LABEL[NT_GNN],
            color=MAP_COLOR[NT_GNN]["main"],
        )
        ax.bar(
            x + width / 2,
            self.sr[NT_GNN][MODE_EVAL],
            width,
            edgecolor="black",
            linewidth=1.0,
            hatch=HATCH_PATTERN,
            label=MAP_LABEL[NT_GNN],
            color=MAP_COLOR[NT_GNN]["secondary"],
        )
        ax.bar(
            x + 2 * width,
            self.sr[NT_TREE][MODE_EVAL],
            width,
            edgecolor="black",
            linewidth=1.0,
            hatch=HATCH_PATTERN,
            label=MAP_LABEL[NT_TREE],
            color=MAP_COLOR[NT_TREE]["main"],
        )

        ax.set_xlabel(LABEL_SKILLSET)
        ax.set_ylabel(LABEL_SR)
        ax.set_xticks(x)
        set_y_ticks(ax)
        ax.set_xticklabels(LIST_DOMAIN)
        ax.set_title("Comparison of Max Success Rates across Skill Sets")
        ax.legend(handles=LEGEND_WITH_TREE)
        self.save()
