from abc import abstractmethod
from dataclasses import dataclass, field
from glob import glob
import re

from heca.misc.run_data import TrainingRunData
from heca.runners.plotters.hecas.heca import (
    HecaPlotter,
    HecaPlotterConfig,
)


@dataclass
class TrainingPlotConfig(HecaPlotterConfig):
    networks: list[str] = field(default_factory=lambda: ["gnn", "baseline"])
    file_pattern: re.Pattern = re.compile(
        r"(?P<tag>\w+_\w+_\w+)_pe(?P<pe>[0-9.]+)_pr(?P<pr>[0-9.]+)"
    )
    tag_pattern: re.Pattern = re.compile(
        r"(?P<ident>\w+)_(?P<origin>\w+)_(?P<dest>\w+)"
    )


class TrainingPlot(HecaPlotter):
    def __init__(self, config: TrainingPlotConfig):
        super().__init__(config)
        self.config = config
        self.runs: list[TrainingRunData] = []
        self.load_results()

    @abstractmethod
    def plot_content(self):
        raise NotImplementedError()

    def get(
        self,
        nt: str,
        mode: str,
        origin: str,
        dest: str,
        pe: float,
        pr: float,
        runs: list[TrainingRunData],
    ) -> TrainingRunData:
        for run in runs:
            if (
                run.metadata.get("nt") == nt
                and run.metadata.get("mode") == mode
                and run.metadata.get("origin") == origin
                and run.metadata.get("dest") == dest
                and run.metadata.get("pe") == pe
                and run.metadata.get("pr") == pr
            ):
                return run
        else:
            raise ValueError(
                f"No run found for network={nt}, mode={mode}, origin={origin}, dest={dest}, pe={pe}, pr={pr}"
            )

    def load_results(self):
        for nt in self.config.networks:
            self.load_network_results(nt)

    def load_network_results(self, nt: str):
        for path in glob(f"results/{nt}//*", recursive=True):
            file_match = self.config.file_pattern.get(path)
            if file_match:
                tag_match = self.config.tag_pattern.get(file_match.group("tag"))
                if tag_match:
                    metadata = {
                        "nt": nt,
                        "mode": tag_match.group("ident"),
                        "pe": float(file_match.group("pe")),
                        "pr": float(file_match.group("pr")),
                        "origin": tag_match.group("origin"),
                        "dest": tag_match.group("dest"),
                    }
                    self.runs.append(TrainingRunData(path, metadata))
