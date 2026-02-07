import glob
import re

from src.plotting.run import RunData, RunDataCollection
import src.plotting.plots.single_time as single_time
import src.plotting.plots.all_sr as all_sr
import src.plotting.plots.all_time as all_time
import src.plotting.plots.domain_sr as domain_sr
import src.plotting.plots.p_sr as p_sr
import src.plotting.plots.retrain_sr as retrain_sr


def make_plots(collection: RunDataCollection):
    single_time.plot(collection)
    all_sr.plot(collection)
    all_time.plot(collection)
    domain_sr.plot(collection)
    p_sr.plot(collection)
    retrain_sr.plot(collection)


def entry_point():
    networks = ["gnn", "baseline", "tree"]
    training_tags = [
        "t_b_b",
        "t_sr_sr",
        "t_sp_sp",
        "t_sb_sb",
        "t_brpb_br",
        "t_brpb_brp",
        "t_brpb_brpb",
    ]
    retraining_tags = [
        "r_brpb_brp",
        "r_brpb_brpb",
    ]
    eval_tags = [
        "e_b_b",
        "e_sr_sr",
        "e_sp_sp",
        "e_sb_sb",
    ]
    tags = training_tags + retraining_tags + eval_tags
    collection = RunDataCollection()
    for nt in networks:
        read_path = f"results/{nt}/"
        all_results = glob.glob(f"{read_path}/*", recursive=True)

        file_pattern = re.compile(
            rf"(?P<tag>{'|'.join(tags)})_pe(?P<pe>[0-9.]+)_pr(?P<pr>[0-9.]+)"
        )
        tag_pattern = re.compile(
            rf"(?P<ident>{'|'.join(['t', 'r', 'e'])})?(?P<origin>\d)(?P<dest>\d)?"
        )

        for path in all_results:
            file_match = file_pattern.search(path)
            if file_match:
                tag_match = tag_pattern.search(file_match.group("tag"))
                if tag_match:
                    metadata = {
                        "nt": nt,
                        "mode": file_match.group("tag"),
                        "pe": float(file_match.group("pe")),
                        "pr": float(file_match.group("pr")),
                        "origin": tag_match.group("origin"),
                        "dest": tag_match.group("dest"),
                    }
                    # Collect data for further analysis
                    collection.add(RunData(path, metadata))

    make_plots(collection)
