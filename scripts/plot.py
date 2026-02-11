import glob
import re
from itertools import product


from src.plotting.run import RunData, RunDataCollection
import src.plotting.plots.dual_time as dual_time
import src.plotting.plots.all_sr as all_sr
import src.plotting.plots.all_time as all_time
import src.plotting.plots.domain_sr as domain_sr
import src.plotting.plots.p_sr as p_sr
import src.plotting.plots.retrain_sr as retrain_sr
import src.plotting.plots.reward_sr as reward_sr
import src.plotting.plots.network_stats as network_stats


def make_plots(collection: RunDataCollection):
    print("Making plots...")
    # dual_time.plot(collection)  # done
    # all_sr.plot(collection)  # done
    # all_time.plot(collection)  # done
    # domain_sr.plot(collection)  # done
    # p_sr.plot(collection)  # done
    # retrain_sr.plot(collection)  # done
    # reward_sr.plot(collection)  # done
    network_stats.plot()  # done
    network_stats.plot_ratio()  # done


def entry_point():
    networks = ["gnn", "baseline", "tree"]
    special_idents = ["d", "re", "s"]
    special_tags = [
        "d_slider_slider",
        "d_red_red",
        "d_pink_pink",
        "d_blue_blue",
        "re_srpb_srp",
        "re_srpb_srpb",
        "s_srpb_sr",
    ]
    # Default tags and patterns for parsing filenames
    idents = ["t", "r", "e"]
    origins = ["slider", "red", "pink", "blue", "srpb"]
    dests = ["slider", "red", "pink", "blue", "sr", "srp", "srpb"]

    # Generate all combinations of tags
    tags = [
        f"{ident}_{origin}_{dest}"
        for ident, origin, dest in product(idents, origins, dests)
    ]

    tags += special_tags
    idents += special_idents

    file_pattern = re.compile(
        rf"(?P<tag>{'|'.join(tags)})_pe(?P<pe>[0-9.]+)_pr(?P<pr>[0-9.]+)"
    )
    tag_pattern = re.compile(
        rf"(?P<ident>{'|'.join(idents )})_(?P<origin>\w+)_(?P<dest>\w+)?"
    )

    collection = RunDataCollection()
    for nt in networks:
        read_path = f"results/{nt}/"
        all_results = glob.glob(f"{read_path}/*", recursive=True)
        print(f"Found {len(all_results)} results for network: {nt}")
        for path in all_results:
            file_match = file_pattern.search(path)
            if file_match:
                tag_match = tag_pattern.search(file_match.group("tag"))
                if tag_match:
                    metadata = {
                        "nt": nt,
                        "mode": tag_match.group("ident"),
                        "pe": float(file_match.group("pe")),
                        "pr": float(file_match.group("pr")),
                        "origin": tag_match.group("origin"),
                        "dest": tag_match.group("dest"),
                    }
                    # Collect data for further analysis
                    collection.add(RunData(path, metadata))
                else:
                    pass  # print(f"    No tag match for: {file_match.group('tag')}")  # Debug
            else:
                pass  # print(f"    No file match")  # Debug

    print(f"Total runs collected: {len(collection.runs)}")
    make_plots(collection)
