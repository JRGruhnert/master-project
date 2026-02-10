import glob
import re


from src.plotting.run import RunData, RunDataCollection
import src.plotting.plots.single_time as single_time
import src.plotting.plots.all_sr as all_sr
import src.plotting.plots.all_time as all_time
import src.plotting.plots.domain_sr as domain_sr
import src.plotting.plots.p_sr as p_sr
import src.plotting.plots.retrain_sr as retrain_sr
import src.plotting.plots.network_stats as network_stats


def make_plots(collection: RunDataCollection):
    print("Making plots...")
    single_time.plot(collection)  # done
    all_sr.plot(collection)  # done
    all_time.plot(collection)  # done
    domain_sr.plot(collection)  # (cant say yet)
    p_sr.plot(collection)  # done
    retrain_sr.plot(collection)  # done
    network_stats.plot()  # done
    network_stats.plot_ratio()  # done


def entry_point():
    networks = ["gnn", "baseline", "tree"]
    training_tags = [
        "t_slider_slider",
        "t_red_red",
        "t_pink_pink",
        "t_blue_blue",
        "t_srpb_sr",
        "t_srpb_srp",
        "t_srpb_srpb",
    ]
    retraining_tags = [
        "r_srpb_srp",
        "r_srpb_srpb",
    ]
    eval_tags = [
        "e_slider_slider",
        "e_red_red",
        "e_pink_pink",
        "e_blue_blue",
    ]
    tags = training_tags + retraining_tags + eval_tags

    file_pattern = re.compile(
        rf"(?P<tag>{'|'.join(tags)})_pe(?P<pe>[0-9.]+)_pr(?P<pr>[0-9.]+)"
    )
    tag_pattern = re.compile(
        rf"(?P<ident>{'|'.join(['t', 'r', 'e'])})_(?P<origin>\w+)_(?P<dest>\w+)?"
    )
    # ...existing code...
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
                    print(f"    No tag match for: {file_match.group('tag')}")  # Debug
            else:
                print(f"    No file match")  # Debug
    # ...existing code...

    for run in collection.runs:
        print(run.name)
    print(f"Total runs collected: {len(collection.runs)}")
    make_plots(collection)
