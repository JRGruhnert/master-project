import argparse
import os
import signal
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib

from heca.experts.expert import ExpertModel
from heca.scenes.scene import Scene

matplotlib.use("Agg")  # headless plotting

import matplotlib.pyplot as plt

from heca.experts.tapas import TapasExpert
from heca.misc import logger
from heca.misc.interrupt import request_stop, stop_requested

from scripts.b03_plot_tapas_models import evaluate_one
from scripts.common.args import (
    add_model_argument,
    add_scene_argument,
    add_use_gt_argument,
)
from scripts.common.plot_lock import PLOT_LOCK
from scripts.common.scenes import agents_by_scene

GRACE_SECONDS = 10.0


def fit_tapas(expert: TapasExpert):
    demos = expert.load_demos()
    expert.fit_stage1(demos)
    with PLOT_LOCK:
        save_plots(expert, "fit_stage1")  # velocity-segmentation debug figures
        expert.plot_stage1()
        save_plots(expert, "stage1")

    expert.fit_stage2(demos)
    with PLOT_LOCK:
        save_plots(expert, "fit_stage2")
        expert.plot_stage2()
        save_plots(expert, "stage2")

    expert.save()


def save_plots(agent: TapasExpert, stage: str):
    out_dir = TapasExpert.save_dir(agent.cfg) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    for num in plt.get_fignums():
        fig = plt.figure(num)
        path = out_dir / f"{stage}_{num}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {path}")
    plt.close("all")


def pipeline_scene(
    scene_cfg: Scene.Config,
    models: list[ExpertModel.Config],
    gt: bool,
    episodes: int,
    max_tries: int,
):
    """Run the full fit/evaluate pipeline for every model of one scene."""
    for cfg in models:
        if stop_requested():
            return
        logger.info(f"[{scene_cfg.tag}] === pipeline for {cfg.tag} ===")
        model = ExpertModel.get(cfg)
        model.use_gt(gt)
        assert isinstance(model, TapasExpert), "Only Tapas is supported atm."
        fit_tapas(model)
        model.fit_conditions()
        evaluate_one(cfg, scene_cfg, episodes, max_tries, gt)


def wait_for_workers(futures: list, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if all(f.done() for f in futures):
                return True
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_scene_argument(parser)
    add_model_argument(parser)
    add_use_gt_argument(parser)
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Evaluation episodes per agent (stage 3).",
    )
    parser.add_argument(
        "--max-tries",
        type=int,
        default=3,
        help="Max attempts per episode before giving up (stage 3).",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))

    jobs = []
    for scene_cfg, models in agents_by_scene():
        if args.scene and scene_cfg.tag != args.scene:
            continue
        models = [m for m in models if not (args.model and m.tag != args.model)]
        if models:
            jobs.append((scene_cfg, models))

    logger.info(f"Running pipeline for {len(jobs)} scenes in parallel threads.")
    pool = ThreadPoolExecutor(max_workers=len(jobs))
    futures = [
        pool.submit(
            pipeline_scene, scene_cfg, models, args.gt, args.episodes, args.max_tries
        )
        for scene_cfg, models in jobs
    ]

    def request_shutdown():
        request_stop()

    try:
        for future in as_completed(futures):
            future.result()
    except KeyboardInterrupt:
        request_shutdown()
        pool.shutdown(wait=False, cancel_futures=True)
        if not wait_for_workers(futures, GRACE_SECONDS):
            logger.warning(
                f"Workers still busy after {GRACE_SECONDS:.0f}s; forcing exit."
            )
            os._exit(130)
        raise SystemExit(130)
    except Exception:
        request_shutdown()
        pool.shutdown(wait=False, cancel_futures=True)
        if not wait_for_workers(futures, GRACE_SECONDS):
            traceback.print_exc()
            logger.error("Workers did not stop in time; forcing exit.")
            os._exit(1)
        raise
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
