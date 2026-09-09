import argparse
import os
import signal
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib

from heca.graphs.graph import SubgoalMode
from heca.heca_gnn.network import Network

matplotlib.use("Agg")

from heca.agents.heca import Heca
from heca.experts.expert import ExpertModel
from heca.learning.fppo import FPPO
from heca.learning.ppo import PPO
from heca.learning.server import FLServer
from heca.misc import logger
from heca.misc.interrupt import request_stop, stop_requested

from scripts.common.args import add_heca_arguments, generate_tag
from scripts.common.scenes import agents_by_scene

import conf.networks

GRACE_SECONDS = 10.0


def generate_clients(
    tag: str,
    network: Network.Config,
    clients: dict[str, list[ExpertModel.Config]],
    smode: SubgoalMode,
    fit_rotation: bool,
    inference: bool,
    federated: bool,
    use_wandb: bool,
    virtual: bool,
    reload: bool,
    use_gt: bool,
    n_batch: int,
):
    wandb = logger.WandBConfig(enabled=use_wandb)
    hecas = []
    server = None
    if federated:
        server_cfg = FLServer.Config(tag=tag, network=network)
        server = FLServer.get(server_cfg)
        for scene, agents in clients.items():
            heca = Heca.Config(
                agents=agents,
                learner=FPPO.Config(
                    tag=f"{scene}_{tag}",
                    network=network,
                    server=server_cfg,
                    wandb=wandb,
                    max_update=n_batch,
                    lr_annealing=True,
                ),
                visualize=False,
                inference=inference,
                virtual=virtual,
                reload=reload,
                use_gt=use_gt,
                fit_rotation=fit_rotation,
                smode=smode,
            )
            hecas.append(heca)
    else:
        for scene, agents in clients.items():
            heca = Heca.Config(
                agents=agents,
                learner=PPO.Config(
                    tag=f"{scene}_{tag}",
                    network=network,
                    wandb=wandb,
                    max_update=n_batch,
                    lr_annealing=True,
                ),
                visualize=False,
                inference=inference,
                virtual=virtual,
                reload=reload,
                use_gt=use_gt,
                fit_rotation=fit_rotation,
                smode=smode,
            )
            hecas.append(heca)
    return hecas, server


def train(
    client_cfgs: list[Heca.Config],
    server: FLServer | None,
    n_batch: int = 1000,
    grace: float = GRACE_SECONDS,
):
    clients = [Heca.get(client) for client in client_cfgs]

    def run(agent: Heca):
        n = 0
        while n < n_batch and not stop_requested():
            if agent.tick():
                n += 1
                agent.learner.sync()
        return n

    def request_shutdown():
        request_stop()
        if server is not None:
            server.stop()

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

    pool = ThreadPoolExecutor(max_workers=len(clients))
    futures = [pool.submit(run, a) for a in clients]
    try:
        for future in as_completed(futures):
            future.result()
    except KeyboardInterrupt:
        request_shutdown()
        pool.shutdown(wait=False, cancel_futures=True)
        if not wait_for_workers(futures, grace):
            logger.warning(f"Workers still busy after {grace:.0f}s; forcing exit.")
            os._exit(130)
        if server is not None and server.version > 0:
            server.save()
        raise SystemExit(130)
    except Exception:
        request_shutdown()
        pool.shutdown(wait=False, cancel_futures=True)
        if not wait_for_workers(futures, grace):
            traceback.print_exc()
            logger.error("Workers did not stop in time; forcing exit.")
            os._exit(1)
        raise
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_heca_arguments(parser)
    args = parser.parse_args()

    def _handle_stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handle_stop)

    network = getattr(conf.networks, args.network)

    clients: dict[str, list] = {}
    for scene_cfg, model_cfgs in agents_by_scene():
        if args.scene and scene_cfg.tag != args.scene:
            continue
        clients.update({scene_cfg.tag: model_cfgs})

    exp, server = generate_clients(
        generate_tag(args),
        network,
        clients,
        inference=args.inference,
        federated=args.federated,
        virtual=args.virtual,
        use_wandb=args.wandb,
        reload=args.reload,
        use_gt=args.gt,
        fit_rotation=args.rotation,
        smode=args.smode,
        n_batch=args.batch,
    )
    train(exp, server, args.batch)


if __name__ == "__main__":
    main()
