import argparse
from conf.networks import NETWORK_NAMES
from heca.graphs.graph import SubgoalMode


def add_smode_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--smode",
        type=SubgoalMode,
        choices=list(SubgoalMode),
        default=SubgoalMode.GOAL,
    )


def add_viewer_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Enable the passive viewer.",
    )


def add_scene_argument(parser: argparse.ArgumentParser, default=None):
    parser.add_argument(
        "--scene",
        default=default,
        help="Scene module tag (e.g. scene1, sceneog).",
    )


def add_tag_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--tag",
        required=True,
        help="Run tag for identification.",
    )


def add_model_argument(parser: argparse.ArgumentParser, default=None):
    parser.add_argument(
        "--model",
        default=default,
        help="Agent tag within the selected scene.",
    )


def add_virtual_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--virtual",
        action="store_true",
        help="Initialize agents in virtual mode.",
    )


def add_federated_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--federated",
        action="store_true",
        help="Federated (FPPO) or plain (PPO) training.",
    )


def add_use_gt_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--gt",
        action="store_true",
        help="Use ground-truth observations (default: true).",
    )


def add_fit_rotation_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--rotation",
        action="store_true",
        help="Fit the conditions with rotations.",
    )


def add_wandb_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable wandb logging. Disabled by default for multi-client runs.",
    )


def add_batch_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--batch",
        type=int,
        default=500,
        help="Number of training batches per client.",
    )


def add_reload_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Reload expert conditions instead of loading the condition cache.",
    )


def add_inference_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Sets the network in inference mode.",
    )


def add_network_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--network",
        choices=NETWORK_NAMES,
        default="default",
        help="Network config name from conf.networks.",
    )


def add_heca_arguments(parser: argparse.ArgumentParser):
    add_network_argument(parser)
    add_federated_argument(parser)
    add_wandb_argument(parser)
    add_use_gt_argument(parser)
    add_fit_rotation_argument(parser)
    add_batch_argument(parser)
    add_virtual_argument(parser)
    add_scene_argument(parser)
    add_tag_argument(parser)
    add_smode_argument(parser)
    add_inference_argument(parser)
    add_reload_argument(parser)


def subgoal_tag(smode: SubgoalMode) -> str:
    if smode == SubgoalMode.GOAL:
        return "g"
    elif smode == SubgoalMode.CHAIN:
        return "c"
    elif smode == SubgoalMode.BOTH:
        return "b"
    raise ValueError


def generate_tag(args: argparse.Namespace) -> str:
    final_tag = ""
    final_tag += args.tag
    final_tag += "-"
    final_tag += args.network
    final_tag += "-"
    final_tag += "f" if args.federated else "_"
    final_tag += "g" if args.gt else "_"
    final_tag += "v" if args.virtual else "_"
    final_tag += "r" if args.rotation else "_"
    final_tag += subgoal_tag(args.smode)
    return final_tag
