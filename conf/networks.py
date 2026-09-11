from heca.heca_gnn.network import Network

default = Network.Config()
interact = Network.Config(use_option_interaction=True)
timeline = Network.Config(use_timeline_memory=True)
both = Network.Config(
    use_timeline_memory=True,
    use_option_interaction=True,
)

NETWORK_NAMES = [
    "default",
    "interact",
    "timeline",
    "both",
]
