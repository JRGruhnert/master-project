from heca.heca_gnn.network import Network

default = Network.Config()
interact = Network.Config(use_option_interaction=True)
timeline = Network.Config(use_timeline_memory=True)
both = Network.Config(
    use_timeline_memory=True,
    use_option_interaction=True,
)
film = Network.Config(use_film_conditioning=True)
film_timeline = Network.Config(use_timeline_memory=True, use_film_conditioning=True)
film_all = Network.Config(
    use_timeline_memory=True,
    use_option_interaction=True,
    use_film_conditioning=True,
)

NETWORK_NAMES = [
    "default",
    "interact",
    "timeline",
    "both",
    "film",
    "film_timeline",
    "film_all",
]
