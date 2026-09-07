from heca.heca_gnn.network import Network

default = Network.Config()
small = Network.Config(feature_dim=128, encoder_depth=2, gnn_mlp_depth=2)
big = Network.Config(feature_dim=512, encoder_depth=4, gnn_mlp_depth=4, attn_heads=8)

NETWORK_NAMES = ["default", "small", "big"]
