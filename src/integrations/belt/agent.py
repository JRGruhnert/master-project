import torch
import torch.nn as nn


class BeLT(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=128, nhead=4, num_layers=4):
        super().__init__()
        self.state_action_dim = state_dim + action_dim

        # Input projection (state + action → embedding)
        self.input_proj = nn.Linear(self.state_action_dim, latent_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=nhead, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Behavior token projection (aggregate)
        self.to_latent = nn.Linear(latent_dim, latent_dim)

        # Transformer decoder (conditioned on latent token)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim, nhead=nhead, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projections
        self.action_pred = nn.Linear(latent_dim, action_dim)
        self.state_pred = nn.Linear(latent_dim, state_dim)

    def forward(self, states, actions):
        """
        states: (B, T, state_dim)
        actions: (B, T, action_dim)
        """
        x = torch.cat([states[:, :-1], actions[:, :-1]], dim=-1)  # (B, T-1, D)
        x = self.input_proj(x)
        enc = self.encoder(x)

        # Mean-pool to get behavior token
        z = enc.mean(dim=1)
        z = self.to_latent(z).unsqueeze(1)  # (B, 1, latent_dim)

        # Decode: condition decoder on z (repeated)
        z_repeated = z.repeat(1, enc.size(1), 1)
        dec = self.decoder(tgt=enc, memory=z_repeated)

        pred_actions = self.action_pred(dec)
        pred_states = self.state_pred(dec)
        return pred_states, pred_actions, z.squeeze(1)
