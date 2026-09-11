import torch
from torch import nn

from heca.data.entity import Entity
from heca.data.free import FreeEntity
from heca.data.prismatic import PrismaticEntity
from heca.data.revolute import RevoluteEntity
from heca.data.static import StaticEntity


class _Block(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _EntityEncoder(nn.Module):
    BLOCKS: tuple[str, ...] = ()

    def __init__(self, out_dim: int):
        super().__init__()
        if "state" not in self.BLOCKS:
            raise ValueError("every entity has a state block")

        # One slice of the embedding per block, sized by the block's share.
        share = max(out_dim // len(self.BLOCKS), 8)
        dims = {name: share for name in self.BLOCKS}
        dims[self.BLOCKS[-1]] = out_dim - share * (len(self.BLOCKS) - 1)

        self.dims = dims
        self.subs = nn.ModuleDict(
            {
                name: (
                    nn.Linear(Entity.LAYOUT[name].dim, dims[name], bias=False)
                    if name == "state"
                    else _Block(Entity.LAYOUT[name].dim, dims[name])
                )
                for name in self.BLOCKS
            }
        )

    def _slices(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        inv = 1.0 / abs(Entity.BASE_LOGSTD)
        for name in self.BLOCKS:
            block = Entity.LAYOUT[name]
            parts = [x[:, block.mean()]]
            if block.logstd_dim:
                parts.append(x[:, block.logstd()] * inv)
            out[name] = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = self._slices(x)
        return torch.cat([self.subs[name](parts[name]) for name in self.BLOCKS], dim=-1)


class FreeEncoder(_EntityEncoder):
    """Free body: position and orientation."""

    BLOCKS = FreeEntity.BLOCKS


class StaticEncoder(_EntityEncoder):
    """Static prop: position only."""

    BLOCKS = StaticEntity.BLOCKS


class PrismaticEncoder(_EntityEncoder):
    """Prismatic joint: position, orientation and the slide offset."""

    BLOCKS = PrismaticEntity.BLOCKS


class RevoluteEncoder(_EntityEncoder):
    """Revolute joint: position, orientation and the joint angle."""

    BLOCKS = RevoluteEntity.BLOCKS


class OptionEncoder(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(Entity.FEATURE_DIM),
            nn.Linear(Entity.FEATURE_DIM, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)
