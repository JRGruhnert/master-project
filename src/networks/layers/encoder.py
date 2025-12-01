import torch.nn as nn


class StateEncoder(nn.Module):
    def __init__(self, input_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.fc(x)


class QuaternionEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, out_dim),
            nn.ReLU(),
        )

    def forward(self, q):
        return self.fc(q)


class PositionEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, out_dim),
            nn.ReLU(),
        )

    def forward(self, e):
        return self.fc(e)


class AreaPositionEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, out_dim),
            nn.ReLU(),
        )

    def forward(self, e):
        return self.fc(e)


class ScalarEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, out_dim),
            nn.ReLU(),
        )

    def forward(self, s):
        return self.fc(s)
