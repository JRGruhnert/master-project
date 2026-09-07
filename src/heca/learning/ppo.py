from dataclasses import dataclass
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.distributions import Categorical

from heca.learning.learner import Learner
from heca.heca_gnn.network import Network
from heca.misc import hardware
from heca.misc.interrupt import stop_requested


def build_chunks(n: int, terminals: list[bool], seq_len: int) -> list[list[int]]:
    """Split the buffer (episode-ordered transitions) into contiguous chunks
    that never cross an episode boundary and are at most ``seq_len`` long
    (``seq_len <= 0`` means one chunk per whole episode). Each chunk is
    self-contained for truncated BPTT: its first step either starts an episode
    (no stored memory) or carries the stored ``mem_step`` to bootstrap from.
    """
    chunks: list[list[int]] = []
    seg: list[int] = []
    for i in range(n):
        seg.append(i)
        if terminals[i] or (seq_len > 0 and len(seg) == seq_len):
            chunks.append(seg)
            seg = []
    if seg:
        chunks.append(seg)
    return chunks


def score_chunks(
    net: Network,
    chunks: list[list[int]],
    data: list,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    logprobs: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    use_mem = net.cfg.use_timeline_memory
    for seg in chunks:
        h: torch.Tensor | None = None
        for pos, t in enumerate(seg):
            mem = h if (use_mem and pos > 0) else None
            logits, value = net.forward(data[t], memory=mem)
            dist = Categorical(logits=logits)
            logprobs.append(dist.log_prob(actions[t : t + 1]))
            values.append(value)
            entropies.append(dist.entropy())
            if use_mem and pos < len(seg) - 1:
                nxt = getattr(data[seg[pos + 1]], "mem_step", None)
                if nxt is not None:
                    u, _ = nxt
                    h = net.timeline(u.clone(), net._last_mem)
                else:
                    h = None
    return torch.cat(logprobs), torch.cat(values), torch.cat(entropies)


class PPO(Learner):
    @dataclass(kw_only=True)
    class Config(Learner.Config):
        # Hyperparameters
        batch_size: int = 64
        n_epoch: int = 4
        lr: float = 3e-4
        eps_clip: float = 0.2
        entropy_coef: float = 0.02
        critic_coef: float = 0.5
        max_grad_norm: float = 0.5
        target_kl: float | None = 0.01
        clip_value_loss: bool = True
        seq_len: int = 0  # Truncated-BPTT chunk

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    def learn(self):
        adv, rtn = self.buffer.compute_advantages()

        self._mini_batch_loop(adv, rtn)

        if self.cfg.lr_annealing:
            lr = self.cfg.lr * (1.0 - self.current_update / self.cfg.max_update)
            for pg in self.optim.param_groups:
                pg["lr"] = lr

    def _fedprox_term(self) -> torch.Tensor:
        return torch.tensor(0.0, device=hardware.device)

    def _mini_batch_loop(self, adv: torch.Tensor, rtn: torch.Tensor):
        old_data = self.buffer.data
        old_actions = self.buffer.actions.detach().squeeze(-1)
        old_logprobs = self.buffer.logprobs.detach().squeeze(-1)
        old_values = self.buffer.values.detach().squeeze(-1)
        N = len(old_data)

        use_chunked = self.network.cfg.use_timeline_memory
        if use_chunked:
            terminals = [d.terminal or d.truncated for d in old_data]
            chunks = build_chunks(N, terminals, self.cfg.seq_len)

        # Accumulators for averaging
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        total_loss = 0.0
        total_fedprox_loss = 0.0
        num_minibatches = 0

        kl_stop = False
        for _ in range(self.cfg.n_epoch):
            if stop_requested():
                # Abort between epochs so a Ctrl-C is not delayed by the whole
                # PPO update (capacity/batch_size * n_epoch minibatches).
                break
            if use_chunked:
                # Minibatches are contiguous episode chunks (shuffled at chunk
                # granularity), each scored by one truncated-BPTT unroll.
                order = torch.randperm(len(chunks)).tolist()
                minibatches: list[list[int]] = []
                cur: list[int] = []
                cur_n = 0
                for ci in order:
                    cur.append(ci)
                    cur_n += len(chunks[ci])
                    if cur_n >= self.cfg.batch_size:
                        minibatches.append(cur)
                        cur, cur_n = [], 0
                if cur:
                    minibatches.append(cur)
            else:
                indices = torch.randperm(N).tolist()
                minibatches = [
                    indices[s : s + self.cfg.batch_size]
                    for s in range(0, N, self.cfg.batch_size)
                ]

            for mb in minibatches:
                if use_chunked:
                    mb_chunks = [chunks[ci] for ci in mb]
                    flat = [i for seg in mb_chunks for i in seg]
                    mb_idx = torch.tensor(flat, dtype=torch.long)
                    logprobs, state_values, entropies = score_chunks(
                        self.network, mb_chunks, old_data, old_actions
                    )
                else:
                    mb_idx = torch.tensor(mb, dtype=torch.long)
                    mb_data = [old_data[i] for i in mb]
                    logprobs, state_values, entropies = self.network.evaluate(
                        mb_data, old_actions[mb_idx]
                    )
                entropy = entropies.mean()

                mb_logprobs = old_logprobs[mb_idx]
                mb_adv = adv[mb_idx]
                mb_rtn = rtn[mb_idx]
                mb_old_val = old_values[mb_idx]

                assert isinstance(entropy, torch.Tensor)
                # Normalize advantages
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # PPO ratio
                ratios = torch.exp(logprobs - mb_logprobs)

                # Policy loss
                surr1 = ratios * mb_adv
                surr2 = (
                    torch.clamp(ratios, 1 - self.cfg.eps_clip, 1 + self.cfg.eps_clip)
                    * mb_adv
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                if self.cfg.clip_value_loss:
                    values_pred = mb_old_val + torch.clamp(
                        state_values - mb_old_val,
                        -self.cfg.eps_clip,
                        self.cfg.eps_clip,
                    )
                    value_loss = self.mse_loss(values_pred, mb_rtn)
                else:
                    value_loss = self.mse_loss(state_values, mb_rtn)

                # Entropy bonus
                loss = (
                    policy_loss
                    + self.cfg.critic_coef * value_loss
                    - self.cfg.entropy_coef * entropy
                )

                fedprox = self._fedprox_term()
                loss = loss + fedprox

                # Gradient step
                self.optim.zero_grad()
                loss.mean().backward()
                clip_grad_norm_(self.network.parameters(), self.cfg.max_grad_norm)
                self.optim.step()

                # KL early stopping
                with torch.no_grad():
                    log_ratio = logprobs - mb_logprobs
                    approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio)
                    clip_fraction = (
                        ((ratios - 1).abs() > self.cfg.eps_clip).float().mean()
                    )
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.item()
                    total_approx_kl += approx_kl.item()
                    total_clip_fraction += clip_fraction.item()
                    total_loss += loss.mean().item()
                    total_fedprox_loss += fedprox.item()
                    num_minibatches += 1

                    if (
                        self.cfg.target_kl is not None
                        and approx_kl > self.cfg.target_kl
                    ):
                        kl_stop = True
                        break

            if kl_stop:
                break

        all_values = self.buffer.values.detach().squeeze(-1)
        var_returns = rtn.var()
        if var_returns > 0:
            explained_var = (1 - (rtn - all_values).var() / var_returns).item()
        else:
            explained_var = 0.0

        if num_minibatches == 0:
            # Aborted by stop_requested() before any minibatch ran; nothing to
            # report (avoid a ZeroDivisionError in the metrics below).
            return

        self.metrics.update(
            {
                "train/policy_loss": total_policy_loss / num_minibatches,
                "train/value_loss": total_value_loss / num_minibatches,
                "train/entropy": total_entropy / num_minibatches,
                "train/approx_kl": total_approx_kl / num_minibatches,
                "train/clip_frac": total_clip_fraction / num_minibatches,
                "train/total_loss": total_loss / num_minibatches,
                "train/fedprox_loss": total_fedprox_loss / num_minibatches,
                "train/expl_var": explained_var,
                "train/lr": self.optim.param_groups[0]["lr"],
            }
        )
