# PPO.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import numpy as np

# -----------------------
# PPO NETWORK
# -----------------------
class PPOActorCritic(nn.Module):
    def __init__(self, state_dim=9, action_dim=3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        self.policy = nn.Linear(128, action_dim)
        self.value = nn.Linear(128, 1)

        # Orthogonal initialization for stability
        for m in self.shared:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.constant_(self.policy.bias, 0.0)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.constant_(self.value.bias, 0.0)

    def forward(self, x):
        x = self.shared(x)
        return self.policy(x), self.value(x)


# -----------------------
# PPO AGENT
# -----------------------
class PPOAgent:
    def __init__(
        self,
        state_dim=9,
        action_dim=3,
        actor_lr=3e-4,
        critic_lr=1e-3,
        gamma=0.99,
        clip_eps=0.2,
        update_epochs=8,
        batch_size=64,
        min_batch_size=256,
        entropy_coef=0.01,
        device=None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PPOActorCritic(state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.optimizer = optim.Adam([
            {"params": self.model.policy.parameters(), "lr": actor_lr},
            {"params": self.model.value.parameters(), "lr": critic_lr},
            {"params": self.model.shared.parameters(), "lr": actor_lr}
        ])

        self.gamma = gamma
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.min_batch_size = min_batch_size
        self.entropy_coef = entropy_coef
        self.last_action = None
        self.memory = {"states": [], "actions": [], "logprobs": [], "rewards": [], "dones": []}

    # ---------- action selection ----------
    def act(self, state):
        # state: numpy array or list (state_dim,)
        state_np = np.array(state, dtype=np.float32)
        state_t = torch.FloatTensor(state_np).unsqueeze(0).to(self.device)  # (1, S)
        logits, _ = self.model(state_t)  # (1, A), (1,1)
        logits = logits.squeeze(0).detach().cpu()  # (A,)

        # Determine unsafe actions based on correct indices:
        # state layout: [b1,b2,b3,b4, beam_dist, left_closeness, right_closeness, lane_offset, heading_error]
        beam_dist = float(state_np[4])        # 0..1 (1 far, 0 near)
        left_close = float(state_np[5])       # 0..1 (0 none, 1 very close)
        right_close = float(state_np[6])

        # Safety thresholds (tunable)
        FRONT_THRESHOLD = 0.25   # if beam_dist < FRONT_THRESHOLD => front blocked (near obstacle)
        SIDE_THRESHOLD = 0.25    # if left_close/right_close > SIDE_THRESHOLD => side too close

        safe_actions = [0, 1, 2]  # 0 straight,1 left,2 right

        # If front is blocked, straight might be unsafe
        if beam_dist < FRONT_THRESHOLD and 0 in safe_actions:
            safe_actions.remove(0)

        # If left side is too close, left turn is unsafe
        if left_close > SIDE_THRESHOLD and 1 in safe_actions:
            safe_actions.remove(1)

        # If right side is too close, right turn is unsafe
        if right_close > SIDE_THRESHOLD and 2 in safe_actions:
            safe_actions.remove(2)

        # If all actions unsafe (rare), fallback to all
        if len(safe_actions) == 0:
            safe_actions = [0, 1, 2]

        # Mask logits (set logits of disallowed actions to -inf)
        mask = torch.full_like(logits, float("-inf"))
        for a in safe_actions:
            mask[a] = logits[a]
        masked_logits = mask

        dist_pi = dist.Categorical(logits=masked_logits)
        action = dist_pi.sample()
        logprob = dist_pi.log_prob(action)

        return int(action.item()), float(logprob.item())

    def remember(self, s, a, lp, r, d):
        self.memory["states"].append(np.array(s, dtype=np.float32))
        self.memory["actions"].append(int(a))
        self.memory["logprobs"].append(float(lp))
        self.memory["rewards"].append(float(r))
        self.memory["dones"].append(bool(d))
        self.last_action = a

    # ---------- PPO update ----------
    def train(self, force=False, entropy_coef=None):
        if entropy_coef is None:
            entropy_coef = self.entropy_coef

        mem_size = len(self.memory["states"])
        if mem_size == 0:
            print("PPO.train(): memory empty -> nothing to do.")
            return 0.0

        if (not force) and mem_size < self.min_batch_size:
            print(f"PPO.train(): insufficient samples ({mem_size} < {self.min_batch_size}) -> skipping update.")
            return 0.0

        print(f"PPO.train(): Starting update on {mem_size} samples.")

        # Convert memory to tensors
        states = torch.FloatTensor(np.array(self.memory["states"], dtype=np.float32)).to(self.device)  # (N, S)
        actions = torch.LongTensor(self.memory["actions"]).to(self.device)                            # (N,)
        old_logprobs = torch.FloatTensor(self.memory["logprobs"]).to(self.device)                    # (N,)

        # compute returns (discounted)
        returns = []
        G = 0.0
        for r, d in zip(reversed(self.memory["rewards"]), reversed(self.memory["dones"])):
            if d:
                G = 0.0
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)  # (N,)
        # Normalize returns for stable training
        returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)

        dataset = torch.utils.data.TensorDataset(states, actions, old_logprobs, returns)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        final_loss = 0.0
        final_actor = None
        final_critic = None
        final_entropy = None

        for _epoch in range(self.update_epochs):
            for (s_batch, a_batch, old_lp_batch, ret_batch) in loader:
                logits, values = self.model(s_batch)  # logits: (B,A), values: (B,1)

                # distribution
                dist_pi = dist.Categorical(logits=logits)
                new_logprobs = dist_pi.log_prob(a_batch)  # (B,)
                entropy = dist_pi.entropy().mean()        # scalar

                # Compute advantages
                values = values.squeeze(1)                 # (B,)
                advantages = ret_batch - values            # (B,)
                critic_loss = 0.5 * (advantages.pow(2).mean())

                # Robust normalization of advantages
                adv_mean = advantages.mean()
                adv_std = advantages.std(unbiased=False)
                if torch.isnan(adv_std) or adv_std.item() < 1e-8:
                    advantages = advantages - adv_mean
                else:
                    advantages = (advantages - adv_mean) / (adv_std + 1e-8)

                ratio = torch.exp(new_logprobs - old_lp_batch)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                loss = actor_loss + critic_loss - entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                final_loss = float(loss.detach().cpu().item())
                final_actor = float(actor_loss.detach().cpu().item())
                final_critic = float(critic_loss.detach().cpu().item())
                final_entropy = float(entropy.detach().cpu().item())

        # Clear memory after update
        self.memory = {k: [] for k in self.memory}
        print("PPO.train(): Update finished. Final loss:", final_loss)
        self.save_model("ppo_checkpoint.pth")

        return final_loss, final_actor, final_critic, final_entropy

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)
        print(f"PPOAgent: Model saved successfully to {filepath}")

    def load_model(self, filepath):
        try:
            self.model.load_state_dict(torch.load(filepath, map_location=self.device))
            print(f"PPOAgent: Model loaded successfully from {filepath}")
            self.model.eval()
        except FileNotFoundError:
            print(f"PPOAgent: Error - Model file not found at {filepath}. Starting from scratch.")
        except RuntimeError as e:
            print(f"PPOAgent: Error loading model state dict. Error: {e}")
