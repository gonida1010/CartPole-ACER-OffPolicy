import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import numpy as np
import random
from collections import deque

# Hyperparameters
n_train_processes = 3
learning_rate = 0.0002
update_interval = 5
gamma = 0.98
max_train_steps = 60000
PRINT_INTERVAL = update_interval * 100
replay_capacity = 2000
replay_ratio = 4
retrace_clip = 1.0
truncation_clip = 10.0
entropy_coef = 0.01
trust_region_delta = 1.0
avg_model_decay = 0.99


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_q = nn.Linear(256, 2)

    def _shared(self, x):
        return F.relu(self.fc1(x))

    def pi(self, x, softmax_dim=1):
        features = self._shared(x)
        prob = F.softmax(self.fc_pi(features), dim=softmax_dim)
        return prob

    def q(self, x):
        features = self._shared(x)
        return self.fc_q(features)

    def v(self, x, softmax_dim=1):
        pi = self.pi(x, softmax_dim=softmax_dim)
        q = self.q(x)
        return (pi * q).sum(dim=softmax_dim, keepdim=True)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def put(self, trajectory):
        self.buffer.append(trajectory)

    def sample(self):
        return random.choice(self.buffer)

    def __len__(self):
        return len(self.buffer)


def soft_update_average_model(model, average_model, decay):
    for average_param, param in zip(average_model.parameters(), model.parameters()):
        average_param.data.mul_(decay).add_(param.data * (1.0 - decay))


def apply_trust_region_update(model, average_model, states, policy_loss):
    policy_params = list(model.fc1.parameters()) + list(model.fc_pi.parameters())
    policy_grads = torch.autograd.grad(policy_loss, policy_params, allow_unused=True)

    with torch.no_grad():
        average_policy = average_model.pi(states, softmax_dim=1)

    current_policy = model.pi(states, softmax_dim=1)
    kl = (average_policy * (torch.log(average_policy + 1e-8) - torch.log(current_policy + 1e-8))).sum(dim=1).mean()
    kl_grads = torch.autograd.grad(kl, policy_params, allow_unused=True)

    grad_dot_kl = torch.zeros(1)
    kl_norm_sq = torch.zeros(1)

    for grad, kl_grad in zip(policy_grads, kl_grads):
        if grad is None or kl_grad is None:
            continue
        grad_dot_kl = grad_dot_kl + (grad * kl_grad).sum()
        kl_norm_sq = kl_norm_sq + (kl_grad * kl_grad).sum()

    trust_factor = torch.clamp((grad_dot_kl - trust_region_delta) / (kl_norm_sq + 1e-8), min=0.0)

    for param, grad, kl_grad in zip(policy_params, policy_grads, kl_grads):
        if grad is None:
            continue

        projected_grad = grad
        if kl_grad is not None:
            projected_grad = grad - trust_factor * kl_grad

        if param.grad is None:
            param.grad = projected_grad.clone()
        else:
            param.grad.add_(projected_grad)


def worker(worker_id, master_end, worker_end):
    # worker 프로세스에서는 master_end를 사용하지 않음
    # 불필요한 파이프 끝을 정리
    master_end.close()
    env = gym.make("CartPole-v1")

    # 최초 시드 설정
    obs, _ = env.reset(seed=worker_id)

    while True:
        # 메인 프로세스로부터 명령을 전달
        # step, reset, close
        cmd, data = worker_end.recv()

        if cmd == "step":
            obs, reward, terminated, truncated, info = env.step(int(data))
            done = terminated or truncated

            if done:
                obs, _ = env.reset()

            # step 결과를 메인 프로세스로 전달
            worker_end.send((obs, reward, done, info))

        elif cmd == "reset":
            obs, _ = env.reset()
            worker_end.send(obs)

        elif cmd == "close":
            env.close()
            worker_end.close()
            break

        elif cmd == "get_spaces":
            worker_end.send((env.observation_space, env.action_space))

        else:
            raise NotImplementedError(f"Unknown command: {cmd}")

# 여러 worker 환경을 한꺼번에 다루기 위한 클래스
class ParallelEnv:
    def __init__(self, n_train_processes):
        # 환경 개수를 저장
        self.nenvs = n_train_processes
        # 현재 step 결과를 기다리는 중인지 여부
        self.waiting = False
        # 이미 종료되었는지 여부
        self.closed = False
        # 생성한 worker 프로세스들을 저장할 리스트
        self.workers = []

        # worker의 수만큼 파이프를 생성
        # 각 파이프는 메인 프로세스 쪽 끝(master_end)과 worker 쪽 끝(worker_end)로 나눔
        master_ends, worker_ends = zip(*[mp.Pipe() for _ in range(self.nenvs)])
        self.master_ends = master_ends
        self.worker_ends = worker_ends

        for worker_id, (master_end, worker_end) in enumerate(zip(master_ends, worker_ends)):
            p = mp.Process(target=worker, args=(worker_id, master_end, worker_end))
            # 데몬 프로세스로 설정
            # 부모 프로세스가 종료되면 함께 종료하도록 설정
            p.daemon = True
            p.start()
            self.workers.append(p)

        # master는 worker_end를 사용하지 않음
        for worker_end in worker_ends:
            worker_end.close()

    def step_async(self, actions):
        # 여러 환경에 행동을 먼저 보내는 함수
        for master_end, action in zip(self.master_ends, actions):
            # 각 worker에게 하나씩 행동을 보냄
            # step 명령과 행동값을 worker에게 전달
            master_end.send(("step", int(action)))
            # 결과를 기다리는 상태로 설정
        self.waiting = True

    # step_async의 결과를 실제로 받아오는 함수
    def step_wait(self):
        # 모든 worker가 보낸 step 결과를 받아옴
        results = [master_end.recv() for master_end in self.master_ends]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        # 여러 worker의 결과를 하나의 배열로 묶어서 반환
        return (
            np.stack(obs).astype(np.float32),
            np.array(rews, dtype=np.float32),
            np.array(dones, dtype=np.bool_),
            infos,
        )
    # 모든 환경을 한꺼번에 리셋하는 함수
    def reset(self):
        for master_end in self.master_ends:
            master_end.send(("reset", None))
        return np.stack([master_end.recv() for master_end in self.master_ends]).astype(np.float32)

    # step_async()와 step_wait()를 한 번에 실행
    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    # worker와 환경을 종료
    def close(self):
        if self.closed:
            return

        if self.waiting:
            _ = [master_end.recv() for master_end in self.master_ends]

        for master_end in self.master_ends:
            master_end.send(("close", None))

        for worker in self.workers:
            worker.join()

        self.closed = True


def test(step_idx, model):
    env = gym.make("CartPole-v1")
    score = 0.0
    num_test = 10

    for _ in range(num_test):
        s, _ = env.reset()
        done = False

        while not done:
            with torch.no_grad():
                prob = model.pi(torch.from_numpy(s).float(), softmax_dim=0)
            a = torch.argmax(prob).item()

            s_prime, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated

            s = s_prime
            score += r

    print(f"Step # : {step_idx}, avg score : {score / num_test:.1f}")
    env.close()

def build_trajectories(states, actions, rewards, masks, behavior_policies, final_states):
    trajectories = []

    for env_idx in range(final_states.shape[0]):
        trajectories.append(
            {
                "states": np.array([step_state[env_idx] for step_state in states], dtype=np.float32),
                "actions": np.array([step_action[env_idx] for step_action in actions], dtype=np.int64),
                "rewards": np.array([step_reward[env_idx] for step_reward in rewards], dtype=np.float32),
                "masks": np.array([step_mask[env_idx] for step_mask in masks], dtype=np.float32),
                "behavior_policies": np.array(
                    [step_policy[env_idx] for step_policy in behavior_policies],
                    dtype=np.float32,
                ),
                "final_state": final_states[env_idx].astype(np.float32),
            }
        )

    return trajectories


def train_acer(model, average_model, optimizer, trajectory):
    states = torch.from_numpy(trajectory["states"]).float()
    actions = torch.from_numpy(trajectory["actions"]).long().unsqueeze(1)
    rewards = torch.from_numpy(trajectory["rewards"]).float()
    masks = torch.from_numpy(trajectory["masks"]).float()
    behavior_policies = torch.from_numpy(trajectory["behavior_policies"]).float()
    final_state = torch.from_numpy(trajectory["final_state"]).float().unsqueeze(0)

    pi = model.pi(states, softmax_dim=1)
    q = model.q(states)
    v = (pi * q).sum(dim=1)
    q_a = q.gather(1, actions).squeeze(1)
    pi_a = pi.gather(1, actions).squeeze(1)
    mu_a = behavior_policies.gather(1, actions).squeeze(1)

    with torch.no_grad():
        q_retrace = model.v(final_state, softmax_dim=1).squeeze(0).squeeze(0)
        retrace_targets = torch.zeros_like(rewards)

        detached_pi_a = pi_a.detach()
        detached_q_a = q_a.detach()
        detached_v = v.detach()

        for t in reversed(range(rewards.size(0))):
            q_retrace = rewards[t] + gamma * q_retrace * masks[t]
            retrace_targets[t] = q_retrace

            rho_t = detached_pi_a[t] / (mu_a[t] + 1e-8)
            q_retrace = torch.clamp(rho_t, max=retrace_clip) * (q_retrace - detached_q_a[t]) + detached_v[t]

    rho = pi_a / (mu_a + 1e-8)
    truncated_rho = torch.clamp(rho, max=truncation_clip)
    advantage = retrace_targets - v

    actor_loss = -(truncated_rho.detach() * torch.log(pi_a + 1e-8) * advantage.detach()).mean()

    rho_all = pi.detach() / (behavior_policies + 1e-8)
    correction_coeff = torch.clamp(1.0 - truncation_clip / (rho_all + 1e-8), min=0.0)
    correction_advantage = (q.detach() - v.detach().unsqueeze(1))
    bias_correction = -(
        correction_coeff * pi * torch.log(pi + 1e-8) * correction_advantage
    ).sum(dim=1).mean()

    critic_loss = F.smooth_l1_loss(q_a, retrace_targets)
    entropy = -(pi * torch.log(pi + 1e-8)).sum(dim=1).mean()
    policy_loss = actor_loss + bias_correction - entropy_coef * entropy

    optimizer.zero_grad()
    critic_loss.backward(retain_graph=True)
    apply_trust_region_update(model, average_model, states, policy_loss)
    optimizer.step()
    soft_update_average_model(model, average_model, avg_model_decay)


def main():
    envs = ParallelEnv(n_train_processes)

    model = ActorCritic()
    average_model = ActorCritic()
    average_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(replay_capacity)

    step_idx = 0
    s = envs.reset()

    while step_idx < max_train_steps:
        s_lst, a_lst, r_lst, mask_lst, mu_lst = [], [], [], [], []

        for _ in range(update_interval):
            with torch.no_grad():
                prob = model.pi(torch.from_numpy(s).float(), softmax_dim=1)

            a = Categorical(prob).sample().numpy()
            s_prime, r, done, info = envs.step(a)

            s_lst.append(s.copy())
            a_lst.append(a.copy())
            r_lst.append(r / 100.0)
            mask_lst.append(1.0 - done.astype(np.float32))
            mu_lst.append(prob.cpu().numpy())

            s = s_prime
            step_idx += n_train_processes

        trajectories = build_trajectories(s_lst, a_lst, r_lst, mask_lst, mu_lst, s)

        for trajectory in trajectories:
            replay_buffer.put(trajectory)
            train_acer(model, average_model, optimizer, trajectory)

        if len(replay_buffer) >= n_train_processes:
            for _ in range(replay_ratio):
                train_acer(model, average_model, optimizer, replay_buffer.sample())

        if step_idx % PRINT_INTERVAL == 0:
            test(step_idx, model)

    envs.close()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()