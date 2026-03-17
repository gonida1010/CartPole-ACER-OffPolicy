import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque

# --- 하이퍼파라미터 설정 ---
gamma = 0.98            # 할인율
lr = 0.001              # 학습률 (안정성을 위해 낮춤)
c_value = 1.0           # Truncation 상수
buffer_limit = 10000    # 버퍼 크기
batch_size = 32         # 배치 크기

# --- 1. Replay Buffer ---
class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst, prob_lst = [], [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask, prob = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])
            prob_lst.append([prob])

        return torch.tensor(np.array(s_lst), dtype=torch.float), \
               torch.tensor(np.array(a_lst)), \
               torch.tensor(np.array(r_lst)), \
               torch.tensor(np.array(s_prime_lst), dtype=torch.float), \
               torch.tensor(np.array(done_mask_lst)), \
               torch.tensor(np.array(prob_lst))

    def size(self):
        return len(self.buffer)

# --- 2. Actor-Critic Network ---
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_q = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return x

    def pi(self, x, softmax_dim=1):
        x = self.forward(x)
        prob = F.softmax(self.fc_pi(x), dim=softmax_dim)
        return prob

    def q(self, x):
        x = self.forward(x)
        q_value = self.fc_q(x)
        return q_value

# --- 3. ACER Update Logic ---
def train(model, optimizer, memory):
    s, a, r, s_prime, done_mask, prob_a = memory.sample(batch_size)

    pi = model.pi(s)
    pi_prime = model.pi(s_prime)
    q = model.q(s)
    q_prime = model.q(s_prime)

    pi_a = pi.gather(1, a)
    
    v = (pi * q).sum(dim=1, keepdim=True)
    v_prime = (pi_prime * q_prime).sum(dim=1, keepdim=True)

    rho = pi_a / prob_a
    c = torch.tensor(c_value, dtype=torch.float)
    rho_bar = torch.min(c, rho)

    q_target = r + gamma * v_prime * done_mask
    
    q_a = q.gather(1, a)
    critic_loss = F.smooth_l1_loss(q_a, q_target.detach())

    advantage = q_target.detach() - v.detach()
    actor_loss = - (rho_bar.detach() * torch.log(pi_a + 1e-8) * advantage).mean()

    # 엔트로피 보너스 (탐험 장려)
    entropy = - (pi * torch.log(pi + 1e-8)).sum(dim=1).mean()
    loss = critic_loss + actor_loss - 0.01 * entropy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# --- 4. Main Training Loop ---
def main():
    env = gym.make('CartPole-v1')
    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    memory = ReplayBuffer()

    print_interval = 100
    score = 0.0

    print("학습을 시작합니다...")
    for n_epi in range(2000):
        s, _ = env.reset()
        done = False
        
        while not done:
            prob = model.pi(torch.from_numpy(s).float().unsqueeze(0))
            prob = prob.view(-1)
            m = torch.distributions.Categorical(prob)
            a = m.sample().item()
            
            mu_prob = prob[a].item()
            
            s_prime, r, terminated, truncated, info = env.step(a) 
            done = terminated or truncated
            done_mask = 0.0 if done else 1.0
            
            # 보상 스케일링 조절
            memory.put((s, a, r/10.0, s_prime, done_mask, mu_prob))
            
            s = s_prime
            score += 1
            
            # 매 스텝 업데이트로 안정성 확보
            if memory.size() > 500:
                train(model, optimizer, memory)
                
            if done:
                break

        if n_epi % print_interval == 0 and n_epi != 0:
            avg_score = score / print_interval
            print(f"에피소드: {n_epi}, 평균 점수: {avg_score:.1f}")
            score = 0.0
            
            # 평균 점수가 450점을 넘으면 학습 조기 종료
            if avg_score > 450.0:
                print("🎉 목표 점수 달성! 학습을 조기 종료합니다.")
                break

    env.close()

    # --- 5. Test & Render (시각화) ---
    print("학습된 에이전트의 플레이를 확인합니다...")
    # render_mode='human'을 통해 실제 플레이 화면을 띄웁니다.
    test_env = gym.make('CartPole-v1', render_mode='human')
    s, _ = test_env.reset()
    done = False
    test_score = 0

    while not done:
        # 테스트 시에는 확률이 가장 높은 행동 선택 (Greedy)
        prob = model.pi(torch.from_numpy(s).float().unsqueeze(0))
        a = torch.argmax(prob).item()
        
        s_prime, r, terminated, truncated, info = test_env.step(a)
        done = terminated or truncated
        
        s = s_prime
        test_score += 1

    print(f"테스트 에피소드 최종 점수: {test_score}")
    test_env.close()

if __name__ == '__main__':
    main()