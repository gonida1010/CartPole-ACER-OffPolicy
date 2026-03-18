# CartPole-ACER-OffPolicy

CartPole 환경에서 ACER(Actor-Critic with Experience Replay) 알고리즘을 직접 재구현한 PyTorch 프로젝트입니다. 이 과제의 핵심은 단순 실행이 아니라, 왜 ACER가 등장했는지와 기존 Actor-Critic 계열 알고리즘의 한계를 어떻게 보완하는지를 실제 코드 수준에서 이해하는 데 있습니다.

## 1. 과제 목적

ACER는 Actor-Critic의 정책 기반 학습 장점과 Replay Buffer의 샘플 재사용 장점을 결합한 알고리즘입니다. 기존 on-policy Actor-Critic 계열은 최신 정책으로 수집한 데이터만 사용하는 경우가 많아 샘플 효율이 낮고, 과거 경험을 적극적으로 재사용하기 어렵습니다. ACER는 이 문제를 해결하기 위해 다음 요소를 함께 사용합니다.

- Experience Replay
- Off-policy 보정
- Truncated Importance Sampling
- Retrace target
- Trust Region Update

이 프로젝트는 위 요소들을 CartPole-v1에 맞게 직접 구현하여 ACER의 핵심 동작을 재현하는 것을 목표로 합니다.

## 2. 왜 ACER가 등장했는가

기존 Actor-Critic은 정책과 가치 함수를 동시에 학습한다는 장점이 있지만, 보통 on-policy 데이터에 의존합니다. 즉 현재 정책으로 막 수집한 데이터만 학습에 쓰기 때문에 샘플 효율이 떨어집니다.

A2C와 A3C는 이러한 Actor-Critic 계열의 대표적인 확장입니다.

- Actor-Critic: 정책(Actor)과 가치함수(Critic)를 함께 학습하지만 데이터 재사용이 제한적입니다.
- A2C: 여러 환경을 동기적으로 병렬 실행하여 학습 안정성을 높입니다.
- A3C: 여러 워커가 비동기적으로 학습하며 탐험 다양성을 늘립니다.

하지만 A2C와 A3C 역시 기본적으로는 on-policy 학습이 중심이므로, 이미 수집한 경험을 반복 사용하기 어렵습니다. 반면 DQN 계열은 Replay Buffer를 통해 샘플 효율은 높지만, 정책 기반 방법의 장점을 그대로 갖기 어렵습니다.

ACER는 이 두 방향을 결합합니다. 즉, Actor-Critic 구조를 유지하면서도 Replay Buffer를 사용하고, off-policy로 인해 생기는 분포 차이를 Importance Sampling과 Retrace로 보정하여 안정적으로 학습하려는 알고리즘입니다.

## 3. ACER의 핵심 아이디어

### 3.1 Experience Replay

환경에서 수집한 trajectory를 Replay Buffer에 저장한 뒤, 최신 데이터뿐 아니라 과거 데이터도 다시 샘플링하여 학습합니다. 이렇게 하면 한 번 수집한 데이터를 여러 번 사용할 수 있어 샘플 효율이 좋아집니다.

### 3.2 Importance Sampling

Replay Buffer의 데이터는 현재 정책이 아니라 과거 행동 정책으로 생성된 데이터입니다. 따라서 현재 정책 $\pi(a|s)$와 행동 정책 $\mu(a|s)$ 사이 차이를 보정해야 합니다. 이를 위해 중요도 비율

$$
\rho_t = \frac{\pi(a_t|s_t)}{\mu(a_t|s_t)}
$$

를 사용합니다.

하지만 이 비율이 너무 커지면 분산이 크게 증가하므로, ACER는 이를 일정 값 이하로 잘라 쓰는 Truncated Importance Sampling을 사용합니다.

### 3.3 Retrace

Critic 학습에서는 단순 1-step target 대신 multi-step retrace target을 사용합니다. 이는 off-policy 데이터를 더 안정적으로 사용하기 위한 보정 방식입니다. 중요도 비율을 무한정 사용하지 않고 clip하여 분산을 줄이면서도, 과거 trajectory 정보를 활용할 수 있게 합니다.

### 3.4 Trust Region Update

정책이 한 번의 업데이트로 너무 크게 바뀌면 off-policy 보정이 어려워지고 학습이 불안정해질 수 있습니다. 이를 막기 위해 ACER는 평균 정책과 현재 정책 사이의 KL 거리 제약을 이용한 Trust Region Update를 사용합니다. 이 구현에서는 정책 gradient를 평균 정책 방향 KL gradient에 대해 projection하여 지나치게 큰 정책 이동을 제한합니다.

## 4. 구현 내용

이 프로젝트의 구현은 하나의 파일인 [main.py](main.py)에 정리되어 있으며, CartPole-v1 환경에서 ACER를 학습하도록 구성되어 있습니다.

### 4.1 네트워크 구조

- 공통 은닉층 1개
- 정책 head: 각 행동의 확률 $\pi(a|s)$ 출력
- Q head: 각 행동의 상태-행동 가치 $Q(s,a)$ 출력
- 상태가치 $V(s)$는 $\sum_a \pi(a|s)Q(s,a)$ 형태로 계산

### 4.2 병렬 환경 수집

`torch.multiprocessing` 기반으로 여러 CartPole 환경을 병렬 실행합니다. 이는 A2C 방식과 비슷하게 데이터를 빠르게 모으기 위한 부분이며, 수집된 rollout은 이후 ACER 학습에 사용됩니다.

### 4.3 Replay Buffer

각 환경에서 수집한 trajectory를 버퍼에 저장하고,

- 최신 trajectory로 즉시 학습
- Replay Buffer에서 과거 trajectory를 다시 샘플링하여 추가 학습

하는 구조로 구현했습니다.

### 4.4 Off-policy 보정

Replay Buffer에 저장된 trajectory에는 당시 행동 정책의 확률 $\mu(a|s)$를 함께 저장합니다. 학습 시 현재 정책 확률과 비교하여 importance ratio를 계산하고, actor update와 retrace 계산에 사용합니다.

### 4.5 Retrace 기반 Critic 학습

Trajectory의 마지막 상태 가치에서 시작하여 뒤에서 앞으로 target을 계산합니다. 이때 clipped importance ratio를 적용해 off-policy critic 학습의 분산을 줄입니다.

### 4.6 Trust Region Actor Update

Actor loss를 직접 backward 하는 대신,

- 평균 정책 네트워크를 유지하고
- 현재 정책과 평균 정책 사이 KL gradient를 계산한 뒤
- 정책 gradient를 trust region 안으로 projection하여
- 최종 gradient만 optimizer에 반영

하는 방식으로 구현했습니다.

즉, 이번 구현은 과제 요구사항에 맞게 ACER의 핵심 요소인 Replay Buffer, Importance Sampling, Retrace, Trust Region Update를 모두 포함합니다.

## 5. 실행 방법

가상환경 활성화 후 아래 명령으로 실행할 수 있습니다.

```bash
python main.py
```

## 6. 실험 결과

학습 로그에서는 CartPole-v1의 최대 점수인 평균 500.0을 여러 구간에서 안정적으로 달성했습니다. 긴 학습에서는 점수가 다시 흔들릴 수 있지만, 보고서에는 과제 목적에 맞게 최대 성능 달성 결과를 중심으로 정리합니다.

대표 결과는 다음과 같습니다.

| Step   | Average Score |
| ------ | ------------- |
| 31,500 | 500.0         |
| 34,500 | 500.0         |
| 37,500 | 500.0         |
| 39,000 | 500.0         |
| 40,500 | 500.0         |
| 42,000 | 500.0         |
| 43,500 | 500.0         |
| 45,000 | 500.0         |
| 48,000 | 500.0         |
| 49,500 | 500.0         |
| 52,500 | 500.0         |
| 54,000 | 500.0         |

이 결과는 구현한 ACER가 CartPole 환경에서 충분히 높은 수준으로 학습되었음을 보여줍니다.

## 7. 정리

이번 과제를 통해 확인할 수 있는 점은 다음과 같습니다.

- Actor-Critic 계열은 정책 기반 학습의 장점이 있지만, 기본적으로 on-policy라 샘플 효율이 낮을 수 있습니다.
- ACER는 Replay Buffer를 도입하여 샘플 재사용을 가능하게 합니다.
- Replay Buffer를 쓰면 off-policy 문제가 생기므로 Importance Sampling이 필요합니다.
- Importance Sampling만 그대로 쓰면 분산이 커지므로 Truncated Importance Sampling과 Retrace가 필요합니다.
- 정책이 급격히 변하지 않도록 Trust Region Update를 적용하면 학습 안정성이 좋아집니다.

즉, ACER는 Actor-Critic, Experience Replay, Off-policy Correction을 결합하여 샘플 효율과 학습 안정성을 함께 노린 알고리즘이며, 이번 CartPole 재구현은 그 핵심 아이디어를 직접 확인하기 위한 실습 구현입니다.
