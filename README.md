# CartPole-ACER-OffPolicy
OpenAI Gym의 CartPole 환경에 적용한 ACER (Actor-Critic with Experience Replay) 알고리즘의 PyTorch 구현체입니다. 이 저장소는 교육 목적으로 작성되었으며, Actor-Critic 기법의 학습 안정성과 경험 재생(Experience Replay)의 샘플 효율성을 어떻게 결합하는지 보여줍니다. Off-policy 학습, 리플레이 버퍼, 그리고 학습을 안정화하기 위한 편향 보정 및 절삭된 중요도 샘플링(Truncated Importance Sampling, Retrace)의 핵심 개념을 직접 구현하고 이해하는 것을 목표로 합니다.
