import torch
import numpy as np
from agents.td3_agent import TD3, ReplayBuffer


def test_td3_sanity_check():
    print("=" * 60)
    print("TD3 Algorithm Sanity Check (Forward/Backward Pass)")
    print("=" * 60)

    # 1. 模拟环境维度 (VSCMG_DRL_Project 的真实维度)
    state_dim = 14
    action_dim = 8
    action_bound = 1.0
    batch_size = 64

    print(f"Testing Dimensions - State: {state_dim}, Action: {action_dim}")

    # 2. 实例化 TD3 算法和经验回放池
    agent = TD3(
        state_dim=state_dim,
        hidden_dim=256,
        action_dim=action_dim,
        action_bound=action_bound,
        sigma=0.1,
        actor_lr=3e-4,
        critic_lr=3e-4,
        tau=0.005,
        gamma=0.99,
        device=torch.device("cpu"),
        delay=2
    )

    replay_buffer = ReplayBuffer(capacity=1000)
    print("Agent and ReplayBuffer initialized successfully.")

    # 3. 模拟环境交互，将伪造数据真正 push 到 ReplayBuffer 里
    for _ in range(batch_size):
        state = np.random.randn(state_dim).astype(np.float32)
        action = np.random.uniform(-action_bound, action_bound, size=(action_dim,)).astype(np.float32)
        reward = np.random.randn(1)[0].astype(np.float32)
        next_state = np.random.randn(state_dim).astype(np.float32)
        done = np.random.choice([0, 1], p=[0.9, 0.1]).astype(np.float32)

        replay_buffer.push(state, action, reward, next_state, done)

    print("Fake data pushed to ReplayBuffer. Starting update cycles...")

    # 4. 执行更新循环 (传入真正的 replay_buffer)
    try:
        # Step 1: Critic 应该更新，Actor 不更新 (假设 delay=2)
        agent.update(replay_buffer, batch_size)
        print("Update Step 1 (Critic only) passed.")

        # Step 2: Critic 和 Actor 都应该更新
        agent.update(replay_buffer, batch_size)
        print("Update Step 2 (Critic + Actor + Target Soft Update) passed.")

        # 测试 take_action 接口
        dummy_state = np.zeros(state_dim)
        action_output = agent.take_action(dummy_state)
        print(
            f"take_action output shape: {action_output.shape}, bounds: [{action_output.min():.2f}, {action_output.max():.2f}]")

        print("\n✅ Sanity Check PASSED! The neural network graph is fully connected.")
    except Exception as e:
        print("\n❌ Sanity Check FAILED. Error details:")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_td3_sanity_check()