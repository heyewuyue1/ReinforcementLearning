from src.grid_world import GridWorld
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

# 定义神经网络
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义DQN算法
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.9  # discount rate
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 1-1e-4
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # 可能有问题
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor(np.array([i[0] for i in minibatch]))
        actions = torch.LongTensor(np.array([i[1] for i in minibatch]))
        rewards = torch.FloatTensor(np.array([i[2] for i in minibatch]))
        next_states = torch.FloatTensor(np.array([i[3] for i in minibatch]))
        dones = torch.FloatTensor(np.array([i[4] for i in minibatch]))

        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_model(next_states).detach().max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def get_policy(self, env):
        policy_matrix = np.zeros((env.num_states, len(env.action_space)))
        for state in range(env.num_states):
            x = state % env.env_size[0]
            y = state // env.env_size[0]
            state_tensor = torch.FloatTensor([x, y])
            q_values = self.model(state_tensor).detach().numpy()  # 获取 Q 值
            best_action = np.argmax(q_values)  # 找到最大 Q 值对应的动作
            policy_matrix[state, best_action] = 1  # 最大 Q 值对应的动作概率设为 1
        return policy_matrix

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

# 训练DQN算法
def train_dqn(env, episodes=1000, batch_size=32):
    state_size = 2  # Assuming state is (x, y)
    action_size = len(env.action_space)
    agent = DQNAgent(state_size, action_size)

    for e in range(episodes):
        state = env.reset()
        state = np.array(state[0])  # Extract the state tuple
        total_reward = 0

        for _ in range(1000):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(env.action_space[action])
            if e % 100 == 0:
                env.render()
            next_state = np.array(next_state)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                agent.update_target_model()
                print(f"Episode: {e}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
                break

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if not done:
            print(f"Episode: {e}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
    agent.save(f"dqn_model_{e}.pth")
    # 训练结束后，绘制学习到的策略
    policy_matrix = agent.get_policy(env)
    env.render()  # 显示策略
    env.add_policy(policy_matrix)
    env.render()  # 显示策略
    plt.show(block=True)


# 主函数
if __name__ == "__main__":
    env = GridWorld()
    train_dqn(env, episodes=1000)