import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from collections import deque, namedtuple # for storing moves/memory

from environment import Environment
from game import EnvReturnCode

# ========== CONSTANTS ==========
MAX_MEMORY      = 100_000   # max number of moves to remember
BATCH_SIZE      = 1000      # how many moves to learn from
LEARNING_RATE   = 0.001     # how fast the NN learns
GAMMA           = 0.9       # Discount factor
EPSILON_START   = 0.9       # Epsilon-greedy start value
EPSILON_END     = 0.05      # Epsilon-greedy end value
TARGET_UPDATE   = 10        # Update target network every 10 episodes
EPISODES        = 1000      # Number of episodes to train for
STEPS           = 1000      # Number of steps per episode
# ===============================


# Transition to map a (state, action) -> (next_state, reward)
# Essentially what it did in a state and the reward/next state
Transition = namedtuple("Transition",
    ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self):
        self.memory = deque(maxlen=MAX_MEMORY)
    
    def push(self, *args):
        """Save transition to memory"""
        self.memory.append(Transition(*args))

    def sample(self):
        """Return random sample of memory"""
        return random.sample(self.memory, BATCH_SIZE)
    
    def __len__(self):
        return len(self.memory)


class QNN(nn.Module):
    def __init__(self):
        super(QNN, self).__init__()

        self.linear1 = nn.Linear(Environment.INPUT_SIZE, 12)
        self.linear2 = nn.Linear(12, 8)
        self.linear3 = nn.Linear(8, Environment.OUTPUT_SIZE)
    
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.linear1)
        x = F.relu(self.linear2)
        x = F.relu(self.linear3)
        return x


class Agent:
    def __init__(self):
        # The policy network (QNN) does the action selection
        self.policy_network = QNN().to(device)
        # The target network is used to calculate the target Q-value
        self.target_net = QNN().to(device)

        # Copy the weights from the policy network to the target network
        self.target_net.load_state_dict(self.policy_network.state_dict())
        # Set the target network to evaluation mode
        self.target_net.eval()

        # Updates the target network
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayMemory()

        self.criterion = nn.MSELoss()

        self.iterations = 0
    
    def get_action(self, state):
        """Get an action from policy network with epsilon-greedy
        The main function of this is to allow for exploration
            The agent will explore more at the start and less at the end
            The probability of getting a random action will decrease over time
        """
        self.iterations += 1

        # Get a random number
        eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * \
            np.exp(-1. * self.iterations / 200)
        
        # Get an action (whether random or not)
        if random.random() > eps_threshold:
            # Get the action with the highest Q-value
            with torch.no_grad():
                # Get the Q-values for the state by passing it through the policy network
                return self.policy_network(state).max(1)[1].view(1, 1)
        else:
            # Get a random action
            return torch.tensor([[random.randrange(Environment.OUTPUT_SIZE)]], device=device, dtype=torch.long)

    def optimize_model(self):
        """Optimize the policy network by sampling from memory
        In-depth explanation:
            1. Sample a batch of transitions from memory
            2. Get a batch of transitions from memory
            3. Make a non-final mask of the transitions' next states
            4. In the variable next_state_values, get the Q-values for the next states variable called next_state_values
            5. Get the expected Q-values for the current states (the target Q-values) variable called expected_state_action_values
            6. Compute Huber loss
            7. Optimize the model
        """
        # Check if there are enough transitions in memory
        if len(self.memory) < BATCH_SIZE:
            return
        
        # Get a random sample of transitions from memory
        transitions = self.memory.sample()

        # Transpose batch-array of Transitions to Transition of batch-arrays
        # Example Input: zip(*[('a', 1), ('b', 2), ('c', 3), ('d', 4)])
        # Example Output: [('a', 'b', 'c', 'd'), (1, 2, 3, 4)]
        batch = Transition(*zip(*transitions))

        # With the batch, get the non-final states through a mask
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # Get the batches for state, action, and reward
        # The state batch is a tensor representing the state 
        state_batch = torch.cat(batch.state)
        # The action batch is a tensor representing the actions taken in the state
        action_batch = torch.cat(batch.action)
        # The reward batch is a tensor representing the reward for the action taken in the state
        reward_batch = torch.cat(batch.reward)

        # Get the Q-values for the state batch
        # Computes Q(sₜ, a) - the model computes Q(sₜ), and we select columns of actions.
        # These are the actions that would've been taken for each batch state (based on the policy network)
        state_action_values = self.policy_network(state_batch).gather(1, action_batch)

        # Get the next state Q-values (V(sₜ₊₁)) for all non-final or next states
        # We use zero-filled tensor as the default value for the next state values
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        # Get the Q-values for the next state batch
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        # Get the expected Q-values (yₜ)
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss (to be less sensitive to outliers) based on the expected Q-values and the state action values
        # Equation: L = 1/2 * (yₜ - Q(sₜ, a))² where yₜ = rₜ + γ * V(sₜ₊₁)
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Finally, optimize the policy network
        self.optimizer.zero_grad()  # Zero the gradients to prevent accumulation
        loss.backward()  # Backpropagate the loss

        # Clamp the gradients to prevent them from "exploding"
        for param in self.policy_network.parameters():
            # For each parameter in the policy network,
            # clamp the gradient to be between -1 and 1
            param.grad.data.clamp_(-1, 1)
        
        self.optimizer.step()  # Update the weights of the policy network

    
    def train(self):
        """Train the agent"""
        # Create the environment
        env = Environment()

        # Get the initial state
        state = env.reset()

        # Loop through the episodes
        for episode in range(EPISODES):
            # Get the initial state
            state = env.reset()

            # Loop through the steps
            for step in range(STEPS):
                # Get an action from the policy network
                action = self.get_action(state)

                # Get the next state, reward, and result
                next_state, reward, result = env.step(action)
                done = result == EnvReturnCode.GAME_OVER

                # If the episode is done, set the next state to None
                if done:
                    next_state = None
                
                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Update the state
                state = next_state

                # Optimize the policy network
                self.optimize_model()

                # If the episode is done, break out of the loop
                if done:
                    break
            
            # Update the target network
            if episode % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_network.state_dict())

            # Print the episode number and the reward
            print(f"Episode: {episode}, Reward: {reward}")
        
    def save(self, fn="models/policy_net.pth"):
        """Save the policy network"""
        torch.save(self.policy_net.state_dict(), fn)
                


if __name__ == "__main__":
    # If GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the agent
    agent = Agent()

    # Train the agent
    agent.train()

    # Save the policy network
    agent.save()