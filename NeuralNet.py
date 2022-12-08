import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np


class DeepQNetwork(torch.nn.Module):
    def __init__(self, lr: float, input_size: int,
                 nr_actions: int, nr_consecutive_frames: int = 1, seed: int = 0):
        super(DeepQNetwork, self).__init__()

        self.seed = seed  # For rng reproducibility
        torch.manual_seed(self.seed)

        self.lr = lr
        self.input_size = input_size
        self.nr_actions = nr_actions
        self.nr_consecutive_frames = nr_consecutive_frames

        # ----------- Defining Layers in Neural Net ------------ #
        self.lin1 = torch.nn.Linear(in_features=self.input_size, out_features=self.input_size*20)
        self.lin2 = torch.nn.Linear(in_features=self.input_size*20, out_features=self.nr_actions)
        self.rectifier = torch.nn.ReLU()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state = state.to(self.device)
        state = self.rectifier(self.lin1(state))
        state = self.lin2(state)
        return state


class PolicyNetwork(nn.Module):
    """
    Params:
        s_size: state space size
        a_size: action space size
        h_size: number of neurons in hidden layer
    """
    def __init__(self, s_size, a_size, h_size):
        super(PolicyNetwork, self).__init__()

        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = state.float().unsqueeze(0).to(self.device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = np.argmax(m.sample())
        return action.item(), m.log_prob(action)