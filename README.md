

## Setting 1)
                                    Hyperparams:
STATE_SPACE = 16
ACTION_SPACE = 4
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.0
EPS_DECAY = 0.00001
TARGET_UPDATE = 50
MY_SEED = 143
LR = 0.0001
MEMORY_SIZE = 1000000
NUM_EPISODES = 2000

                            With network arcitecture:
self.lin1 = torch.nn.Linear(in_features=self.input_size, out_features=self.input_size*20)
self.lin2 = torch.nn.Linear(in_features=self.input_size*20, out_features=self.nr_actions)
def forward(self, state: torch.Tensor) -> torch.Tensor:
    state = state.to(self.device)
    state = self.rectifier(self.lin1(state))
    state = self.lin2(state)
    return state

---------------------------------------------------------------------------------------------------