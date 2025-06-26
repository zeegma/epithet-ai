import torch.nn as nn
import torch.nn.functional as F


class PersonalityNN(nn.Module):
    def __init__(self, input_size=15, hidden_size=[64, 32, 16], output_size=8):
        super(PersonalityNN, self).__init__()
        # Linear layer: 15 inpput features -> 64 hidden
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        # Linear layer: 64 hidden -> 32 hidden
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        # Linear layer: 32 hidden -> 16 hidden
        self.fc3 = nn.Linear(hidden_size[1], hidden_size[2])
        # Output layer: 16 hidden -> 8 categories
        self.out = nn.Linear(hidden_size[2], output_size)
        # Regularization technique to prevent overfitting
        self.dropout = nn.Dropout(0.3)
        # Init weights using He
        self.init_weights()

    def init_weights(self):
        # He initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="leaky_relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Layer 1 + leaky ReLU then apply dropout
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = self.dropout(x)

        # Layer 2 + leaky ReLU then apply dropout
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = self.dropout(x)

        # Layer 3 + leaky ReLU then apply dropout
        x = F.leaky_relu(self.fc3(x), negative_slope=0.01)
        x = self.dropout(x)

        # Output layer
        x = self.out(x)

        return x
