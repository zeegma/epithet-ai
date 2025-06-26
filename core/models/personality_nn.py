import torch
import torch.nn as nn


class PersonalityNN(nn.Module):
    def __init__(self, input_size=15, hidden_size=16, output_size=8):
        super(PersonalityNN, self).__init__()
        # Linear layer: 15 inpput features -> 16 hidden
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Output layer: 16 hidden -> 8 categories
        self.fc2 = nn.Linear(hidden_size, output_size)
        # Regularization technique to prevent overfitting
        self.dropout = nn.Dropout(0.3)
        self.init_weights()

    def init_weights(self):
        # He initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Layer 1 + ReLU then apply dropout
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)

        # Output layer
        x = self.fc2(x)

        return x
