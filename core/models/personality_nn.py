import torch
import torch.nn as nn


class PersonalityNN(nn.Module):
    def __init__(self, input_size=15, hidden_size=32, output_size=8):
        super(PersonalityNN, self).__init__()
        # Linear layer: 15 inpput features -> 32 hidden
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Linear layer: 32 hidden -> 32 hidden
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Output layer: 32 hidden -> 8 categories
        self.fc3 = nn.Linear(hidden_size, output_size)
        # Regularization technique to prevent overfitting
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Layer 1 + ReLU then apply dropout
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)

        # Layer 2 + ReLU then apply dropout
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)

        # Output layer + Softmax
        x = torch.softmax(self.fc3(x), dim=1)

        # Then return softmax activated output
        return x
