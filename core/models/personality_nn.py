import torch
import torch.nn as nn
import torch.nn.functional as F


class PersonalityNN(nn.Module):
    def __init__(
        self,
        fan_in=15,
        num_choices=5,
        emb_dim=4,
        hidden_size=[48, 24],
        fan_out=8,
    ):
        super(PersonalityNN, self).__init__()
        # 15 questions -> 15 embedding layers [shared size: 4]
        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_choices, emb_dim) for _ in range(fan_in)]
        )
        # Calculate input dimensions [15 * 40 = 60]
        input_dim = fan_in * emb_dim
        # Linear layer: 15 inpput features -> 32 hidden
        self.fc1 = nn.Linear(input_dim, hidden_size[0])
        # Linear layer: 32 hidden -> 16 hidden
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        # Output layer: 16 hidden -> 8 categories
        self.out = nn.Linear(hidden_size[1], fan_out)
        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(0.4)
        # Init weights using He
        self.init_weights()

    def init_weights(self):
        # He initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="leaky_relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # X shape: (batch size, 15)
        # Convert each column to an embedding then concatenate
        embeds = [self.embeddings[i](x[:, i].long()) for i in range(x.shape[1])]
        # shape: (batch size, 15 * emb_dim)
        x = torch.cat(embeds, dim=1)

        # Layer 1 + leaky ReLU then no dropout for first layer
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)

        # Layer 2 + leaky ReLU then apply dropout only on final HL
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = self.dropout(x)

        # Output layer
        x = self.out(x)

        return x
