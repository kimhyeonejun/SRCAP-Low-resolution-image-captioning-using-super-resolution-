import torch
import os
import torch.nn as nn

class ProposalClassifier(nn.Module):
    def __init__(self, num_proposals, hidden_size=256):
        super(ProposalClassifier, self).__init__()
        self.num_proposals = num_proposals

        # Define layers
        self.fc1 = nn.Linear(2048, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        #self.sigmoid = nn.Sigmoid()  # Activation function

    def forward(self, x):
        # Input x has shape (batch_size, num_proposals, 2048)
        batch_size = x.size(0)

        # Reshape input to (batch_size * num_proposals, 2048)
        x = x.view(-1, 2048)

        # Apply fully connected layers
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        # Reshape output to (batch_size, num_proposals)
        #print(x)
        x = x.view(batch_size, self.num_proposals)

        return x