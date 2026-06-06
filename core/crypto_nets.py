import torch
import torch.nn as nn

class EncryptorDecoder(nn.Module):
    def __init__(self, block_dim, role=None):
        super(EncryptorDecoder, self).__init__()
        self.block_dim = block_dim
        self.role = role if role else "CommParty"

        self.fc_input = nn.Linear(block_dim * 2, block_dim * 2)

        self.conv1 = nn.Conv1d(1, 2, kernel_size=4, stride=1, padding=2)
        self.conv2 = nn.Conv1d(2, 2, kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(2, 4, kernel_size=1, stride=1)
        self.conv4 = nn.Conv1d(4, 1, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.fc_input(x)
        x = torch.sigmoid(x)
        x = x.unsqueeze(1)                     # add channel dim
        x = self.conv1(x)
        x = torch.sigmoid(x)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        x = self.conv3(x)
        x = torch.sigmoid(x)
        x = self.conv4(x)
        x = torch.tanh(x)
        return x.view(-1, self.block_dim)


class Eavesdropper(nn.Module):
    def __init__(self, block_dim, role=None):
        super(Eavesdropper, self).__init__()
        self.block_dim = block_dim
        self.role = role if role else "Adversary"

        self.fc_input = nn.Linear(block_dim, block_dim * 2)

        self.conv1 = nn.Conv1d(1, 2, kernel_size=4, stride=1, padding=2)
        self.conv2 = nn.Conv1d(2, 2, kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(2, 4, kernel_size=1, stride=1)
        self.conv4 = nn.Conv1d(4, 1, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.fc_input(x)
        x = torch.sigmoid(x)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = torch.sigmoid(x)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        x = self.conv3(x)
        x = torch.sigmoid(x)
        x = self.conv4(x)
        x = torch.tanh(x)
        return x.view(-1, self.block_dim)