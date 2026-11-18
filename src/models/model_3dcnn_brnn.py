import torch
import torch.nn as nn


class CNN3D(nn.Module):
    def __init__(self, embedding_dim=16):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool3d(1),
            nn.Flatten()
        )

        self.projection = nn.Linear(16, embedding_dim)

    def forward(self, x):
        return self.projection(self.features(x))


class CNN3DBRNN(nn.Module):
    def __init__(self, embedding_dim=16, hidden_size=8):
        super().__init__()

        self.cnn = CNN3D(embedding_dim)

        self.brnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        batch_size, sequence_length = x.shape[:2]

        embeddings = []

        for t in range(sequence_length):
            embedding = self.cnn(x[:, t])
            embeddings.append(embedding)

        sequence = torch.stack(embeddings, dim=1)

        _, hidden = self.brnn(sequence)

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        combined = torch.cat(
            [forward_hidden, backward_hidden],
            dim=1
        )

        return self.classifier(combined).squeeze(1)
