import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

#сама модель
class SentimentModel(nn.Module):
    def __init__(self, embedding_matrix, embed_dim, hidden_dim, num_classes):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            freeze=False,
            padding_idx=0
        )

        #dropout для борьбы с переобучением
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, lengths):
        embedded = self.embedding(x)

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (hidden, _) = self.lstm(packed)

        out = torch.cat((hidden[-2], hidden[-1]), dim=1)
        out = self.dropout(out)

        return self.fc(out)