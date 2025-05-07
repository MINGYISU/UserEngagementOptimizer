import torch
import torch.nn as nn
import torch.nn.functional as F

class SessionModel(nn.Module):
    def __init__(self, feature_sizes, embedding_dim=64, hidden_size=256, out_size=1, dropout=0.3):
        super(SessionModel, self).__init__()
        self.embedding = nn.ModuleList([
            nn.Embedding(feature_size, embedding_dim)
            for feature_size in feature_sizes])
        self.dropout = nn.Dropout(dropout)
        self.lstm_head = nn.LSTM(input_size=embedding_dim * len(feature_sizes), 
                            hidden_size=hidden_size, 
                            batch_first=True)
        self.fc = nn.Linear(in_features=embedding_dim * len(feature_sizes), 
                            out_features=hidden_size)
        self.lstm_hidden = nn.LSTM(input_size=hidden_size,
                            hidden_size=hidden_size,
                            batch_first=True)
        self.out = nn.Linear(in_features=hidden_size, out_features=out_size)

    def forward(self, x):
        _, _, num_features = x.size()
        embedded = []
        for i in range(num_features):
            embedded.append(self.embedding[i](x[:, :, i]))
        embedded = torch.cat(embedded, dim=-1)
        embedded = self.dropout(embedded)
        lstm_out, _ = self.lstm_head(embedded)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.fc(lstm_out)
        lstm_out = self.dropout(lstm_out)
        lstm_out, _ = self.lstm_hidden(lstm_out)
        lstm_out = lstm_out[:, -1, :]
        out = self.out(lstm_out)
        return out