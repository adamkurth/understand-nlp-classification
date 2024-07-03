#model.py
import torch
import torch.nn as nn
import numpy as np

class GenreClassifierLinear(nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, output_dim: int):
        super(GenreClassifierLinear, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False  # Freeze embedding weights
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        x = x.long()  # Cast to Long type
        x = self.embedding(x)  # Apply embedding
        x = torch.mean(x, dim=1)  # Average over the sequence length
        x = torch.sigmoid(self.fc(x))
        return x

class GenreClassifierLogistic(nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, output_dim: int):
        super(GenreClassifierLogistic, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False  # Freeze embedding weights
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        x = x.long()  # Cast to Long type
        x = self.embedding(x)  # Apply embedding
        x = torch.sigmoid(self.fc(x))
        return x

class GenreClassifierCNN(nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, output_dim: int):
        super(GenreClassifierCNN, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False  # Freeze embedding weights
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = x.long()  # Cast to Long type
        x = self.embedding(x).permute(0, 2, 1)  # Apply embedding and permute for Conv1d
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max(x, dim=2)[0]  # Global max pooling
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

class GenreClassifierRNN(nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, output_dim: int):
        super(GenreClassifierRNN, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False  # Freeze embedding weights
        self.rnn = nn.RNN(embedding_dim, 128, batch_first=True)
        self.fc1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = x.long()  # Cast to Long type
        x = self.embedding(x)  # Apply embedding
        x, _ = self.rnn(x)  # Apply RNN
        x = x[:, -1, :]  # Use the output of the last time step
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

class GenreClassifierLSTM(nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, output_dim: int):
        super(GenreClassifierLSTM, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False  # Freeze embedding weights
        self.lstm = nn.LSTM(embedding_dim, 128, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(128 * 2, 128)  # 128 * 2 because it's bidirectional
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = x.long()  # Cast to Long type
        x = self.embedding(x)  # Apply embedding
        x, _ = self.lstm(x)  # Apply LSTM
        x = x[:, -1, :]  # Use the output of the last time step
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

class GenreClassifierGRU(nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, output_dim: int):
        super(GenreClassifierGRU, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False  # Freeze embedding weights
        self.gru = nn.GRU(embedding_dim, 128, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(128 * 2, 128)  # 128 * 2 because it's bidirectional
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = x.long()  # Cast to Long type
        x = self.embedding(x)  # Apply embedding
        x, _ = self.gru(x)  # Apply GRU
        x = x[:, -1, :]  # Use the output of the last time step
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x
