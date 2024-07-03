import numpy as np
import pandas as pd
import random
import os
import gc
import psutil
from typing import Tuple, List
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from .data import Book, Embedding
from .model import GenreClassifierLinear, GenreClassifierCNN, GenreClassifierRNN, GenreClassifierLSTM, GenreClassifierGRU

def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory Usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

class Classifier:
    def __init__(self, book: Book, embedding: Embedding, choice: str) -> None:
        self.book: Book = book
        self.embedding: Embedding = embedding 
        self.device = device()
        self.embedding_matrix, self.word_index = self.embedding.create_embedding_matrix()
        self.model = self.build_model(choice=choice).to(self.device)
        self.mlb = MultiLabelBinarizer().fit(book.genres.apply(lambda x: [x]))
        self.class_weights = self.compute_class_weights()
        self.train_losses: List[float] = []
        self.test_losses: List[float] = []

    def compute_class_weights(self) -> torch.Tensor:
        # Count the occurrences of each class
        class_counts = self.book.labels.sum(axis=0).values
        total_samples = len(self.book.labels)
        
        # Compute the class weights
        class_weights = total_samples / (len(class_counts) * class_counts)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        return class_weights

    def build_model(self, choice: str = 'linear') -> nn.Module:
        input_dim, output_dim = len(self.word_index) + 1, len(self.book.labels.columns)
        if choice == 'linear':
            return GenreClassifierLinear(self.embedding_matrix, output_dim)
        elif choice == 'gru':
            return GenreClassifierGRU(self.embedding_matrix, output_dim)
        elif choice == 'cnn':
            return GenreClassifierCNN(self.embedding_matrix, output_dim)
        elif choice == 'rnn':
            return GenreClassifierRNN(self.embedding_matrix, output_dim)
        elif choice == 'lstm':
            return GenreClassifierLSTM(self.embedding_matrix, output_dim)
        elif choice == 'logistic':
            return GenreClassifierLogistic(self.embedding_matrix, output_dim)
        else:
            raise ValueError("Invalid choice. Choose from 'linear', 'cnn', 'rnn', 'lstm', 'gru', 'logistic'.")

    def prepare(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        X = [torch.tensor(self.text_to_sequence(text), dtype=torch.long) for text in self.book.summaries]
        X = nn.utils.rnn.pad_sequence(X, batch_first=True)  # Pad sequences
        y = torch.tensor(self.mlb.transform(self.book.genres.apply(lambda x: [x]).tolist()), dtype=torch.float32)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_test, self.y_test = X_test, y_test  # Store test data for later use
        return X_train, X_test, y_train, y_test

    def text_to_sequence(self, text: List[str]) -> List[int]:
        return [self.word_index.get(word, 0) for word in text]

    def train(self, epochs: int = 10, batch_size: int = 32) -> None:
        X_train, X_test, y_train, y_test = self.prepare()
        criterion = nn.BCELoss(weight=self.class_weights)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        train_data = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_data = torch.utils.data.TensorDataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

        for epoch in range(epochs):
            self.model.train()
            epoch_train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

            self.train_losses.append(epoch_train_loss / len(train_loader))

            self.model.eval()
            epoch_test_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    epoch_test_loss += loss.item()

            self.test_losses.append(epoch_test_loss / len(test_loader))

            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {self.train_losses[-1]}, Test Loss: {self.test_losses[-1]}')
            print_memory_usage()  # Add memory usage check

    def predict(self, random_idx: int) -> pd.DataFrame:
        if random_idx not in range(0, len(self.book.titles)):
            return "Index out of range"
        
        random_title = self.book.titles.iloc[random_idx]
        random_summary = self.concat_tokens(self.book.summaries.iloc[random_idx])
        print(f"Title: {random_title}")
        print(f"Summary: {random_summary}")
        
        sequence = torch.tensor(self.text_to_sequence(random_summary.split()), dtype=torch.long).to(self.device)  # Convert to sequence and tensor
        sequence = nn.utils.rnn.pad_sequence([sequence], batch_first=True)  # Pad sequence
        self.model.eval()
        with torch.no_grad():
            pred = self.model(sequence)
        return pd.DataFrame(data=pred.cpu().numpy(), columns=self.book.labels.columns)
    
    def concat_tokens(self, tokens: List[str]) -> str:
        return " ".join(tokens)

    def evaluate(self) -> pd.DataFrame:
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(self.X_test.to(self.device)).cpu().numpy()
        y_true = self.y_test.cpu().numpy()
        
        # Binarize predictions (0.5 threshold)
        y_pred = (y_pred > 0.5).astype(int)
        
        # Create DataFrame
        true_labels = self.mlb.inverse_transform(y_true)
        pred_labels = self.mlb.inverse_transform(y_pred)
        results_df = pd.DataFrame({'True Labels': [', '.join(labels) for labels in true_labels],
                                   'Predicted Labels': [', '.join(labels) for labels in pred_labels]})
        
        # Print classification report
        print(classification_report(y_true, y_pred, target_names=self.book.labels.columns))
        
        self.y_pred = y_pred  # Store predictions for later use
        return results_df

    def plot_losses(self) -> None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(1, len(self.train_losses) + 1)), y=self.train_losses, mode='lines+markers', name='Train Loss'))
        fig.add_trace(go.Scatter(x=list(range(1, len(self.test_losses) + 1)), y=self.test_losses, mode='lines+markers', name='Test Loss'))

        fig.update_layout(title='Training and Test Loss across Epochs', xaxis_title='Epochs', yaxis_title='Loss')
        fig.show()

    def plot_roc_auc(self):
        fig = make_subplots(rows=1, cols=1, subplot_titles=['ROC Curve'])
        for i in range(self.y_test.shape[1]):
            fpr, tpr, _ = roc_curve(self.y_test[:, i], self.y_pred[:, i])
            roc_auc = auc(fpr, tpr)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (area = {roc_auc:.2f})'))
        
        fig.update_layout(title='Receiver Operating Characteristic (ROC) Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
        fig.show()

    def plot_precision_recall(self):
        fig = make_subplots(rows=1, cols=1, subplot_titles=['Precision-Recall Curve'])
        for i in range(self.y_test.shape[1]):
            precision, recall, _ = precision_recall_curve(self.y_test[:, i], self.y_pred[:, i])
            fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'Precision-Recall curve'))
        
        fig.update_layout(title='Precision-Recall Curve', xaxis_title='Recall', yaxis_title='Precision')
        fig.show()
