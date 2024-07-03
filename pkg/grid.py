# grid.py
import itertools
from typing import Dict, List, Union
from .classifier import Classifier
from .data import Book, Embedding

def grid_search(book: Book, embedding: Embedding, param_grid: Dict[str, List[Union[str, int, float]]]):
    keys, values = zip(*param_grid.items())
    best_config = None
    best_loss = float('inf')
    
    for config_values in itertools.product(*values):
        config = dict(zip(keys, config_values))
        print(f"Testing configuration: {config}")
        
        classifier = Classifier(book=book, embedding=embedding, config=config)
        classifier.train()
        test_loss = classifier.test_losses[-1]
        
        if test_loss < best_loss:
            best_loss = test_loss
            best_config = config
            print(f"New best configuration: {best_config} with loss {best_loss}")

    return best_config, best_loss

if __name__ == '__main__':
    book = Book()
    embedding = Embedding(data=book)
    
    param_grid = {
        'model_type': ['linear', 'gru', 'cnn', 'rnn', 'lstm', 'logistic'],
        'learning_rate': [0.001, 0.0001],
        'batch_size': [16, 32, 64],
        'epochs': [10, 20]
    }

    best_config, best_loss = grid_search(book, embedding, param_grid)
    print(f"Best configuration: {best_config} with loss {best_loss}")
