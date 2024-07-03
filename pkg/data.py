# data.py
import pandas as pd
import numpy as np
import gensim   
import nltk
from nltk.tokenize import word_tokenize
import gensim.downloader as api
from pathlib import Path
from collections import namedtuple
from typing import List, Union, Any,Tuple
import csv

@staticmethod
def preprocess_text(text: str) -> List[str]:
    tokens = word_tokenize(text.lower())
    # non-alphabetic characters
    tokens = [token for token in tokens if token.isalpha()]
    return tokens

class Book:
    def __init__(self) -> None:
        self._books: pd.DataFrame = pd.read_csv('data/books.csv')
        self._titles = self._books['title']
        self._genres = self._books['genre']
        self._summaries = self._books['summary'].apply(preprocess_text)
        self._labels = self._prepare_binary_df()
        self._corpus = self._prepare_corpus()

    def _prepare_binary_df(self) -> pd.DataFrame:
        unique_genres = self._genres.unique()
        binary_df = pd.DataFrame(columns=unique_genres)
        for genre in unique_genres:
            binary_df[genre] = self._genres.apply(lambda x: 1 if x == genre else 0)
        return binary_df
    
    def _prepare_corpus(self) -> List[List[str]]:
        return [preprocess_text(summary) for summary in self._books['summary']]

    @property
    def titles(self):
        return self._titles

    @property
    def genres(self):
        return self._genres

    @property
    def summaries(self):
        return self._summaries

    @property
    def labels(self):
        return self._labels

    @property
    def corpus(self):
        return self._corpus

    def get_summary(self, title: str) -> pd.Series:
        book = self._books[self._books['title'] == title]
        return book['summary']
    
    def get_data(self) -> namedtuple:
        return namedtuple('data', ['titles', 'genres', 'summaries', 'labels', 'corpus'])(self._titles, self._genres, self._summaries, self._labels, self._corpus)

class StackData:
    def __init__(self) -> None:
        self._archive = Path('/Users/adamkurth/Documents/vscode/code/nlp-demos/data/archive')
        self._questions_path = self._archive / 'Questions.csv'
        self._tags_path = self._archive / 'Tags.csv'
        self._titles = []
        self._bodies = []
        self._labels = []
        self._corpus = []
        self._load_data()

    def _load_data(self) -> None:
        # Read the CSV files using pandas with error handling
        questions = pd.read_csv(self._questions_path, encoding='latin1', quoting=csv.QUOTE_NONE, on_bad_lines='skip')
        tags = pd.read_csv(self._tags_path, encoding='latin1')

        # Convert columns to appropriate data types after loading
        questions['Id'] = pd.to_numeric(questions['Id'], errors='coerce').dropna().astype('int64')
        questions['Score'] = pd.to_numeric(questions['Score'], errors='coerce').dropna().astype('int64')

        # Process the data
        self._titles = questions['Title'].tolist()
        self._bodies = questions['Body'].apply(preprocess_text).tolist()
        self._corpus = self._bodies
        self._labels = self._prepare_binary_df(tags)

    def _prepare_binary_df(self, tags: pd.DataFrame) -> pd.DataFrame:
        all_tags = tags['Tag'].unique()
        binary_df = pd.DataFrame(columns=all_tags)
        for tag in all_tags:
            binary_df[tag] = tags.groupby('Id')['Tag'].apply(lambda x: tag in x.values).astype(int)
        return binary_df

    @property
    def titles(self):
        return self._titles

    @property
    def bodies(self):
        return self._bodies

    @property
    def labels(self):
        return self._labels

    @property
    def corpus(self):
        return self._corpus

    def get_question(self, title: str) -> pd.Series:
        index = self._titles.index(title)
        return pd.Series(self._bodies[index])
    
    def get_data(self) -> namedtuple:
        return namedtuple('data', ['titles', 'bodies', 'labels', 'corpus'])(self._titles, self._bodies, self._labels, self._corpus)

class Embedding:
    def __init__(self, data: Union[Book, StackData], train: bool = False) -> None:
        self._data:Union[Book, StackData] = data
        self._path: Path = Path('/Users/adamkurth/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz')
        self.keyed = gensim.models.KeyedVectors.load_word2vec_format(self.path, binary=True, limit=int(1e6)) #pretrained word2vec
        self.model = gensim.models.Word2Vec(vector_size=self.keyed.vector_size, window=5, min_count=2, workers=5)
        self.setup()
        self.doc_vectors:pd.DataFrame = self.vectorize_docs()
        self.train_model(corpus=self.data.corpus, update=False) if train else None

    def train_model(self, corpus: List[List[str]], epochs: int = 5, update: bool = True) -> None:
        self.model.build_vocab(corpus, update=update)
        self.model.train(corpus, total_examples=self.model.corpus_count, epochs=epochs)

    def setup(self) -> None:
        # maintain internal structure
        self.model.wv.vectors = self.keyed.vectors  # Transfer weights to model
        self.model.wv.index_to_key = self.keyed.index_to_key  # Transfer index to key
        self.model.wv.key_to_index = self.keyed.key_to_index  # Transfer key to index

    def create_embedding_matrix(self) -> Tuple[np.ndarray, dict]:
        vocab = self.keyed.key_to_index
        embedding_matrix = np.zeros((len(vocab) + 1, self.keyed.vector_size))
        word_index = {word: idx + 1 for idx, word in enumerate(vocab)}

        for word, idx in word_index.items():
            if word in self.keyed:
                embedding_matrix[idx] = self.keyed[word]

        return embedding_matrix, word_index

    # @property: b.path (getter)
    # @path.setter: b.path = 'path' (setter)
    # @path.deleter: del b.path 
    @property
    def data(self):
        return self._data

    @property
    def path(self):
        return self._path

    def save_model(self) -> None:
        # save Word2Vec model
        self.model.save(self._path)

    def load_model(self) -> None:
        # load Word2Vec model 
        self.model = gensim.models.Word2Vec.load(self._path)

    # convert all docs to vectors
    def vectorize_docs(self) -> pd.DataFrame:
        vectors = [self.get_document_vector(self.model.wv, doc) for doc in self.data.corpus]
        return pd.DataFrame(vectors, columns=[f"dim_{i}" for i in range(self.model.vector_size)])
    
    # get vector for a specific title
    def get_vector(self, title: str) -> np.ndarray:
        if isinstance(self.data, Book):
            text = self.data.get_summary(title=title) 
        elif isinstance(self.data, StackData):
            text = self.data.get_question(title=title)
        # tokenize desc/summary
        tokenized_text = preprocess_text(text.iloc[0])
        return self.get_document_vector(model=self.model.wv, document=tokenized_text)
    
    @staticmethod
    def get_document_vector(model: gensim.models.KeyedVectors, document: List[str]) -> np.ndarray: 
        """Convert a document to a vector by averaging the vectors of the words in the document."""
        words = [word for word in document if word in model.key_to_index]
        if not words:  # If no words in the document are in the model, return a zero vector
            return np.zeros(model.vector_size) # return zero vector
        return np.mean([model[word] for word in words], axis=0) # average of all word vectors
        
    def compute_similarity(self, word1: str, word2: str) -> float:
        # cosine similarity
        return self.model.wv.similarity(word1, word2)

    def complete_analogy(self, word1: str, word2: str, word3: str) -> float:
        # complete analogy
        return self.model.wv.most_similar(positive=[word2, word3], negative=[word1])

    def most_similar(self, word:str, topn:int = 10)-> List[str]:
        # most similar to given word
        return self.model.wv.most_similar(word, topn=topn)

if __name__ == "__main__":
    # # book
    book_data = Book()
    embedding = Embedding(data=book_data, path=Path('~/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz'), train=True)
    
    print(book_data.titles.head())
    print(book_data.genres.head())
    print(book_data.summaries.head())
    print(book_data.labels.head())
    print(book_data.corpus[:5])

    # stack 
    # stack_data = StackData()
    # embedding = Embedding(data=stack_data, path=Path('~/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz'), train=True)

    # embedding
    print(embedding.doc_vectors.head(50))
    print('', embedding.compute_similarity('king', 'running'))
    print(embedding.complete_analogy('king', 'queen', 'man'))
    print(embedding.most_similar('king'))
    


    

    