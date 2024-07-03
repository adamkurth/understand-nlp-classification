# understanding-nlp-classification

## Introduction
This repository contains the code for understanding NLP classification. This code is written just for educational purposes. The Kaggle dataset used in this code is [here](https://www.kaggle.com/datasets/athu1105/book-genre-prediction), and is located under `data` directory. The dataset contains 10,000 books with their title, author, and description. The goal is to predict the genre of the book based on the description. 

First I explored the dataset, and made classes for preprocessing the summaries of each book. Then to extract the features of the dataset I used Word2Vec and the pretrained `google-news-300-Word2Vec` from `gensim` for broader contextual understanding of the kaggle data. Then I used different classification algorithms from `scikit-learn` then `PyTorch` to predict the genre of the book.

This is an ongoing project, and I will be updating the code as I learn more about NLP classification.
