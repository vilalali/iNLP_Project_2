import os
import random
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np

class Vocabulary:
    def __init__(self, tokens):
        self.token_to_index = {}
        self.index_to_token = {}
        self.vocab_size = 0
        self.add_token('<PAD>')
        self.add_token('<UNK>')
        self.build_vocab(tokens)


    def build_vocab(self, tokens):
        for token in tokens:
                self.add_token(token)

    def add_token(self, token):
        if token not in self.token_to_index:
            self.token_to_index[token] = self.vocab_size
            self.index_to_token[self.vocab_size] = token
            self.vocab_size += 1

    def get_index(self, token):
        return self.token_to_index.get(token, self.token_to_index['<UNK>'])

    def get_token(self, index):
        return self.index_to_token.get(index, '<UNK>')


def load_data(file_path):
    """Loads text data from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def preprocess_text(text):
    """Tokenizes text into sentences and words, lowercasing words."""
    sentences = sent_tokenize(text)
    tokenized_sentences = [word_tokenize(s.lower()) for s in sentences]
    return tokenized_sentences

def create_train_test_split(sentences, val_size=500, test_size=500):
    """Splits sentences into training, validation, and test sets."""
    random.shuffle(sentences)
    test_sentences = sentences[:test_size]
    val_sentences = sentences[test_size:test_size+val_size]
    train_sentences = sentences[test_size+val_size:]
    return train_sentences, val_sentences, test_sentences

def build_vocabulary_from_sentences(sentences, min_freq=1):
    """Builds vocabulary from sentences."""
    word_counts = Counter()
    for sentence in sentences:
        word_counts.update(sentence)

    vocab = ['<PAD>', '<UNK>'] # PAD token at index 0
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab.append(word)
    return vocab

def build_vocabulary(sentences, min_freq=1):
    """Builds vocabulary from sentences."""
    tokens = []
    for sentence in sentences:
      tokens.extend(sentence)
    vocab = Vocabulary(tokens)
    return vocab

def prepare_ngram_data(sentences, word_to_index, n):
    """Prepares n-gram data from sentences."""
    ngrams = []
    for sentence in sentences:
        indexed_sentence = [word_to_index.get(word, word_to_index['<UNK>']) for word in sentence]
        if len(indexed_sentence) < n:
            continue
        for i in range(n - 1, len(indexed_sentence)):
            context = indexed_sentence[i - (n - 1):i]
            target = indexed_sentence[i]
            ngrams.append((context, target))
    return ngrams

def prepare_rnn_data(sentences, word_to_index):
    """Prepares RNN sequence data from sentences."""
    sequences = []
    for sentence in sentences:
        indexed_sentence = [word_to_index.get(word, word_to_index['<UNK>']) for word in sentence]
        for i in range(1, len(indexed_sentence)):
            input_seq = indexed_sentence[:i]
            target_word = indexed_sentence[i]
            sequences.append((input_seq, target_word))
    return sequences

# --- 3. Dataset Classes ---

class NgramDataset(Dataset):
    def __init__(self, ngrams):
        self.ngrams = ngrams

    def __len__(self):
        return len(self.ngrams)

    def __getitem__(self, idx):
        context, target = self.ngrams[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)

class RNNSequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_seq, target_word = self.sequences[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_word, dtype=torch.long)


def create_data_loader(dataset, batch_size, shuffle = True, rnn = False):
  if rnn:
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda batch: (pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0), torch.stack([item[1] for item in batch])), pin_memory=True) # pin_memory=True
  else:
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True) # pin_memory=True
  return data_loader

if __name__ == '__main__':
  # Test data loading and preprocessing
  file_path = '../data/Pride-and-Prejudice-Jane-Austen.txt'
  text = load_data(file_path)
  sentences = preprocess_text(text)
  train_sentences, val_sentences, test_sentences = create_train_test_split(sentences)
  vocab = build_vocabulary(train_sentences)
  print(f"Vocabulary Size: {vocab.vocab_size}")
  print(f"Sample tokens: {list(vocab.token_to_index.items())[:10]}")


  # Test Ngram Data
  n_gram = 3
  ngrams = prepare_ngram_data(train_sentences, vocab.token_to_index, n_gram)
  dataset_ngram = NgramDataset(ngrams)
  data_loader_ngram = create_data_loader(dataset_ngram, batch_size= 128)
  for data, target in data_loader_ngram:
      print("Ngram batch shape:", data.shape, target.shape)
      break


  # Test RNN Data
  rnn_sequences = prepare_rnn_data(train_sentences, vocab.token_to_index)
  dataset_rnn = RNNSequenceDataset(rnn_sequences)
  data_loader_rnn = create_data_loader(dataset_rnn, batch_size= 128, rnn = True)
  for data, target in data_loader_rnn:
      print("RNN batch shape:", data.shape, target.shape)
      break

  print("Data Loaders are working fine")