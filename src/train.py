import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import argparse
import sys
sys.path.append("./src")
import matplotlib.pyplot as plt
import numpy as np

from model import FFNNLM, RNNLM, LSTMLM
from data_utils import load_data, preprocess_text, create_train_test_split, build_vocabulary, prepare_ngram_data, prepare_rnn_data, NgramDataset, RNNSequenceDataset, create_data_loader
from eval import calculate_loss

def calculate_perplexity(model, data_loader, criterion, model_type, device):
    """Calculates perplexity using the best saved model."""
    model.eval()
    total_loss = 0
    word_count = 0
    with torch.no_grad():
        for data in data_loader:
            if model_type == 'ffnn':
                contexts, targets = data
                contexts, targets = contexts.to(device), targets.to(device)
                outputs = model(contexts)
            elif model_type in ['rnn', 'lstm']:
                input_seq_batch, targets = data
                input_seq_batch, targets = input_seq_batch[0].to(device), input_seq_batch[1].to(device)
                hidden = model.init_hidden(input_seq_batch.size(0), device)
                outputs, _ = model(input_seq_batch, hidden)
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")

            loss = criterion(outputs, targets)
            total_loss += loss.item() * targets.size(0)
            word_count += targets.size(0)

    avg_loss = total_loss / word_count if word_count > 0 else 0
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity

def train_model(model, train_loader, val_loader, epochs, learning_rate, model_type, model_name, device, args):
    """Generic training function for FFNN, RNN, and LSTM models."""
    criterion = nn.NLLLoss(ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate) # AdamW Optimizer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    model.to(device)
    model.train()

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    path_to_best_model = f'best_{model_name}_model.pth'
    epoch_batches_print = len(train_loader) // 5 if len(train_loader) > 5 else 1

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        hidden = None # Initialize hidden for every epoch
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            if model_type == 'ffnn':
                contexts, targets = batch
                contexts, targets = contexts.to(device), targets.to(device)
                outputs = model(contexts)
            elif model_type in ['rnn', 'lstm']:
                input_seq_batch, targets = batch
                input_seq_batch, targets = input_seq_batch[0].to(device), input_seq_batch[1].to(device)
                if hidden is None or isinstance(model, RNNLM):
                    hidden = model.init_hidden(input_seq_batch.size(0), device) # Initialize hidden at first batch or if model is RNN
                outputs, hidden = model(input_seq_batch, hidden)
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx == 0 or (batch_idx + 1) % epoch_batches_print == 0 or batch_idx == len(train_loader) - 1 :
                 avg_loss_batch = total_loss / (batch_idx+1)
                 print(f'\rEpoch: {epoch+1}, Batch: {batch_idx+1}/{len(train_loader)}, Training Batch Loss: {avg_loss_batch:.4f}', end="")
        print() # New Line for formating purposes.

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        avg_val_loss = calculate_loss(model, val_loader, criterion, model_type, device)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), path_to_best_model)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

    print(f"Training finished. Best Validation Loss: {best_val_loss:.4f}. Loading best model from {path_to_best_model}")
    model.load_state_dict(torch.load(path_to_best_model))

    # Plotting losses
    plot_loss_curves(train_losses, val_losses, model_name, args.model_dir, os.path.basename(args.corpus_path).split('.')[0])

    return model

def plot_loss_curves(train_losses, val_losses, model_name, model_dir, corpus_name):
    """Plots training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss for {model_name}')
    plt.legend()
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    plot_path = os.path.join(model_dir, f"{model_name}_training_plot_{corpus_name}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Training plot saved to {plot_path}")


def main():
  parser = argparse.ArgumentParser(description="Train Neural Language Model")
  parser.add_argument("--lm_type", type=str, required = True, choices=["ffnn", "rnn", "lstm"], help = "Type of language model to train")
  parser.add_argument("--corpus_path", type=str, required=True, help="Path to the corpus")
  parser.add_argument("--epochs", type = int, default = 10, help = "Number of epochs")
  parser.add_argument("--embedding_dim", type = int, default = 100, help = "Embedding dimension")
  parser.add_argument("--hidden_dim", type = int, default = 256, help = "Hidden dimension")
  parser.add_argument("--batch_size", type = int, default = 128, help = "Batch Size")
  parser.add_argument("--seq_len", type = int, default = 20, help = "Sequence Length for RNN and LSTM")
  parser.add_argument("--n_gram", type = int, default = 3, help = "N gram size for FFNN model")
  parser.add_argument("--min_freq", type = int, default = 1, help = "Minimum frequency for vocab")
  parser.add_argument("--lr", type = float, default = 0.001, help = "Learning rate")
  parser.add_argument("--model_dir", type = str, default = "../models", help = "Directory to save models")
  args = parser.parse_args()

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("Device: ", device)

  # --- Data Loading and Preprocessing ---
  text = load_data(args.corpus_path)
  sentences = preprocess_text(text)
  train_sentences, test_sentences, _ = create_train_test_split(sentences)
  vocab = build_vocabulary(train_sentences, min_freq=args.min_freq)

  print(f"{os.path.basename(args.corpus_path).split('.')[0]} Vocabulary size: {vocab.vocab_size}")

  # --- Prepare Data for Models ---
  n_gram_size = args.n_gram

  if args.lm_type == 'ffnn':
        train_data = prepare_ngram_data(train_sentences, vocab.token_to_index, n_gram_size)
        val_data = prepare_ngram_data(test_sentences, vocab.token_to_index, n_gram_size)
        train_dataset = NgramDataset(train_data)
        val_dataset = NgramDataset(val_data)
        train_loader = create_data_loader(train_dataset, args.batch_size, shuffle = True)
        val_loader = create_data_loader(val_dataset, args.batch_size)
        model = FFNNLM(len(vocab.token_to_index), args.embedding_dim, n_gram_size - 1, args.hidden_dim, dropout_prob = 0.2).to(device)

  elif args.lm_type == 'rnn':
        train_data = prepare_rnn_data(train_sentences, vocab.token_to_index)
        val_data = prepare_rnn_data(test_sentences, vocab.token_to_index)
        train_dataset = RNNSequenceDataset(train_data)
        val_dataset = RNNSequenceDataset(val_data)
        train_loader = create_data_loader(train_dataset, args.batch_size, shuffle = True, rnn = True)
        val_loader = create_data_loader(val_dataset, args.batch_size, rnn = True)
        model = RNNLM(len(vocab.token_to_index), args.embedding_dim, args.hidden_dim, dropout_prob = 0.2, num_layers=1).to(device)

  elif args.lm_type == 'lstm':
        train_data = prepare_rnn_data(train_sentences, vocab.token_to_index)
        val_data = prepare_rnn_data(test_sentences, vocab.token_to_index)
        train_dataset = RNNSequenceDataset(train_data)
        val_dataset = RNNSequenceDataset(val_data)
        train_loader = create_data_loader(train_dataset, args.batch_size, shuffle = True, rnn = True)
        val_loader = create_data_loader(val_dataset, args.batch_size, rnn = True)
        model = LSTMLM(len(vocab.token_to_index), args.embedding_dim, args.hidden_dim, dropout_prob = 0.2, num_layers=2).to(device)
  else:
        raise ValueError(f"Invalid model type: {args.lm_type}")



  # --- Initialize and Train Model---
  trained_model = train_model(model, train_loader, val_loader, args.epochs, args.lr, args.lm_type, args.lm_type, device, args)

  # --- Save Model ---
  if not os.path.exists(args.model_dir):
      os.makedirs(args.model_dir)
  model_path = os.path.join(args.model_dir, f"{args.lm_type}_{os.path.basename(args.corpus_path).split('.')[0]}.pt")
  torch.save(trained_model.state_dict(), model_path)
  print(f"Model saved to {model_path}")




if __name__ == '__main__':
    main()