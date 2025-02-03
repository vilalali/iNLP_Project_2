import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import os
import numpy as np
import sys
sys.path.append("./src")
import csv

from model import FFNNLM, RNNLM, LSTMLM
from data_utils import load_data, preprocess_text, create_train_test_split, build_vocabulary, prepare_ngram_data, prepare_rnn_data, NgramDataset, RNNSequenceDataset, create_data_loader


def calculate_perplexity(model, data_loader, criterion, model_type, device, rnn = False):
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
            print(f"\r Batch: {batch_idx + 1}/{len(data_loader)}", end="")
        print() # Print new line for better formatting after the batch progress
    avg_loss = total_loss / word_count if word_count > 0 else 0
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity

def calculate_loss(model, data_loader, criterion, model_type, device, rnn = False):
    """Evaluates the model on the given data loader and returns the average loss."""
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
    return avg_loss


def evaluate_model(model, sentences, vocab, criterion, device, args, is_test = True):
   model.eval()
   total_perplexity = 0.0
   perplexities = []
   if args.lm_type == "ffnn":
      n_gram_size = args.n_gram
      for sent in sentences:
          ngrams = prepare_ngram_data([sent], vocab.token_to_index, n_gram_size)
          if len(ngrams) == 0:
              perplexities.append(float('inf'))
              continue
          dataset = NgramDataset(ngrams)
          data_loader = create_data_loader(dataset, batch_size=args.batch_size, shuffle=False)
          perplexity = calculate_perplexity(model, data_loader, criterion, model_type='ffnn', device=device)
          perplexities.append(perplexity)
          total_perplexity += perplexity


   elif args.lm_type in ["rnn", "lstm"]:
    sequences = prepare_rnn_data(sentences, vocab.token_to_index)
    if len(sequences) == 0:
        return [], float('inf')
    dataset = RNNSequenceDataset(sequences)
    data_loader = create_data_loader(dataset, batch_size=args.batch_size, shuffle=False, rnn=True)
    total_perplexity = calculate_perplexity(model, data_loader, criterion, model_type=args.lm_type, device=device, rnn=True)
    perplexities = [total_perplexity] * len(sentences)

   if len(sentences) > 0:
     avg_perplexity = total_perplexity/ len(sentences) if args.lm_type == "ffnn" else total_perplexity
   else:
     avg_perplexity = float('inf')
   return perplexities, avg_perplexity

def save_perplexities_to_csv(sentences, perplexities, lm_type, corpus_name, model_dir, is_test = True):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    file_prefix = "test" if is_test else "train"
    file_path = os.path.join(model_dir, f"{file_prefix}_{lm_type}_{corpus_name}_perplexity.csv")
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Sentence", "Perplexity"])
        for sentence, perplexity in zip(sentences, perplexities):
            writer.writerow([sentence, f"{perplexity:.4f}"])
    print(f"Perplexity scores saved to {file_path}")

def main():
  parser = argparse.ArgumentParser(description = "Evaluate Neural Language Model")
  parser.add_argument("--lm_type", type=str, required = True, choices=["ffnn", "rnn", "lstm"], help = "Type of language model to train")
  parser.add_argument("--corpus_path", type=str, required=True, help="Path to the corpus")
  parser.add_argument("--embedding_dim", type = int, default = 100, help = "Embedding dimension")
  parser.add_argument("--hidden_dim", type = int, default = 256, help = "Hidden dimension")
  parser.add_argument("--batch_size", type = int, default = 128, help = "Batch Size")
  parser.add_argument("--seq_len", type = int, default = 20, help = "Sequence Length for RNN and LSTM")
  parser.add_argument("--n_gram", type = int, default = 3, help = "N gram size for FFNN model")
  parser.add_argument("--min_freq", type = int, default = 1, help = "Minimum frequency for vocab")
  parser.add_argument("--model_dir", type = str, default = "./models", help = "Directory to save models")
  args = parser.parse_args()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("Device: ", device)

  # Data loading and preprocessing
  text = load_data(args.corpus_path)
  sentences = preprocess_text(text)
  train_sentences, test_sentences, _ = create_train_test_split(sentences)
  vocab = build_vocabulary(train_sentences, min_freq=args.min_freq)
  print(f"{os.path.basename(args.corpus_path).split('.')[0]} Vocabulary size: {vocab.vocab_size}")


  # Initialize Model
  if args.lm_type == "ffnn":
        model = FFNNLM(len(vocab.token_to_index), args.embedding_dim, args.n_gram-1, args.hidden_dim, dropout_prob=0.2).to(device)
  elif args.lm_type == "rnn":
        model = RNNLM(len(vocab.token_to_index), args.embedding_dim, args.hidden_dim, dropout_prob = 0.2, num_layers=1).to(device)
  elif args.lm_type == "lstm":
        model = LSTMLM(len(vocab.token_to_index), args.embedding_dim, args.hidden_dim, dropout_prob = 0.2, num_layers=2).to(device)
  else:
    raise ValueError("Invalid model type")

  # Load Model
  model_path = os.path.join(args.model_dir, f"{args.lm_type}_{os.path.basename(args.corpus_path).split('.')[0]}.pt")
  if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
  model.load_state_dict(torch.load(model_path, map_location=device))

  # Define loss
  criterion = nn.NLLLoss(ignore_index=0)

  print("Evaluating on training set ...")
  train_perplexities, avg_train_perplexity = evaluate_model(model, train_sentences, vocab, criterion, device, args, is_test = False)

  print("Evaluating on test set ...")
  test_perplexities, avg_test_perplexity = evaluate_model(model, test_sentences, vocab, criterion, device, args, is_test = True)

  # Saving the results
  print("Perplexity Results :")
  print(f"Average train perplexity: {avg_train_perplexity:.4f}")
  print(f"Average test perplexity: {avg_test_perplexity:.4f}")


  # Save to CSV
  save_perplexities_to_csv(train_sentences, train_perplexities, args.lm_type, os.path.basename(args.corpus_path).split('.')[0], args.model_dir, is_test = False)
  save_perplexities_to_csv(test_sentences, test_perplexities, args.lm_type, os.path.basename(args.corpus_path).split('.')[0], args.model_dir, is_test = True)


if __name__ == '__main__':
    main()
    
    
def calculate_and_save_perplexities(model, data_loader, criterion, model_type, device, filename, multi_gpu = False, collate_fn=None):
    """Calculates per-sentence perplexities and saves to CSV."""
    losses, perplexities = evaluate_loss_per_sentence(model, data_loader, criterion, model_type, device, multi_gpu)

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Sentence Index", "Loss", "Perplexity"])
        for i, (loss, perplexity) in enumerate(zip(losses, perplexities)):
            writer.writerow([i, f"{loss:.4f}", f"{perplexity:.4f}"])
    print(f"Per-sentence perplexities saved to {filename}")

def calculate_average_perplexity(model, data_loader, criterion, model_type, device, multi_gpu=False):
    """Calculates average perplexity for the entire dataloader."""
    losses, perplexities = evaluate_loss_per_sentence(model, data_loader, criterion, model_type, device, multi_gpu)
    if perplexities:
        return np.mean(perplexities)
    else:
        return float('inf')  # return inf for empty dataloader
    


        calculate_and_save_perplexities(trained_pp_ffnn_N_gram_model, test_loader_ffnn_N_gram_pp, criterion_per_sentence, model_type='ffnn', device=device, filename=os.path.join(args.model_dir, f'ffnn_{args.n_gram}_test_perplexities.csv'), multi_gpu = multi_gpu)
        calculate_and_save_perplexities(trained_pp_ffnn_N_gram_model, train_loader_ffnn_N_gram_pp, criterion_per_sentence, model_type='ffnn', device=device, filename=os.path.join(args.model_dir, f'ffnn_{args.n_gram}_train_perplexities.csv'), multi_gpu = multi_gpu)    
        calculate_and_save_perplexities(trained_pp_rnn_model, test_loader_rnn_pp, criterion_per_sentence, model_type='rnn', device=device, filename=os.path.join(args.model_dir, 'rnn_test_perplexities.csv'), multi_gpu=multi_gpu, collate_fn = collate_fn_rnn_loss)
        calculate_and_save_perplexities(trained_pp_rnn_model, train_loader_rnn_pp, criterion_per_sentence, model_type='rnn', device=device, filename=os.path.join(args.model_dir, 'rnn_train_perplexities.csv'), multi_gpu=multi_gpu, collate_fn=collate_fn_rnn_loss)        