import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
import numpy as np
import random
from nltk.tokenize import sent_tokenize, word_tokenize
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import os
import argparse
import optuna  # Import Optuna for hyperparameter tuning

# --- 1. Setup and Data Loading ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_gpus = torch.cuda.device_count()
print(f"Using device: {device}, Number of GPUs available: {n_gpus}")
multi_gpu = n_gpus > 1

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

def create_train_val_test_split(sentences, val_size=500, test_size=500):
    """Splits sentences into training, validation, and test sets."""
    random.shuffle(sentences)
    test_sentences = sentences[:test_size]
    val_sentences = sentences[test_size:test_size+val_size]
    train_sentences = sentences[test_size+val_size:]
    return train_sentences, val_sentences, test_sentences

def build_vocabulary(sentences, min_freq=1):
    """Builds vocabulary from sentences."""
    word_counts = Counter()
    for sentence in sentences:
        word_counts.update(sentence)

    vocab = ['<PAD>', '<UNK>'] # PAD token at index 0
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab.append(word)

    word_to_index = {word: idx for idx, word in enumerate(vocab)}
    index_to_word = {idx: word for idx, word in enumerate(vocab)}
    return vocab, word_to_index, index_to_word

# --- 2. Data Preparation for Models ---

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
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long) # Corrected: Return tensors

class RNNSequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_seq, target_word = self.sequences[idx]
        return input_seq, target_word # Return raw lists/ints, tensors created in DataLoader


# --- 4. Model Definitions ---

class FFNNLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim, dropout_prob=0.2):
        super(FFNNLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.dropout_emb = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.fc1_norm = nn.LayerNorm(hidden_dim)
        self.dropout_hidden = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, context):
        emb = self.dropout_emb(self.embedding_norm(self.embedding(context))).view(context.size(0), -1)
        hidden = F.relu(self.fc1_norm(self.fc1(emb)))
        hidden = self.dropout_hidden(hidden)
        output = F.log_softmax(self.fc2(hidden), dim=1)
        return output

class RNNLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_prob=0.2, num_layers=2): # num_layers=2
        super(RNNLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.dropout_emb = nn.Dropout(dropout_prob)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True, num_layers=num_layers, dropout=dropout_prob) # dropout in RNN layer
        self.rnn_output_norm = nn.LayerNorm(hidden_dim)
        self.dropout_rnn_output = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.num_layers = num_layers # Store num_layers

    def forward(self, input_seq, hidden):
        emb = self.dropout_emb(self.embedding_norm(self.embedding(input_seq)))
        output, hidden = self.rnn(emb, hidden)
        output = self.dropout_rnn_output(self.rnn_output_norm(output))
        output_last_step = output[:, -1, :]
        output_logprobs = F.log_softmax(self.fc(output_last_step), dim=1)
        return output_logprobs, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device) # Initialize hidden for num_layers

class LSTMLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_prob=0.2, num_layers=2): # num_layers=2
        super(LSTMLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.dropout_emb = nn.Dropout(dropout_prob)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=num_layers, dropout=dropout_prob) # dropout in LSTM layer
        self.lstm_output_norm = nn.LayerNorm(hidden_dim)
        self.dropout_lstm_output = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.num_layers = num_layers # Store num_layers


    def forward(self, input_seq, hidden):
        emb = self.dropout_emb(self.embedding_norm(self.embedding(input_seq)))
        output, hidden = self.lstm(emb, hidden)
        output = self.dropout_lstm_output(self.lstm_output_norm(output))
        output_last_step = output[:, -1, :]
        output_logprobs = F.log_softmax(self.fc(output_last_step), dim=1)
        return output_logprobs, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device), # Initialize hidden for num_layers
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)) # Initialize cell state for num_layers

# --- 5. Training and Evaluation Functions ---

def collate_fn_rnn(batch):
    """Pads sequences within a batch and stacks targets."""
    input_seqs, targets = zip(*batch)
    padded_input_seqs = pad_sequence([torch.tensor(seq, dtype=torch.long) for seq in input_seqs], batch_first=True, padding_value=0)
    targets = torch.stack([torch.tensor(target, dtype=torch.long) for target in targets])
    return padded_input_seqs, targets

def evaluate_loss(model, data_loader, criterion, model_type, device):
    """Evaluates the model on the given data loader and returns the average loss."""
    model.eval()
    total_loss = 0
    word_count = 0
    with torch.no_grad():
        for batch in data_loader:
            if model_type == 'ffnn':
                contexts, targets = batch
                contexts = contexts.to(device, non_blocking=True) # non_blocking=True for async GPU transfer
                targets = targets.to(device, non_blocking=True)
                outputs = model(contexts)
            elif model_type in ['rnn', 'lstm']:
                input_seq_batch, targets = batch
                input_seq_batch = input_seq_batch.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                hidden = model.module.init_hidden(input_seq_batch.size(0)) if multi_gpu else model.init_hidden(input_seq_batch.size(0)) # Handle DP hidden init
                outputs, _ = model(input_seq_batch, hidden)
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")

            loss = criterion(outputs, targets)
            if multi_gpu:
                loss = loss.mean() # Average loss over GPUs if using DataParallel
            total_loss += loss.item() * targets.size(0)
            word_count += targets.size(0)
    avg_loss = total_loss / word_count if word_count > 0 else 0
    return avg_loss

def train_model(model, train_loader, val_loader, epochs, learning_rate, model_type, model_name, patience=3):
    """Generic training function for FFNN, RNN, and LSTM models with early stopping."""
    criterion = nn.NLLLoss(ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    model.to(device)
    if multi_gpu:
        model = nn.DataParallel(model) # Wrap model for multi-GPU

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    path_to_best_model = f'best_{model_name}_model.pth'
    epoch_batches_print = len(train_loader) // 5 if len(train_loader) > 5 else 1
    epochs_no_improve = 0

    for epoch in range(epochs):
        total_loss = 0
        model.train() # Ensure train mode is set at the beginning of each epoch
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            if model_type == 'ffnn':
                contexts, targets = batch
                contexts = contexts.to(device, non_blocking=True) # non_blocking=True
                targets = targets.to(device, non_blocking=True)
                outputs = model(contexts)
            elif model_type in ['rnn', 'lstm']:
                input_seq_batch, targets = batch
                input_seq_batch = input_seq_batch.to(device, non_blocking=True) # non_blocking=True
                targets = targets.to(device, non_blocking=True)

                # Correct hidden state initialization for DataParallel
                if multi_gpu:
                    # Get sub-batch size for current GPU (replica)
                    sub_batch_size = input_seq_batch.size(0) // n_gpus
                    # Initialize hidden state with sub-batch size
                    hidden = model.module.init_hidden(sub_batch_size)
                else:
                    hidden = model.init_hidden(input_seq_batch.size(0))

                outputs, _ = model(input_seq_batch, hidden)
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")

            loss = criterion(outputs, targets)
            if multi_gpu:
                loss = loss.mean() # Average loss over GPUs if using DataParallel
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % epoch_batches_print == 0:
                 avg_loss_batch = total_loss / (batch_idx+1)
                 print(f'Epoch: {epoch+1}, Batch: {batch_idx+1}/{len(train_loader)}, Training Batch Loss: {avg_loss_batch:.4f}')

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        avg_val_loss = evaluate_loss(model, val_loader, criterion, model_type, device)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if multi_gpu:
                torch.save(model.module.state_dict(), path_to_best_model) # Save state_dict of the module
            else:
                torch.save(model.state_dict(), path_to_best_model)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs!")
            break

    print(f"Training finished. Best Validation Loss: {best_val_loss:.4f}. Loading best model from {path_to_best_model}")
    if multi_gpu:
        model.module.load_state_dict(torch.load(path_to_best_model, map_location=device)) # Load to correct device
    else:
        model.load_state_dict(torch.load(path_to_best_model, map_location=device))

    # Plotting losses
    plot_loss_curves(train_losses, val_losses, model_name)
    return model

def plot_loss_curves(train_losses, val_losses, model_name):
    """Plots training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss for {model_name}')
    plt.legend()
    plt.show()

def calculate_perplexity(model, data_loader, criterion, model_type):
    """Calculates perplexity using the best saved model."""
    avg_loss = evaluate_loss(model, data_loader, criterion, model_type, device)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity

def tune_hyperparameters(text_path, model_type, data_preparation_func, dataset_class, collate_fn=None, n_gram_size=None, n_trials=10):
    """Tunes hyperparameters using Optuna."""
    pp_text = load_data(text_path)
    pp_sentences = preprocess_text(pp_text)
    pp_train_sentences, pp_val_sentences, _ = create_train_val_test_split(pp_sentences)
    pp_vocab, pp_word_to_index, _ = build_vocabulary(pp_train_sentences)

    def objective(trial):
        embedding_dim = trial.suggest_int('embedding_dim', 8, 64)
        hidden_dim = trial.suggest_int('hidden_dim', 60, 200)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
        dropout_prob = trial.suggest_float('dropout_prob', 0.1, 0.5)
        num_rnn_layers = trial.suggest_int('num_rnn_layers', 1, 3) if model_type in ['rnn', 'lstm'] else None

        if model_type == 'ffnn':
            context_size = n_gram_size - 1
            model = FFNNLM(len(pp_vocab), embedding_dim, context_size, hidden_dim, dropout_prob)
            train_data = data_preparation_func(pp_train_sentences, pp_word_to_index, n_gram_size)
            val_data = data_preparation_func(pp_val_sentences, pp_word_to_index, n_gram_size)
            train_dataset = dataset_class(train_data)
            val_dataset = dataset_class(val_data)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=32, pin_memory=True, num_workers=2)

        elif model_type == 'rnn':
            model = RNNLM(len(pp_vocab), embedding_dim, hidden_dim, dropout_prob, num_rnn_layers)
            train_data = data_preparation_func(pp_train_sentences, pp_word_to_index)
            val_data = data_preparation_func(pp_val_sentences, pp_word_to_index)
            train_dataset = dataset_class(train_data)
            val_dataset = dataset_class(val_data)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=2, collate_fn=collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=32, pin_memory=True, num_workers=2, collate_fn=collate_fn)

        elif model_type == 'lstm':
            model = LSTMLM(len(pp_vocab), embedding_dim, hidden_dim, dropout_prob, num_rnn_layers)
            train_data = data_preparation_func(pp_train_sentences, pp_word_to_index)
            val_data = data_preparation_func(pp_val_sentences, pp_word_to_index)
            train_dataset = dataset_class(train_data)
            val_dataset = dataset_class(val_data)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=2, collate_fn=collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=32, pin_memory=True, num_workers=2, collate_fn=collate_fn)
        else:
            raise ValueError("Invalid model_type")

        trained_model = train_model(model, train_loader, val_loader, epochs=5, learning_rate=learning_rate, model_type=model_type, model_name=f'tune_{model_type}') # Reduced epochs for tuning
        val_loss = evaluate_loss(trained_model, val_loader, nn.NLLLoss(ignore_index=0), model_type, device)
        return val_loss

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    print(f"Best hyperparameters for {model_type}: {best_params}")
    return best_params, pp_vocab, pp_word_to_index

# --- 7. Main Execution ---
if __name__ == '__main__':
    # --- Data Loading and Preprocessing ---
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
    parser.add_argument("--model_dir", type = str, default = "./models", help = "Directory to save models")
    args = parser.parse_args()

    pride_prejudice_path = args.corpus_path

    # --- Hyperparameter Tuning ---
    n_trials_optuna = 5 # Reduced trials for example run, increase for thorough tuning

    # FFNN 3-gram
    ffnn_3gram_best_params, pp_vocab_ffnn_3gram, pp_word_to_index_ffnn_3gram = tune_hyperparameters(
        pride_prejudice_path, 'ffnn', prepare_ngram_data, NgramDataset, n_gram_size=3, n_trials=n_trials_optuna)

    # FFNN 5-gram
    ffnn_5gram_best_params, pp_vocab_ffnn_5gram, pp_word_to_index_ffnn_5gram = tune_hyperparameters(
        pride_prejudice_path, 'ffnn', prepare_ngram_data, NgramDataset, n_gram_size=5, n_trials=n_trials_optuna)

    # RNN
    rnn_best_params, pp_vocab_rnn, pp_word_to_index_rnn = tune_hyperparameters(
        pride_prejudice_path, 'rnn', prepare_rnn_data, RNNSequenceDataset, collate_fn=collate_fn_rnn, n_trials=n_trials_optuna)

    # LSTM
    lstm_best_params, pp_vocab_lstm, pp_word_to_index_lstm = tune_hyperparameters(
        pride_prejudice_path, 'lstm', prepare_rnn_data, RNNSequenceDataset, collate_fn=collate_fn_rnn, n_trials=n_trials_optuna)


    # --- Re-train models with best hyperparameters and evaluate ---
    epochs_final_train = args.epochs # Set epochs for final training
    batch_size= args.batch_size
    # --- FFNN 3-gram ---
    pp_ffnn_3gram_model = FFNNLM(len(pp_vocab_ffnn_3gram), ffnn_3gram_best_params['embedding_dim'], 3-1, ffnn_3gram_best_params['hidden_dim'], dropout_prob=ffnn_3gram_best_params['dropout_prob'])
    pp_text = load_data(args.corpus_path)
    pp_sentences = preprocess_text(pp_text)
    pp_train_sentences, _, _ = create_train_val_test_split(pp_sentences) # Using full train split data for final training.
    pp_ngrams_3_train = prepare_ngram_data(pp_train_sentences[: -1000], pp_word_to_index_ffnn_3gram, 3) # Example using full train data after tuning vocab
    pp_ngrams_3_val = prepare_ngram_data(pp_train_sentences[-1000:-500], pp_word_to_index_ffnn_3gram, 3) # Example using full val data
    pp_ngrams_3_test = prepare_ngram_data(pp_train_sentences[-500:], pp_word_to_index_ffnn_3gram, 3) # Example using full test data

    train_dataset_ffnn_3gram_pp = NgramDataset(pp_ngrams_3_train)
    val_dataset_ffnn_3gram_pp = NgramDataset(pp_ngrams_3_val)
    test_dataset_ffnn_3gram_pp = NgramDataset(pp_ngrams_3_test)
    train_loader_ffnn_3gram_pp = DataLoader(train_dataset_ffnn_3gram_pp, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2) # Optimized DataLoader
    val_loader_ffnn_3gram_pp = DataLoader(val_dataset_ffnn_3gram_pp, batch_size=batch_size, pin_memory=True, num_workers=2) # Optimized DataLoader
    test_loader_ffnn_3gram_pp = DataLoader(test_dataset_ffnn_3gram_pp, batch_size=batch_size, pin_memory=True, num_workers=2) # Optimized DataLoader

    trained_pp_ffnn_3gram_model = train_model(pp_ffnn_3gram_model, train_loader_ffnn_3gram_pp, val_loader_ffnn_3gram_pp, epochs_final_train, model_type='ffnn', model_name='ffnn_3gram', learning_rate=ffnn_3gram_best_params['learning_rate'], patience=3)
    pp_test_perplexity_ffnn_3gram = calculate_perplexity(trained_pp_ffnn_3gram_model, test_loader_ffnn_3gram_pp, nn.NLLLoss(reduction='mean', ignore_index=0), model_type='ffnn')
    print(f"Pride & Prejudice FFNN (3-gram) - Test Perplexity: {pp_test_perplexity_ffnn_3gram:.4f}")

    # --- FFNN 5-gram ---
    pp_ffnn_5gram_model = FFNNLM(len(pp_vocab_ffnn_5gram), ffnn_5gram_best_params['embedding_dim'], 5-1, ffnn_5gram_best_params['hidden_dim'], dropout_prob=ffnn_5gram_best_params['dropout_prob'])
    pp_ngrams_5_train = prepare_ngram_data(preprocess_text(load_data(pride_prejudice_path))[: -1000], pp_word_to_index_ffnn_5gram, 5) # Example using full train data after tuning vocab
    pp_ngrams_5_val = prepare_ngram_data(preprocess_text(load_data(pride_prejudice_path))[-1000:-500], pp_word_to_index_ffnn_5gram, 5) # Example using full val data
    pp_ngrams_5_test = prepare_ngram_data(preprocess_text(load_data(pride_prejudice_path))[-500:], pp_word_to_index_ffnn_5gram, 5) # Example using full test data
    train_dataset_ffnn_5gram_pp = NgramDataset(pp_ngrams_5_train)
    val_dataset_ffnn_5gram_pp = NgramDataset(pp_ngrams_5_val)
    test_dataset_ffnn_5gram_pp = NgramDataset(pp_ngrams_5_test)
    train_loader_ffnn_5gram_pp = DataLoader(train_dataset_ffnn_5gram_pp, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2) # Optimized DataLoader
    val_loader_ffnn_5gram_pp = DataLoader(val_dataset_ffnn_5gram_pp, batch_size=batch_size, pin_memory=True, num_workers=2) # Optimized DataLoader
    test_loader_ffnn_5gram_pp = DataLoader(test_dataset_ffnn_5gram_pp, batch_size=batch_size, pin_memory=True, num_workers=2) # Optimized DataLoader
    trained_pp_ffnn_5gram_model = train_model(pp_ffnn_5gram_model, train_loader_ffnn_5gram_pp, val_loader_ffnn_5gram_pp, epochs_final_train, model_type='ffnn', model_name='ffnn_5gram', learning_rate=ffnn_5gram_best_params['learning_rate'], patience=3)
    pp_test_perplexity_ffnn_5gram = calculate_perplexity(trained_pp_ffnn_5gram_model, test_loader_ffnn_5gram_pp, nn.NLLLoss(reduction='mean', ignore_index=0), model_type='ffnn')
    print(f"Pride & Prejudice FFNN (5-gram) - Test Perplexity: {pp_test_perplexity_ffnn_5gram:.4f}")


    # --- RNN ---
    pp_rnn_model = RNNLM(len(pp_vocab_rnn), rnn_best_params['embedding_dim'], rnn_best_params['hidden_dim'], dropout_prob=rnn_best_params['dropout_prob'], num_layers = rnn_best_params['num_rnn_layers'] if 'num_rnn_layers' in rnn_best_params else 2)
    pp_text = load_data(args.corpus_path)
    pp_sentences = preprocess_text(pp_text)
    pp_train_sentences, _, _ = create_train_val_test_split(pp_sentences)
    pp_rnn_sequences_train = prepare_rnn_data(pp_train_sentences[: -1000], pp_word_to_index_rnn)
    pp_rnn_sequences_val = prepare_rnn_data(pp_train_sentences[-1000:-500], pp_word_to_index_rnn)
    pp_rnn_sequences_test = prepare_rnn_data(pp_train_sentences[-500:], pp_word_to_index_rnn)
    train_dataset_rnn_pp = RNNSequenceDataset(pp_rnn_sequences_train)
    val_dataset_rnn_pp = RNNSequenceDataset(pp_rnn_sequences_val)
    test_dataset_rnn_pp = RNNSequenceDataset(pp_rnn_sequences_test)
    train_loader_rnn_pp = DataLoader(train_dataset_rnn_pp, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_rnn, pin_memory=True, num_workers=2) # Optimized DataLoader
    val_loader_rnn_pp = DataLoader(val_dataset_rnn_pp, batch_size=batch_size, collate_fn=collate_fn_rnn, pin_memory=True, num_workers=2) # Optimized DataLoader
    test_loader_rnn_pp = DataLoader(test_dataset_rnn_pp, batch_size=batch_size, collate_fn=collate_fn_rnn, pin_memory=True, num_workers=2) # Optimized DataLoader
    trained_pp_rnn_model = train_model(pp_rnn_model, train_loader_rnn_pp, val_loader_rnn_pp, epochs_final_train, model_type='rnn', model_name='rnn', learning_rate=rnn_best_params['learning_rate'], patience=3)
    pp_test_perplexity_rnn = calculate_perplexity(trained_pp_rnn_model, test_loader_rnn_pp, nn.NLLLoss(reduction='mean', ignore_index=0), model_type='rnn')
    print(f"Pride & Prejudice RNN - Test Perplexity: {pp_test_perplexity_rnn:.4f}")

    # --- LSTM ---
    pp_lstm_model = LSTMLM(len(pp_vocab_lstm), lstm_best_params['embedding_dim'], lstm_best_params['hidden_dim'], dropout_prob=lstm_best_params['dropout_prob'], num_layers=lstm_best_params['num_rnn_layers'] if 'num_rnn_layers' in lstm_best_params else 2)
    pp_rnn_sequences_train = prepare_rnn_data(preprocess_text(load_data(pride_prejudice_path))[: -1000], pp_word_to_index_lstm)
    pp_rnn_sequences_val = prepare_rnn_data(preprocess_text(load_data(pride_prejudice_path))[-1000:-500], pp_word_to_index_lstm)
    pp_rnn_sequences_test = prepare_rnn_data(preprocess_text(load_data(pride_prejudice_path))[-500:], pp_word_to_index_lstm)
    train_dataset_lstm_pp = RNNSequenceDataset(pp_rnn_sequences_train)
    val_dataset_lstm_pp = RNNSequenceDataset(pp_rnn_sequences_val)
    test_dataset_lstm_pp = RNNSequenceDataset(pp_rnn_sequences_test)
    train_loader_lstm_pp = DataLoader(train_dataset_lstm_pp, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_rnn, pin_memory=True, num_workers=2) # Optimized DataLoader
    val_loader_lstm_pp = DataLoader(val_dataset_lstm_pp, batch_size=batch_size, collate_fn=collate_fn_rnn, pin_memory=True, num_workers=2) # Optimized DataLoader
    test_loader_lstm_pp = DataLoader(test_dataset_lstm_pp, batch_size=batch_size, collate_fn=collate_fn_rnn, pin_memory=True, num_workers=2) # Optimized DataLoader
    trained_pp_lstm_model = train_model(pp_lstm_model, train_loader_lstm_pp, val_loader_lstm_pp, epochs_final_train, model_type='lstm', model_name='lstm', learning_rate=lstm_best_params['learning_rate'], patience=3)
    pp_test_perplexity_lstm = calculate_perplexity(trained_pp_lstm_model, test_loader_lstm_pp, nn.NLLLoss(reduction='mean', ignore_index=0), model_type='lstm')
    print(f"Pride & Prejudice LSTM - Test Perplexity: {pp_test_perplexity_lstm:.4f}")


    print("="*50) # Separator
    print("Remember to train and evaluate Ulysses models as well, following the same structure!")
    print("="*50)
    print(f"FFNN (3-gram) best params {ffnn_3gram_best_params}")
    print(f"FFNN (5-gram) best params {ffnn_5gram_best_params}")
    print(f"RNN best params {rnn_best_params}")
    print(f"LSTM best params {lstm_best_params}")