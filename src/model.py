import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_prob=0.2, num_layers=1): # num_layers=1
        super(RNNLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.dropout_emb = nn.Dropout(dropout_prob)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True, num_layers=num_layers) # num_layers=1
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

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device) # Initialize hidden for num_layers

class LSTMLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_prob=0.2, num_layers=2): # num_layers=2
        super(LSTMLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.dropout_emb = nn.Dropout(dropout_prob)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=num_layers) # num_layers=2
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

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device), # Initialize hidden for num_layers
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)) # Initialize cell state for num_layers

if __name__ == '__main__':
    vocab_size = 1000
    embedding_dim = 32
    context_size = 2
    hidden_dim = 128
    dropout_prob = 0.2
    num_layers = 2
    batch_size = 64
    seq_len = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ffnn = FFNNLM(vocab_size, embedding_dim, context_size, hidden_dim, dropout_prob)
    rnn = RNNLM(vocab_size, embedding_dim, hidden_dim, dropout_prob, num_layers=1)
    lstm = LSTMLM(vocab_size, embedding_dim, hidden_dim, dropout_prob, num_layers)

    dummy_input_ffnn = torch.randint(0, vocab_size, (batch_size, context_size))
    dummy_input_rnn = torch.randint(0, vocab_size, (batch_size, seq_len))
    dummy_input_lstm = torch.randint(0, vocab_size, (batch_size, seq_len))
    dummy_hidden_rnn = rnn.init_hidden(batch_size, device)
    dummy_hidden_lstm = lstm.init_hidden(batch_size, device)

    output_ffnn = ffnn(dummy_input_ffnn)
    output_rnn, _ = rnn(dummy_input_rnn, dummy_hidden_rnn)
    output_lstm, _ = lstm(dummy_input_lstm, dummy_hidden_lstm)


    print("FFNN Output Shape:", output_ffnn.shape)
    print("RNN Output Shape:", output_rnn.shape)
    print("LSTM Output Shape:", output_lstm.shape)