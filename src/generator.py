import torch
import argparse
import os
import sys
sys.path.append("./src")
from model import FFNN, RNN, LSTM
from data_utils import load_and_preprocess_data, tokenize_text
import torch.nn.functional as F


def predict_next_word(model, input_sentence, vocab, k, device, n_gram = 3):
    model.eval() # set the model in evaluation mode
    with torch.no_grad():
        tokens = tokenize_text(input_sentence)
        if isinstance(model, FFNN):
            if len(tokens) < n_gram:
                print("Input should have at least n words for this FFNN model.")
                return
            input_indices = [vocab.get_index(token) for token in tokens[-n_gram:]]
            input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device) # Add batch dimension
            output = model(input_tensor)

        elif isinstance(model, (RNN, LSTM)):
            input_indices = [vocab.get_index(token) for token in tokens]
            input_tensor = torch.tensor(input_indices, dtype = torch.long).unsqueeze(0).to(device)
            output = model(input_tensor)
            output = output[:, -1, :] # select output of last time step

        probabilities = F.softmax(output, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probabilities, k, dim=-1)
        top_k_words = [vocab.get_token(idx.item()) for idx in top_k_indices.squeeze()]
        top_k_probs = top_k_probs.squeeze().tolist()
        return list(zip(top_k_words, top_k_probs))



def main():
    parser = argparse.ArgumentParser(description="Generate next words using a trained model.")
    parser.add_argument("--lm_type", type=str, choices=["ffnn", "rnn", "lstm"], required = True, help = "Type of the language model")
    parser.add_argument("corpus_path", type=str, help = "Path to the corpus file")
    parser.add_argument("k", type=int, help = "Number of next word candidates")
    parser.add_argument("--embedding_dim", type=int, default = 100, help = "embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=256, help = "hidden dimension")
    parser.add_argument("--n_gram", type = int, default = 3, help = "N gram size for FFNN model")
    parser.add_argument("--min_freq", type = int, default = 2, help = "Min frequency for vocabulary")
    parser.add_argument("--model_dir", type = str, default = "./models", help = "Directory for loading models.")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Vocabulary
    _, vocab = load_and_preprocess_data(args.corpus_path, min_freq=args.min_freq)

    # Load Model
    model_path = os.path.join(args.model_dir, f"{args.lm_type}_{os.path.basename(args.corpus_path).split('.')[0]}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    if args.lm_type == "ffnn":
        model = FFNN(vocab.vocab_size, args.embedding_dim, args.hidden_dim, args.n_gram).to(device)
    elif args.lm_type == "rnn":
        model = RNN(vocab.vocab_size, args.embedding_dim, args.hidden_dim).to(device)
    elif args.lm_type == "lstm":
        model = LSTM(vocab.vocab_size, args.embedding_dim, args.hidden_dim).to(device)
    else:
        raise ValueError("Invalid model type")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    while True:
        input_sentence = input("Input Sentence: ")
        if input_sentence.lower() == "exit":
            break
        next_words_probs = predict_next_word(model, input_sentence, vocab, args.k, device, args.n_gram)
        if next_words_probs:
           print("Output:")
           for word, prob in next_words_probs:
              print(f"{word} {prob:.3f}")


if __name__ == '__main__':
    main()