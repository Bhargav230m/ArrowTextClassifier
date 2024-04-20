import torch
from model import TextCNN
import numpy as np
from tokenizer import Tokenizer
import pickle

def load_tokenizer(vocab_file):
    tokenizer = Tokenizer()
    with open(vocab_file, 'rb') as f:
        tokenizer.vocab = pickle.load(f)

    return tokenizer

def load_model(model_path, h_paramPath, device):
    # Load hyperparameters
    with open(h_paramPath, "rb") as f:
        hyperparameters = pickle.load(f)
        
    model = TextCNN(hyperparameters["vocab_size"], hyperparameters["embedding_dim"], hyperparameters["num_filters"], hyperparameters["filter_sizes"], hyperparameters["output_dim"], hyperparameters["dropout"])
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()

    return model

def load_hparams(hparams_path):
    with open(hparams_path, "rb") as f:
        hyperparameters = pickle.load(f)

        return hyperparameters["labels"]

def predict_label(text, model, tokenizer, class_names):
    tokens = tokenizer.tokenize(text)
    encoded = [tokenizer.vocab.get(token, 0) for token in tokens]
    encoded = encoded[:100] + [0] * (100 - len(encoded))  # pad sequence
    input_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        output = model(input_tensor)
    predicted_probs = torch.softmax(output, dim=1).squeeze().tolist()
    predicted_label_index = torch.argmax(output, dim=1).item()
    highest_prob_index = torch.argmax(output)
    highest_prob_class = class_names[highest_prob_index]

    return predicted_label_index, predicted_probs, highest_prob_class

def classifyText(vocab_file, model_path, hyperparametersPath, text, device, log=False):
    tokenizer = load_tokenizer(vocab_file)
    model = load_model(model_path, hyperparametersPath, device=device)
    class_names = load_hparams(hyperparametersPath)

    predicted_label_index, predicted_probability, highest_prob_class = predict_label(text, model, tokenizer, class_names)
    predicted_label = class_names[predicted_label_index]

    result = {}
    for label, prob in zip(class_names, predicted_probability):
        if log == True:
            print(f"{label}: {prob:.4f}" if label == predicted_label else f"{label}: {prob:.4f}")
        result[label] = prob

    if log == True:
        print(f"\nHighest probability class: {highest_prob_class}")
        
    result["highest_probability_class"] = highest_prob_class
    return result
