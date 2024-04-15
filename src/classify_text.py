import torch
from model import TextCNN
from tokenizer import Tokenizer
import pickle

def load_tokenizer(vocab_file):
    tokenizer = Tokenizer()
    with open(vocab_file, 'rb') as f:
        tokenizer.vocab = pickle.load(f)
    return tokenizer

def load_model(model_path, h_paramPath):
    # Load hyperparameters
    with open(h_paramPath, "rb") as f:
        hyperparameters = pickle.load(f)

    model = TextCNN(hyperparameters["vocab_size"], hyperparameters["embedding_dim"], hyperparameters["num_filters"], hyperparameters["filter_sizes"], hyperparameters["output_dim"], hyperparameters["dropout"])
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_label(text, model, tokenizer):
    tokens = tokenizer.tokenize(text)
    encoded = [tokenizer.vocab.get(token, 0) for token in tokens]
    encoded = encoded[:100] + [0] * (100 - len(encoded))  # pad sequence
    input_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        output = model(input_tensor)
    predicted_label_index = torch.argmax(output, dim=1).item()
    predicted_probability = torch.softmax(output, dim=1).squeeze().tolist()
    return predicted_label_index, predicted_probability

vocab_file = input("Enter vocabulary file path (Ends with .vocab): ")
tokenizer = load_tokenizer(vocab_file)

model_path = input("Enter trained model path (Ends with .pt): ")
hyperparametersPath = input("Enter hyperparameter path (Ends with .hparams): ")
model = load_model(model_path, hyperparametersPath)

with open(hyperparametersPath, "rb") as f:
    hyperparameters = pickle.load(f)
    class_names = hyperparameters["labels"]

while True:
    text = input("Enter text to classify (type 'quit' to exit): ")
    if text.lower() == "quit":
        break
    predicted_label_index, predicted_probability = predict_label(text, model, tokenizer)
    predicted_label = class_names[predicted_label_index]
    print("Predicted labels with probabilities:")
    for label, prob in zip(class_names, predicted_probability):
        print(f"{label}: {prob:.4f}" if label == predicted_label else f"{label}: {prob:.4f}")
