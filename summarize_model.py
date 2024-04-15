from torchsummary import summary
import torch
from model import TextCNN
import pickle
import sys

def load_model(model_path, hyperparameters):
    model = TextCNN(hyperparameters["vocab_size"], hyperparameters["embedding_dim"], hyperparameters["num_filters"], hyperparameters["filter_sizes"], hyperparameters["output_dim"], hyperparameters["dropout"])
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def load_vocabulary(vocabulary_path):
    with open(vocabulary_path, "rb") as f:
        return pickle.load(f)
    
def load_hparams(hparams_path):
    with open(hparams_path, "rb") as f:
        return pickle.load(f)


hyperparameters = load_hparams(input("Enter .hparams path: "))
vocabulary = load_vocabulary(input("Enter .vocab path: "))

# Load model
model_path = input("Please specify the model path: ")  # Path to your trained model
model = load_model(model_path, hyperparameters)

# Display model summary
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

sys.stdout = open("model_summary.txt", "w", encoding="utf-8")
summary(model, (200,))
sys.stdout.close()

with open("model_summary.txt", "a", encoding="utf-8") as f:
    dataToWrite = f"\n\nHyperparameters config:\n{hyperparameters}\n\nVocabulary:\n{vocabulary}"
    f.write(dataToWrite)