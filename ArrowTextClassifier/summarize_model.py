from torchsummary import summary
import torch
from ArrowTextClassifier.model import TextCNN
import pickle
import sys

def load_model(model_path, hyperparameters, device):
    model = TextCNN(hyperparameters["vocab_size"], hyperparameters["embedding_dim"], hyperparameters["num_filters"], hyperparameters["filter_sizes"], hyperparameters["output_dim"], hyperparameters["dropout"])
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()
    return model

def load_vocabulary(vocabulary_path):
    with open(vocabulary_path, "rb") as f:
        return pickle.load(f)
    
def load_hparams(hparams_path):
    with open(hparams_path, "rb") as f:
        return pickle.load(f)


def summarize_model(model_path, hparams_path, vocab_path, device, modelSummary_write_path):
    hyperparameters = load_hparams(hparams_path)
    vocabulary = load_vocabulary(vocab_path) if vocab_path else None

    # Load model
    device = torch.device(device)
    model = load_model(model_path, hyperparameters, device)
    
    # Move the model to the device you gave
    model = model.to(device)

    if modelSummary_write_path is not None:
        with open("model_summary.txt", "w", encoding="utf-8") as f:
            sys.stdout = f
            summary(model, (200,))
            sys.stdout = sys.__stdout__
            dataToWrite = f"\n\nHyperparameters config:\n{hyperparameters}\n\nVocabulary:\n{vocabulary}"
            f.write(dataToWrite)
    else:
        summary(model, (200,))
        print(f"\n\nHyperparameters config:\n{hyperparameters}\n\nVocabulary:\n{vocabulary}")