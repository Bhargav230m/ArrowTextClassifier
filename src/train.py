import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score
from model import TextCNN
from tokenizer import Tokenizer
from tqdm import tqdm
import pickle
import sys
import os

class TextClassificationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        text = self.data.iloc[index]['example']
        label = self.data.iloc[index]['label']
        tokens = self.tokenizer.tokenize(text)
        encoded = [self.tokenizer.vocab[token] for token in tokens]
        encoded = encoded[:self.max_len] + [0] * (self.max_len - len(encoded))  # pad sequencess
        return torch.tensor(encoded), torch.tensor(label, dtype=torch.long)  # Ensure label is of type torch.lon

# Function to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, tokenizer):
    device = torch.device('cpu') # 'cuda' if torch.cuda.is_available else 'cpu' <-- Use this if you are using cuda torch lmao
    model = model.to(device)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        val_loss, precision, recall, f1 = evaluate_model(model, val_loader, criterion, device)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'arrow.pt')
            tokenizer.save_vocab("vocabulary.vocab")
            print("Saved best model.")
    
    # Save vocabulary
    tokenizer.save_vocab("vocabulary.vocab")
    
    print("Training complete.")

# Function to evaluate the model and see if it perfrm good..
def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss /= len(val_loader.dataset)
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')

    return val_loss, precision, recall, f1

def main():
    # Load dataset, should be parquet
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        df = pd.read_parquet(sys.argv[1])
    else:
        print("No dataset provided, defaulting to the default dataset")
        df = pd.read_parquet("classification/dataset.parquet")
    
    # Tokenize texts and build vocabulary..
    tokenizer = Tokenizer()
    tokenized_texts = [tokenizer.tokenize(text) for text in df['example']]
    tokenizer.build_vocab(tokenized_texts)
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])
    
    # Split dataset into train and validation sets.
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Define model hyperparameters
    VOCAB_SIZE = len(tokenizer.vocab) + 1
    EMBEDDING_DIM = 500
    NUM_FILTERS = 500
    FILTER_SIZES = [3, 4, 5]
    OUTPUT_DIM = len(label_encoder.classes_)
    DROPOUT = 0.2
    BATCH_SIZE = 128
    NUM_EPOCHS = 100
    WEIGHT_DECAY = 0.001

    # Saving hyperparameters
    print("Saving hyperparameters")
    with open("hyperparams.hparams", 'wb') as f:
            pickle.dump({
                "vocab_size": VOCAB_SIZE,
                "embedding_dim": EMBEDDING_DIM,
                "num_filters": NUM_FILTERS,
                "filter_sizes": FILTER_SIZES,
                "output_dim": OUTPUT_DIM,
                "dropout": DROPOUT,
                "batch_size": BATCH_SIZE,
                "num_epochs": NUM_EPOCHS,
                "labels": label_encoder.classes_,
                "weight_decay": WEIGHT_DECAY
            }, f)
    
    # Create model, criterion, optimizer
    model = TextCNN(VOCAB_SIZE, EMBEDDING_DIM, NUM_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=WEIGHT_DECAY)
    
    # Create datasets and data loaders
    train_dataset = TextClassificationDataset(train_df, tokenizer, max_len=200)
    val_dataset = TextClassificationDataset(val_df, tokenizer, max_len=200)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=NUM_EPOCHS, tokenizer=tokenizer)

main()
