import argparse
import os

os.environ.setdefault("USE_TF", "0")

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class LogDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        self.labels = [int(label) for label in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to the training CSV file")
    args = parser.parse_args()

    # 1. Load Data
    df = pd.read_csv(args.data)
    
    # Map our exact categories to integers
    label_mapping = {"Error": 0, "CONFIGURATION": 1, "SYSTEM": 2, "Noise": 3}
    df['label'] = df['class'].map(label_mapping)
    df = df.dropna(subset=['LogMessage', 'label']) # Drop unparseable rows
    if df.empty:
        raise ValueError("Training data contains no valid rows after label mapping.")

    # 2. Load DeBERTa xSmall (Fast for CPU training)
    model_name = "microsoft/deberta-v3-xsmall"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = LogDataset(df["LogMessage"], df["label"], tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3

    # 4. Train and Save
    print("Starting DeBERTa training...")
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in train_dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_dataloader), 1)
        print(f"Epoch {epoch + 1}/{num_epochs} - loss: {avg_loss:.4f}")
    
    # Save the final model weights to the outputs folder for download
    output_dir = "./outputs/final_model"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training complete. Model saved to {output_dir}")

if __name__ == "__main__":
    main()
