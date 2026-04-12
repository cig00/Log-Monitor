import argparse
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import os

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

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df[['LogMessage', 'label']])
    
    # 2. Load DeBERTa xSmall (Fast for CPU training)
    model_name = "microsoft/deberta-v3-xsmall"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["LogMessage"], padding="max_length", truncation=True, max_length=128)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

    # 3. Setup Trainer
    # Azure automatically saves anything in the "./outputs" folder to the cloud blob
    training_args = TrainingArguments(
        output_dir="./outputs", 
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_strategy="no", # Save only at the end to save space
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )

    # 4. Train and Save
    print("Starting DeBERTa training...")
    trainer.train()
    
    # Save the final model weights to the outputs folder for download
    trainer.save_model("./outputs/final_model")
    tokenizer.save_pretrained("./outputs/final_model")
    print("Training complete. Model saved to ./outputs/final_model")

if __name__ == "__main__":
    main()