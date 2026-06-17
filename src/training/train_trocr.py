import os
import torch
from torch.utils.data import Dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)
from PIL import Image
import pandas as pd

def load_fftca_trocr_model(model_name="microsoft/trocr-base-handwritten"):
    """
    Loads the TrOCR model and applies the Focused Fine-Tuning Causal Attention (FFTCA)
    logic from the paper:
    - Freezes the entire ViT Encoder.
    - Freezes the first 10 layers of the RoBERTa Decoder.
    - Unfreezes the last 2 layers of the Decoder and the Language Modeling Head.
    """
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    for param in model.encoder.parameters():
        param.requires_grad = False

    for i in range(10):
        for param in model.decoder.model.decoder.layers[i].parameters():
            param.requires_grad = False

    for i in range(10, 12):
        for param in model.decoder.model.decoder.layers[i].parameters():
            param.requires_grad = True

    for param in model.decoder.output_projection.parameters():
        param.requires_grad = True

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 64
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    return processor, model


class TrOCRDataset(Dataset):
    """
    Dataset loader for cropped handwritten words.
    Expects a DataFrame with 'file_name' and 'text' columns.
    """
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        image = Image.open(os.path.join(self.root_dir, file_name)).convert("RGB")

        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        labels = self.processor.tokenizer(
            text, 
            padding="max_length", 
            max_length=self.max_target_length
        ).input_ids

        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        return {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}


def train_model(data_dir, csv_file, output_dir="fftca_trocr_model"):
    print("Initializing FFTCA-TrOCR Model...")
    processor, model = load_fftca_trocr_model()

    print("Loading Dataset...")
    df = pd.read_csv(csv_file)
    train_df = df.sample(frac=0.9, random_state=42)
    val_df = df.drop(train_df.index)
    
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    train_dataset = TrOCRDataset(root_dir=data_dir, df=train_df, processor=processor)
    val_dataset = TrOCRDataset(root_dir=data_dir, df=val_df, processor=processor)

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        fp16=torch.cuda.is_available(), 
        output_dir=output_dir,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        num_train_epochs=20,  # As specified in the paper
        learning_rate=5e-5,
        save_total_limit=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )

    print("Starting Focused Fine-Tuning...")
    trainer.train()
    
    print(f"Training complete! Saving model to {output_dir}")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Directory containing cropped word images")
    parser.add_argument("--csv_file", type=str, help="CSV file mapping file_name to text")
    args = parser.parse_args()
    
    if args.data_dir and args.csv_file:
        train_model(args.data_dir, args.csv_file)
    else:
        print("Please provide --data_dir and --csv_file to start training.")
