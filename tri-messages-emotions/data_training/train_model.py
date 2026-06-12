"""Fine-tuning DistilBERT sur le dataset dair-ai/emotion.

Par defaut, le script utilise un petit sous-ensemble pour obtenir rapidement un
modele de demonstration. Pour entrainer sur tout le dataset, utilisez --full-data.
"""

import argparse
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / ".hf_cache" / "datasets"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from datasets import DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    DataCollatorWithPadding,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)


OUTPUT_DIR = PROJECT_ROOT / "data_training" / "saved_model"
RESULTS_DIR = PROJECT_ROOT / "data_training" / "results"
LOGS_DIR = PROJECT_ROOT / "data_training" / "logs"

DATASET_NAME = "dair-ai/emotion"
MODEL_NAME = "distilbert-base-uncased"
SEED = 42


def parse_args():
    """Recupere les options simples pour lancer un entrainement rapide ou complet."""
    parser = argparse.ArgumentParser(description="Entrainer DistilBERT sur dair-ai/emotion")
    parser.add_argument("--full-data", action="store_true", help="Utiliser tout le dataset")
    parser.add_argument("--epochs", type=float, default=1.0, help="Nombre d'epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Taille des batchs")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=800,
        help="Nombre d'exemples train en mode rapide",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=200,
        help="Nombre d'exemples validation en mode rapide",
    )
    return parser.parse_args()


def select_split(dataset, split_name, max_samples, use_full_data):
    """Selectionne une partie du dataset pour accelerer la demo locale."""
    split = dataset[split_name]

    if use_full_data:
        return split

    sample_count = min(max_samples, len(split))
    return split.shuffle(seed=SEED).select(range(sample_count))


def tokenize_batch(batch, tokenizer):
    """Convertit les textes en tokens compris par DistilBERT."""
    return tokenizer(batch["text"], truncation=True, max_length=128)


def compute_metrics(eval_prediction):
    """Calcule les scores principaux pendant l'evaluation."""
    predictions = np.argmax(eval_prediction.predictions, axis=1)
    labels = eval_prediction.label_ids

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
    }


def main():
    args = parse_args()
    set_seed(SEED)

    print("Chargement du dataset dair-ai/emotion...")
    dataset = load_dataset(DATASET_NAME)

    label_names = dataset["train"].features["label"].names
    id2label = {index: label for index, label in enumerate(label_names)}
    label2id = {label: index for index, label in enumerate(label_names)}

    selected_dataset = DatasetDict(
        {
            "train": select_split(
                dataset,
                "train",
                args.max_train_samples,
                args.full_data,
            ),
            "validation": select_split(
                dataset,
                "validation",
                args.max_eval_samples,
                args.full_data,
            ),
        }
    )

    print("Chargement de DistilBERT...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
    )

    tokenized_dataset = selected_dataset.map(
        lambda batch: tokenize_batch(batch, tokenizer),
        batched=True,
        remove_columns=["text"],
    )
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(RESULTS_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir=str(LOGS_DIR),
        logging_steps=20,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Debut de l'entrainement...")
    trainer.train()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Modele sauvegarde dans : {OUTPUT_DIR}")
    print("Vous pouvez maintenant lancer l'API avec : python -m uvicorn api.main:app --reload")


if __name__ == "__main__":
    main()
