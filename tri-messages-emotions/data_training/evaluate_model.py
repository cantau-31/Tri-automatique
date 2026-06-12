"""Evaluation du modele sauvegarde dans data_training/saved_model/."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import (
    DataCollatorWithPadding,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / ".hf_cache" / "datasets"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_DIR = PROJECT_ROOT / "data_training" / "saved_model"
METRICS_PATH = PROJECT_ROOT / "data_training" / "metrics.json"
CONFUSION_MATRIX_PATH = PROJECT_ROOT / "data_training" / "confusion_matrix.png"
EVAL_OUTPUT_DIR = PROJECT_ROOT / "data_training" / "eval_results"

DATASET_NAME = "dair-ai/emotion"
SEED = 42


def parse_args():
    """Recupere les options d'evaluation."""
    parser = argparse.ArgumentParser(description="Evaluer le modele d'emotions")
    parser.add_argument("--full-data", action="store_true", help="Evaluer tout le test set")
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=500,
        help="Nombre d'exemples test en mode rapide",
    )
    return parser.parse_args()


def tokenize_batch(batch, tokenizer):
    """Prepare les textes pour le modele."""
    return tokenizer(batch["text"], truncation=True, max_length=128)


def save_metrics(labels, predictions, label_names):
    """Sauvegarde les scores dans metrics.json."""
    report = classification_report(
        labels,
        predictions,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 4),
    }

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    return metrics


def save_confusion_matrix(labels, predictions, label_names):
    """Genere une image PNG de la matrice de confusion."""
    matrix = confusion_matrix(labels, predictions)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
    )
    plt.xlabel("Prediction")
    plt.ylabel("Vraie emotion")
    plt.title("Matrice de confusion")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()


def main():
    args = parse_args()

    if not (MODEL_DIR / "config.json").exists():
        raise FileNotFoundError(
            "Modele introuvable. Lancez d'abord : python data_training/train_model.py"
        )

    print("Chargement du dataset de test...")
    dataset = load_dataset(DATASET_NAME)
    label_names = dataset["test"].features["label"].names

    test_dataset = dataset["test"]
    if not args.full_data:
        sample_count = min(args.max_test_samples, len(test_dataset))
        test_dataset = test_dataset.shuffle(seed=SEED).select(range(sample_count))

    print("Chargement du modele sauvegarde...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)

    tokenized_test = test_dataset.map(
        lambda batch: tokenize_batch(batch, tokenizer),
        batched=True,
        remove_columns=["text"],
    )
    tokenized_test = tokenized_test.rename_column("label", "labels")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=str(EVAL_OUTPUT_DIR), report_to="none"),
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    print("Evaluation en cours...")
    output = trainer.predict(tokenized_test)
    predictions = np.argmax(output.predictions, axis=1)
    labels = output.label_ids

    metrics = save_metrics(labels, predictions, label_names)
    save_confusion_matrix(labels, predictions, label_names)

    print(f"Metrics sauvegardees dans : {METRICS_PATH}")
    print(f"Matrice de confusion sauvegardee dans : {CONFUSION_MATRIX_PATH}")
    print(metrics)


if __name__ == "__main__":
    main()
