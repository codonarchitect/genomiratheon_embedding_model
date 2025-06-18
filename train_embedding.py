"""
Loading and preprocessing

Lowercasing and cleaning

Converting to embedding training format

Fine-tuning using Sentence Transformers

Evaluating Trained model

Saving the trained model

"""

from huggingface_hub import login
login("your huggingface token here")

import json
import os
import time
import random
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader, random_split
import numpy as np

# =====================
# Step 1: Load JSON Data
# =====================
def load_pairs(filepath):
    print("\n Loading and preprocessing...")
    time.sleep(0.5)
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# =============================
# Step 2: Preprocess and Lowercase
# =============================
def preprocess(data):
    print(" Lowercasing and cleaning...")
    time.sleep(0.5)
    processed = []
    for item in data:
        prompt = item['prompt'].strip().lower()
        response = item['response'].strip().lower()
        processed.append(InputExample(texts=[prompt, response]))
    return processed

# =============================
# Step 3: Train + Evaluate Model
# =============================
from sentence_transformers.util import cos_sim

def train_model(train_examples, model_name='all-mpnet-base-v2', save_path='genomiratheon_embedding_model', epochs=4):
    print(" Converting to embedding training format...")
    time.sleep(0.5)
    model = SentenceTransformer(model_name)

    # Split dataset
    test_size = max(1, int(0.2 * len(train_examples)))  # 20% for testing
    train_size = len(train_examples) - test_size
    train_set, test_set = random_split(train_examples, [train_size, test_size])

    train_dataloader = DataLoader(train_set, shuffle=True, batch_size=4)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # SKIP evaluator or use dummy cosine similarity evaluation
    print(" Fine-tuning using Sentence Transformers...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        evaluation_steps=None,
        show_progress_bar=True,
        output_path=save_path,
    )

    print(" Saving the trained model...")
    model.save(save_path)
    print(f" Model saved to: {save_path}")

    # Final Evaluation
    print("\n Running final evaluation on test set...")
    sim_scores = []
    sentences1 = [ex.texts[0] for ex in test_set]
    sentences2 = [ex.texts[1] for ex in test_set]
    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)
    cosine_scores = cos_sim(embeddings1, embeddings2)

    for i in range(len(sentences1)):
        sim_scores.append(float(cosine_scores[i][i]))

    print("\n Evaluation Results:")
    print(f" Average Cosine Similarity: {np.mean(sim_scores):.4f}")
    print(f" Min Similarity: {np.min(sim_scores):.4f}")
    print(f" Max Similarity: {np.max(sim_scores):.4f}")

    print("\n All tasks completed successfully!")


# =============================
# Main Execution
# =============================
if __name__ == "__main__":
    # Path to your 12-pair JSON file
    json_path = "genomiratheon benchark dataset/genomiratheon_benchmark.json"

    # Load and preprocess
    raw_data = load_pairs(json_path)
    train_data = preprocess(raw_data)

    # Train, evaluate, and save model
    train_model(train_data)

