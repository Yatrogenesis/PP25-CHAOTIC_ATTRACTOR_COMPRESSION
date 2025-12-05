#!/usr/bin/env python3
"""
Generate REAL BERT embeddings from actual Wikipedia and news articles
Author: Francisco Molina Burgos (ORCID: 0009-0008-6093-8267)
Date: 2025-12-05
Version: VALIDATION - Real data sources

This script replaces the templated sentences with ACTUAL text from:
- Wikipedia: Real articles from HuggingFace datasets
- News: Real news articles from CC-News dataset
"""

import numpy as np
import json
from pathlib import Path
import sys

print("🔬 Generating REAL BERT Embeddings - Validation Script")
print("=" * 70)

# Check and install dependencies
try:
    from transformers import BertTokenizer, BertModel
    import torch
    print("✅ transformers installed")
except ImportError:
    print("📥 Installing transformers...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "torch"])
    from transformers import BertTokenizer, BertModel
    import torch

try:
    from datasets import load_dataset
    print("✅ datasets installed")
except ImportError:
    print("📥 Installing datasets...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
    from datasets import load_dataset

def get_real_wikipedia_sentences(n_samples=2000):
    """Extract real sentences from Wikipedia dataset"""
    print("📥 Loading Wikipedia dataset (this may take a few minutes)...")

    # Load Wikipedia dataset
    # Using the simpler 'wikimedia/wikipedia' dataset which has better access
    try:
        dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

        sentences = []
        for i, article in enumerate(dataset):
            if len(sentences) >= n_samples:
                break

            # Extract text from article
            text = article.get('text', '')
            if not text or len(text) < 50:
                continue

            # Split into sentences (simple approach)
            text_sentences = text.split('. ')
            for sent in text_sentences:
                if len(sent) > 30 and len(sent) < 500:  # Filter reasonable length
                    sentences.append(sent.strip())
                    if len(sentences) >= n_samples:
                        break

            if i % 100 == 0:
                print(f"   Processed {i} articles, collected {len(sentences)} sentences...")

        print(f"✅ Collected {len(sentences)} real Wikipedia sentences")
        return sentences[:n_samples]

    except Exception as e:
        print(f"⚠️  Wikipedia dataset failed: {e}")
        print("⚠️  Falling back to alternative Wikipedia source...")

        # Fallback: Use simple Wikipedia
        dataset = load_dataset("wikipedia", "20220301.simple", split="train[:200]")
        sentences = []
        for article in dataset:
            text = article.get('text', '')
            text_sentences = text.split('. ')
            for sent in text_sentences:
                if len(sent) > 30 and len(sent) < 500:
                    sentences.append(sent.strip())
                    if len(sentences) >= n_samples:
                        break
            if len(sentences) >= n_samples:
                break

        print(f"✅ Collected {len(sentences)} real Wikipedia sentences (simple)")
        return sentences[:n_samples]

def get_real_news_articles(n_samples=2000):
    """Extract real sentences from news articles"""
    print("📥 Loading CC-News dataset (this may take a few minutes)...")

    try:
        # CC-News is a large dataset, use streaming
        dataset = load_dataset("cc_news", split="train", streaming=True)

        sentences = []
        for i, article in enumerate(dataset):
            if len(sentences) >= n_samples:
                break

            # Extract text from article
            text = article.get('text', '')
            if not text or len(text) < 50:
                continue

            # Split into sentences
            text_sentences = text.split('. ')
            for sent in text_sentences:
                if len(sent) > 30 and len(sent) < 500:
                    sentences.append(sent.strip())
                    if len(sentences) >= n_samples:
                        break

            if i % 100 == 0:
                print(f"   Processed {i} articles, collected {len(sentences)} sentences...")

        print(f"✅ Collected {len(sentences)} real news sentences")
        return sentences[:n_samples]

    except Exception as e:
        print(f"⚠️  CC-News failed: {e}")
        print("⚠️  Falling back to alternative news source...")

        # Fallback: Use AG News
        dataset = load_dataset("ag_news", split="train[:2000]")
        sentences = [item['text'] for item in dataset]
        print(f"✅ Collected {len(sentences)} real news sentences (AG News)")
        return sentences

def compute_consecutive_similarity(embeddings):
    """Compute average cosine similarity between consecutive vectors"""
    n = len(embeddings)
    sims = []
    for i in range(n-1):
        sim = np.dot(embeddings[i], embeddings[i+1])
        sim /= (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1]))
        sims.append(sim)
    return np.mean(sims)

def main():
    print("\n🔧 Loading BERT-base-uncased model (768D)...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.eval()

    def get_bert_embedding(text):
        """Get [CLS] token embedding from BERT"""
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        # Use [CLS] token (first token) as sentence embedding
        return outputs.last_hidden_state[0, 0, :].numpy()

    print("✅ BERT-base-uncased loaded (768D embeddings)")

    # Generate REAL datasets
    datasets = {
        "wikipedia_2k_REAL": get_real_wikipedia_sentences(2000),
        "news_temporal_2k_REAL": get_real_news_articles(2000),
    }

    output_dir = Path("data/real_embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, sentences in datasets.items():
        print(f"\n{'='*70}")
        print(f"📊 Dataset: {name}")
        print(f"{'='*70}")
        print(f"Sentences: {len(sentences)}")
        print(f"Sample sentence: {sentences[0][:100]}...")

        # Generate embeddings
        print("🔄 Generating BERT embeddings (this will take several minutes)...")
        embeddings = []
        for i, sentence in enumerate(sentences):
            emb = get_bert_embedding(sentence)
            embeddings.append(emb)

            if (i + 1) % 100 == 0:
                print(f"   Progress: {i+1}/{len(sentences)} ({100*(i+1)/len(sentences):.1f}%)")

        embeddings = np.array(embeddings)

        print(f"✅ Embeddings shape: {embeddings.shape}")
        print(f"   Dimension: {embeddings.shape[1]}D")
        print(f"   Count: {embeddings.shape[0]} vectors")

        # Compute consecutive similarity
        consec_sim = compute_consecutive_similarity(embeddings)
        print(f"🔑 Consecutive Similarity: {consec_sim:.4f}")

        # Save as numpy (for analysis)
        np.save(output_dir / f"{name}.npy", embeddings)
        print(f"💾 Saved: {output_dir / name}.npy")

        # Save as JSON (for Rust code)
        vectors_json = embeddings.tolist()
        with open(output_dir / f"{name}.json", 'w') as f:
            json.dump({
                "name": name,
                "dimension": embeddings.shape[1],
                "count": embeddings.shape[0],
                "consecutive_similarity": float(consec_sim),
                "data_source": "REAL - HuggingFace datasets (Wikipedia/CC-News)",
                "vectors": vectors_json
            }, f)
        print(f"💾 Saved: {output_dir / name}.json")

    print("\n" + "="*70)
    print("✅ DONE - Real BERT embeddings generated from actual Wikipedia and news")
    print(f"📂 Output directory: {output_dir.absolute()}")
    print("\nNext: Update Rust code to use these REAL datasets and re-run validation")

if __name__ == "__main__":
    main()
