# Embedding Manifold Compression

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17680988.svg)](https://doi.org/10.5281/zenodo.17680988)
[![License](https://img.shields.io/badge/Paper-CC--BY--4.0-green.svg)](https://creativecommons.org/licenses/by/4.0/)
[![License](https://img.shields.io/badge/Code-AGPL--3.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

**Exploiting Low-Dimensional Manifold Structure for Archival Compression of High-Dimensional ML Embeddings: A Dynamical Systems Approach**

**Author**: Francisco Molina-Burgos, Avermex Research Division  
**ORCID**: [0009-0008-6093-8267](https://orcid.org/0009-0008-6093-8267)  
**Contact**: fmolina@avermex.com  
**Version**: 1.1.0 (April 2026)

---

## Abstract

Large-scale machine learning systems generate billions of high-dimensional embedding vectors (768D for BERT-base, up to 12,288D for GPT-4), creating multi-terabyte storage requirements for archival and cold storage applications. While standard compression (GZIP, Zstd) achieves only ~1.1× on floating-point embeddings and Product Quantization trades accuracy for ratio, we propose exploiting the *intrinsic low-dimensional manifold structure* of embedding sequences for archival compression.

Using tools from dynamical systems theory — correlation dimension (D₂) and Lyapunov exponents (λ₁) — we characterize the geometric structure of BERT embeddings from Wikipedia and CC-News corpora. Our PCA-based differential encoding achieves **167–178× compression** with 13–16% cosine similarity loss, suitable for applications where storage cost dominates over retrieval precision (regulatory archives, historical corpus preservation, embedding forensics).

Critically, we demonstrate why classical entropy coders (GZIP/LZ77) fail on low-entropy floating-point deltas, achieving only 6.3% of theoretical efficiency.

**Key Results** (real BERT embeddings — Wikipedia + CC-News corpora):
- Wikipedia: **167× compression**, 16% cosine similarity loss
- CC-News: **178× compression**, 13% cosine similarity loss
- GZIP baseline: 1.1× (confirms the 6.3% efficiency bound)
- Correlation dimension D₂ characterizes manifold intrinsic dimensionality
- Positive Lyapunov exponent λ₁ confirms chaotic dynamics in embedding sequences

---

## Publication Status

**Current Status** (April 2026):
- ✅ **Preprint v1.0**: Zenodo DOI: [10.5281/zenodo.17680988](https://doi.org/10.5281/zenodo.17680988)
- ✅ **v1.1 Correction**: Real BERT data replacing synthetic benchmarks
- ⏳ **arXiv**: Pending submission (cs.LG / cs.IT)
- 📋 **Peer Review**: Targeting NeurIPS 2026 (~May 2026 deadline) or TMLR (rolling)

**Venues under consideration**:
- NeurIPS 2026 — ~May 2026 deadline, highest visibility
- TMLR (Transactions on Machine Learning Research) — rolling, open access
- Entropy (MDPI) — rolling, information theory focus

---

## Repository Structure

```
Embedding-Manifold-Compression/
├── README.md
├── LICENSE
├── CITATION.cff
├── CHANGELOG.md
├── PUBLICATION_STRATEGY.md
├── paper/
│   ├── paper_arxiv.tex          ← LaTeX source (corrected, real data)
│   ├── paper_arxiv.pdf          ← Compiled PDF
│   └── figures/
├── code/
│   ├── src/                     ← Rust implementation (9 compression methods)
│   ├── Cargo.toml
│   └── results/                 ← Experimental results (real BERT)
├── arxiv_submission/
│   └── figures/
└── supplementary/
```

---

## Quick Start

### Requirements
- Rust 1.75+
- Python 3.9+ (for BERT embedding generation)
- 4 GB RAM minimum

### Installation

```bash
git clone https://github.com/Yatrogenesis/Embedding-Manifold-Compression.git
cd Embedding-Manifold-Compression/code
cargo build --release
```

### Generate Real BERT Embeddings

```bash
pip install transformers torch datasets
python generate_bert_embeddings.py --corpus wikipedia --output data/wikipedia_embeddings.bin
python generate_bert_embeddings.py --corpus cc_news --output data/ccnews_embeddings.bin
```

### Run Compression Experiments

```bash
# Full experiment (9 compression methods)
cargo run --release --bin compression-experiment -- --input data/wikipedia_embeddings.bin

# Attractor analysis
cargo run --release --bin analyze_attractor -- --input data/wikipedia_embeddings.bin

# Delta diagnostics
cargo run --release --bin analyze_deltas -- --input data/wikipedia_embeddings.bin
```

### Expected Output (Wikipedia corpus)

```
Dataset: Wikipedia BERT embeddings (768D, real)
  GZIP:              1.1x  (0.0% loss)
  Zstd:              1.2x  (0.0% loss)
  Int8+GZIP:         9.1x  (23.1% loss)
  Delta+GZIP:        1.1x  (0.0% loss)
  Attractor(PCA-k):  167x  (16.0% loss)  ← primary result

Attractor Properties:
  D₂ = 1.21  (correlation dimension)
  λ₁ = +0.12 (Lyapunov exponent — positive → chaotic)
```

---

## Compression Methods

| Method | Wikipedia | CC-News | Loss | Notes |
|--------|-----------|---------|------|-------|
| GZIP | 1.1× | 1.1× | 0% | Baseline — fails on float deltas |
| Zstd | 1.2× | 1.3× | 0% | Lossless |
| Int8+GZIP | 9.1× | 9.3× | ~23% | Best lossless-adjacent |
| Delta+GZIP | 1.1× | 1.1× | 0% | Confirms GZIP inefficiency |
| **Attractor(PCA)** | **167×** | **178×** | **13-16%** | **Primary contribution** |

---

## Citation

### BibTeX

```bibtex
@misc{molina2026embedding,
  title={Exploiting Low-Dimensional Manifold Structure for Archival Compression
         of High-Dimensional ML Embeddings: A Dynamical Systems Approach},
  author={Molina-Burgos, Francisco},
  year={2026},
  note={Preprint v1.1, real BERT validation},
  howpublished={Zenodo},
  doi={10.5281/zenodo.17680988},
  url={https://github.com/Yatrogenesis/Embedding-Manifold-Compression}
}
```

### APA

Molina-Burgos, F. (2026). *Exploiting Low-Dimensional Manifold Structure for Archival Compression of High-Dimensional ML Embeddings* [Preprint v1.1]. Zenodo. https://doi.org/10.5281/zenodo.17680988

---

## Key Contributions

1. **Real-data validation**: First measurement of manifold structure (D₂, λ₁) in production BERT embeddings (Wikipedia, CC-News)
2. **Root cause analysis**: Formal proof that GZIP achieves only 6.3% efficiency on low-entropy float deltas
3. **PCA manifold compression**: 167–178× archival compression with bounded 13-16% cosine similarity loss
4. **Theoretical framework**: Connection between Lyapunov exponents and compression potential (rate-distortion bound)
5. **Open-source Rust implementation**: 9 compression methods, reproducible benchmark

---

## License

- **Paper / Documentation**: [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Code**: [AGPL-3.0-or-later](https://www.gnu.org/licenses/agpl-3.0)
- Commercial licensing: fmolina@avermex.com

---

## Author

**Francisco Molina-Burgos**  
Avermex Research Division — Mérida, Yucatán, México  
ORCID: [0009-0008-6093-8267](https://orcid.org/0009-0008-6093-8267)  
GitHub: [@Yatrogenesis](https://github.com/Yatrogenesis)

---

**Last Updated**: April 2026  
**Repository**: https://github.com/Yatrogenesis/Embedding-Manifold-Compression  
**DOI**: [10.5281/zenodo.17680988](https://doi.org/10.5281/zenodo.17680988)
