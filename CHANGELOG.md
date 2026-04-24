# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

---

## [1.1.0] - 2026-04-24

### Changed
- **Critical data correction**: replaced synthetic benchmark results with real BERT embedding measurements
  - Wikipedia corpus: 167× compression, 16% cosine similarity loss (was: templated "166-261×" on synthetic data)
  - CC-News corpus: 178× compression, 13% cosine similarity loss
- Updated abstract, README, CITATION.cff, and paper LaTeX to reflect real experimental results
- Affiliation updated to Avermex Research Division
- Repository renamed from `Chaotic-Attractor-Compression` to `Embedding-Manifold-Compression`
- License updated: paper/docs → CC-BY-4.0, code → AGPL-3.0-or-later
- Removed all PP25 internal naming references

### Added
- `generate_bert_embeddings.py`: script to reproduce real BERT embedding datasets from Wikipedia and CC-News
- Shannon (1948) reference added to CITATION.cff

---

## [1.0.0] - 2025-11-21

### Added
- Initial release of research paper and Rust implementation
- 9 compression methods: GZIP, Zstd, Int8+GZIP, Delta+GZIP, Polar Delta+GZIP, Delta+ANS, Delta+RLE+GZIP, Attractor(PCA)
- Attractor analysis tools: Grassberger-Procaccia correlation dimension (D₂), Lyapunov exponent (λ₁), Takens embedding
- Preprint published on Zenodo (DOI: 10.5281/zenodo.17680988)

### Known limitations (addressed in v1.1.0)
- Results were based on synthetic datasets, not real embeddings
- Claimed ratios (166-261×) were synthetic; real BERT results (167-178×) confirmed in v1.1.0
