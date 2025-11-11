# 🧠 Lightweight RAG: Alzheimer’s Gene Curation Assistant

This project implements a **tiny retrieval-augmented generation (RAG)** pipeline designed to summarize and cross-reference genomic findings from Alzheimer’s disease (AD) studies.  
The system demonstrates transparent retrieval, reranking, and grounded citation using a **local corpus** of 5 curated snippets from 2 key publications.

---

## 🧬 Context & Goal

This lightweight agent helps scientists query a small genomic knowledge base about Alzheimer’s disease.  
It retrieves relevant evidence snippets and generates short, citation-grounded answers.

**Core Goals**
1. Index a small, transparent local corpus (5 text snippets).  
2. Retrieve at the sentence level with local context (former + after sentence).  
3. Generate ≤8-sentence answers with inline `[S#]` citations.  
4. Evaluate retrieval and answer correctness.  

---

## ⚙️ Setup

Run the following on **Google Colab** or follow requirements.txt on a local CPU-only environment.

```bash
!pip install langchain==1.0.5 langchain-community chromadb sentence-transformers transformers umap-learn matplotlib
```


## 🧩 Model & Embedding Choices

| Component | Choice | Rationale |
|------------|---------|------------|
| **Embedding Model** | `NeuML/pubmedbert-base-embeddings` | Biomedical domain specific → gene & disease representations won't be too close. |
| **Vector DB** | `Chroma` (LangChain wrapper) | Lightweight local store; persistent on disk. |
| **Retrieval Granularity** | Sentence-level with ±1 context window | Mimics abstract-like paragraph continuity; avoids losing context. |
| **Normalization** | ❌ Not applied | Empirically good performance for PubMedBERT and when visualized. |
| **Top-k (dense)** | 5 (before reranking) | Balances recall vs compute. |
| **Cross-encoder Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Re-scores candidates by semantic relevance and select top 2. |
| **Distance Threshold** | Average of top-2 distance between 'APOE Alzheimer disease' and the database.| Requery or give up when average distance of retrieved is larger.|
| **LLM** | `google/flan-t5-base` | Within 1-min inference time for each question on CPU and better than its smaller variant, flexible for text2text. |
---

## 🔍 Retrieval Settings

- **Chunking**: sentence-level + previous + next sentence for context (adjust # of sentence per chunk in application) 
- **Top-k**: 5 candidates → reranked using cross encoder → top-2 used for answer. (Tunable, just for demonstration)
- **Feedback loop**: if average retrieved distance > distance threshold, re-query after gene-synonym replacement, or if maximum trials exceeded, give up retrieval and answer the question directly
- **Synonym tool**: CSV mapping (e.g., APOE → apolipoprotein E)
- **Potential Improvement**: Use other methods to decide if retrieval should be made, eg. average entropy across generated token weighted by self-attention

---

## 📚 Corpus Summary

| ID | Title | Origin |
|----|--------|--------|
| S1–S3 | *Identification of Genetic Heterogeneity of Alzheimer’s Disease across Age* | PMC6544706 |
| S4–S5 | *Genetic Heterogeneity of AD in Subjects with and without Hypertension* | PMC6836675 |

Total sentences indexed: **≈11 (5 snippets × 2 sentences each)**  
Each stored with metadata (`id`, `title`, `source`, `position`).

<!-- ---

## 🧮 Retrieval Pipeline

1. **Encode query → dense retrieval**  
2. **Rerank with cross-encoder**  
3. **Take top-3**  
4. **Compose grounded answer**  
5. **If similarity < threshold → synonym requery → repeat**

--- -->

## 🧠 Example Query & Output

**Q:** *How does APOE ε4’s contribution to Alzheimer’s risk differ between younger and older adults?*

**Retrieved snippets:** S1, S3  
**A:**  
APOE 4 risk is markedly age-dependent: in ADGC data, carrying one 4 allele raised risk 4.6 in ages 60–79 and 2.8 in ages 80; two 4 alleles raised risk 15 vs 3.6, respectively. Partitioned heritability also differs by age—chromosome 19 (harboring APOE) explains a substantially larger share in younger vs 1% in older, and APOE 4 explained 12–13% of phenotypic variation in younger vs 4–5% in older—evidence for a more polygenic architecture at older ages beyond APOE.[S1][S3]

---

The retrieval was correct but APOE 4 missed a ε and could emphasize the heritability partition more clearly.

## 📊 Evaluation

### 🔹 Retrieval Quality

| Query | Smallest k to retrieve enough (without cross-encoding) | Smallest k-final to retrieve enough |
|--------|----------------|------------------------------|
| Q1 (APOE age effect) | 3 | 2|
| Q2 (Younger-onset genes) | 5 | 3 |
| Q3 (Vascular HTN pathways) | 3 | 3 |
| Q4 (Average Life Expectancy) | x | x |

---
Q3 triggers requery once. At Q2, the retriever would miss the correct snippets if using all-MiniLM-L6-v2 for embeddings or not using cross-encoder.

### 🔹 Answer Accuracy
If using "NeuML/pubmedbert-base-embeddings" and cross encoder, (correctness graded by GPT).
| Query | Correctness | Notes |
|--------|----------------|-------|
| Q1 | 9/10 | Quantitative APOE effects recovered |
| Q2 | 7/10 | Correct loci (BIN1, PICALM, MS4A4E) |
| Q3 | 5/10 | Correctly cites BBB & mitochondrial pathways |
| Q4 | 1/10 | Hallucination but answer is relevant|
---

## 🌈 Embedding Space Visualization

Use UMAP (n_neighbors=8, min_dist=0.1) to project embeddings to 2-D.

When normalized,
![UMAP of AD Snippet Embeddings_normalized]('normalized.png.')

When not normalized,
![UMAP of AD Snippet Embeddings]('Pubmed.png.')

Although they both split the 5 snippets quite well, note that S1-3 and S4-5 are respectively from two papers.

---

## 💰 Cost & Efficiency

- **Compute**: CPU-only, < 1 min indexing, < 6GB RAM
- **Models**: All open source and can be downloaded(PubMedBERT, MiniLM cross-encoder)  
- **Inference cost**: $0 (local execution), < 1 min

---

## 🧩 Future Extensions

- Consider **embedding normalization** of another model and **hybrid BM25+dense retrieval**  
- Evaluate LLM routing confidence vs retrieval threshold  
- Expand corpus with more ADVP/NIAGADS publications  
- Integrate **synonym feedback loop** and **query reformulation logging**

---

### 📄 References
1. Jansen et al. (2019) *Identification of Genetic Heterogeneity of Alzheimer’s Disease across Age*. PMC6544706  
2. Wang et al. (2019) *Genetic Heterogeneity of Alzheimer’s Disease in Subjects with and without Hypertension*. PMC6836675

