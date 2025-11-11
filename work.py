import json
from pathlib import Path
from typing import List, Tuple
import csv
from collections import defaultdict
import re

import numpy as np

import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab')

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline

from sentence_transformers import CrossEncoder

# --------------------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------------------

CHROMA_DIR = "chroma_db"
CHROMA_DIR2 = "chroma_db2"
CHROMA_DIR3 = "chroma_db3"
CORPUS_PATH = "corpus.jsonl"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_MODEL_NAME2 = "NeuML/pubmedbert-base-embeddings"
GEN_MODEL_NAME = "google/flan-t5-base" 

# --------------------------------------------------------------------------------------
# 1. Corpus loading & vectorstore (Chroma)
# --------------------------------------------------------------------------------------

class NormalizedEmbeddings(SentenceTransformerEmbeddings):
    def embed_query(self, text):
        v = super().embed_query(text)
        return (v / np.linalg.norm(v)).tolist()

    def embed_documents(self, texts):
        vecs = super().embed_documents(texts)
        arr = np.array(vecs)
        arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
        return arr.tolist()

def load_corpus_sentence_level(path: str):
    docs = []
    corpus_index = {}
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            orig_id = obj["id"]
            text = obj["text"]
            cite = obj["id"]+'_cite'

            # split into sentences
            sentences = sent_tokenize(text)

            corpus_index[orig_id] = {
                "sentences": sentences,
                "cite": cite,
                "doc_idx": int(orig_id[1]), 
            }

            for j, sent in enumerate(sentences):
                if not sent.strip():
                    continue
                # create a unique ID per sentence
                sent_id = f"{orig_id}_sent{j}"
                docs.append(
                    Document(
                        page_content=sent,
                        metadata={
                            "source_id": orig_id, 
                            "sent_id": sent_id,
                            "sent_idx": j,
                        },
                    )
                )
    return docs, corpus_index

def build_vectorstore(
    docs,
    persist_dir: str = CHROMA_DIR,
    model_name = EMBED_MODEL_NAME,
    normalize=False
) -> Chroma:
    print(f"Creating embeddings with {model_name}...")
    if normalize:
      embedding_fn = NormalizedEmbeddings(
          model_name=model_name
      )
    else:
      embedding_fn = SentenceTransformerEmbeddings(
        model_name=model_name
      )
    print(f"Building Chroma index at {persist_dir}...")
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedding_fn,
        persist_directory=persist_dir,
    )
    vectordb.persist()
    return vectordb

def load_vectorstore(persist_dir: str = CHROMA_DIR,model_name=EMBED_MODEL_NAME,normalized=False) -> Chroma:
    if normalized:
      embedding_fn = NormalizedEmbeddings(
          model_name=model_name
      )
    else:
      embedding_fn = SentenceTransformerEmbeddings(
        model_name=model_name
    )
    vectordb = Chroma(
        embedding_function=embedding_fn,
        persist_directory=persist_dir,
    )
    return vectordb

docs,corpus_index = load_corpus_sentence_level("corpus.jsonl")
vectordb = load_vectorstore(persist_dir = 'chroma_db_normalized')

# --------------------------------------------------------------------------------------
# 2. Visualization (UMAP)
# --------------------------------------------------------------------------------------
# import umap.umap_ as umap
# import matplotlib.pyplot as plt
# docs1 = vectordb.get(include=["documents", "metadatas", "embeddings"])
# X = np.array(docs1["embeddings"])
# texts = docs1["documents"]
# titles = [m.get("source_id", "") for m in docs1["metadatas"]]
# titles2 = [m.get("sent_id", "") for m in docs1["metadatas"]]
# reducer = umap.UMAP(n_neighbors=8, min_dist=0.1, metric="cosine", random_state=42)
# embedding_2d = reducer.fit_transform(X)
# colors = [hash(t) for t in titles]  # rough color per paper
# plt.figure(figsize=(6,5))
# plt.scatter(embedding_2d[:,0], embedding_2d[:,1], c=colors, cmap="tab10", s=50)
# for i, txt in enumerate(range(len(texts))):
#     plt.text(embedding_2d[i,0]+0.02, embedding_2d[i,1], titles2[i], fontsize=8)
# plt.title("UMAP: AD corpus by publication")
# plt.show()

retrieved = vectordb.similarity_search_with_score(
        'APOE Alzheimer’s disease',
        k=2,)
threshold = sum([t[1] for t in retrieved])/2

with open('queries.jsonl', 'r') as json_file:
    json_list = list(json_file)
queries = []
for json_str in json_list:
    result = json.loads(json_str)
    queries.append(result['q'])

# --------------------------------------------------------------------------------------
# 3. LLM setup
# --------------------------------------------------------------------------------------

def create_llm(model_name: str = GEN_MODEL_NAME) -> HuggingFacePipeline:
    print(f"Loading generation model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    gen_pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        min_new_tokens=10,    # avoid 1–2 word answers
        do_sample=True,       # enable sampling; otherwise it's greedy
        temperature=0.7,      # higher = more diverse, lower = more deterministic
        top_p=0.9,            # nucleus sampling cutoff
        repetition_penalty=1.1,
    )

    llm = HuggingFacePipeline(pipeline=gen_pipe)
    return llm


ANSWER_PROMPT = PromptTemplate.from_template(
    """
Use evidence to answer a question in at most 8 sentences and cite the evidence inline, eg. [S1].

Example 1:
Question: Who is James?

Evidence: 
source_id=S1: James lives in US.James is a scientist.
source_id=S3: James is a scientist.

Answer: 
James is a scientist [S3], and he lives in US [S1].

Example 2:
Question: What is AD?

Evidence: 
source_id=S1: AD is Alzheimer’s disease.

Answer: 
AD is Alzheimer’s disease [S1].

Problem:
Question: {question}

Evidence: 
{evidence}

Answer:
"""
)

ANSWER_PROMPT2 = PromptTemplate.from_template(
    """
Question: {question}

Answer:
"""
)


def expand_with_context(
    retrieved,        # List[(Document, distance)]
    corpus_index,     # output of build_corpus_index
    window: int = 1   # 1 sentence before and after
):
    """
    Returns a list of segments:
      [
        {
          "source_id": ...,
          "start_idx": ...,
          "end_idx": ...,
          "sentences": [...],
          "best_distance": float,
        },
        ...
      ]
    The segments are ordered by:
      1) best_distance ascending (most similar first), then
      2) doc_idx, then
      3) start_idx
    """
    # 1) Collect hit sentence indices & distances per source
    per_source = defaultdict(list)  # source_id -> [(sent_idx, distance), ...]
    for doc, dist in retrieved:
        source_id = doc.metadata["source_id"]
        sent_idx = doc.metadata["sent_idx"]
        per_source[source_id].append((sent_idx, dist))

    segments = []

    for source_id, idx_dist_list in per_source.items():
        sent_list = corpus_index[source_id]["sentences"]
        n = len(sent_list)

        # For each hit sentence, create a [start, end] window
        ranges = []
        best_dist_for_doc = float("inf")

        for sent_idx, dist in idx_dist_list:
            best_dist_for_doc = min(best_dist_for_doc, dist)
            start = max(0, sent_idx - window)
            end = min(n - 1, sent_idx + window)
            ranges.append((start, end))

        # Merge overlapping windows in this doc
        ranges.sort()
        merged = []
        cur_start, cur_end = ranges[0]
        for s, e in ranges[1:]:
            if s <= cur_end + 1:  # overlapping or touching
                cur_end = max(cur_end, e)
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = s, e
        merged.append((cur_start, cur_end))

        # Build segments for this doc
        for start, end in merged:
            seg_sents = sent_list[start:end+1]
            segments.append({
                "source_id": source_id,
                "start_idx": start,
                "end_idx": end,
                "sentences": seg_sents,
                "best_distance": best_dist_for_doc,
            })

    # 2) Sort segments across docs:
    #    first by best_distance, then by doc order in corpus, then by start_idx
    def sort_key(seg):
        source_id = seg["source_id"]
        doc_idx = corpus_index[source_id]["doc_idx"]
        return (seg["best_distance"], doc_idx, seg["start_idx"])

    segments.sort(key=sort_key)
    return segments

def build_evidence_from_segments(segments):
    """
    Build the evidence string and also track which source_ids we used.
    Returns:
      evidence_str: str to put into the prompt
      used_source_ids: set of source_ids
    """
    lines = []
    used_source_ids = set()

    for i, seg in enumerate(segments, start=1):
        sid = f"S{i}"
        source_id = seg["source_id"]
        used_source_ids.add(source_id)

        text_block = " ".join(seg["sentences"]).replace("\n", " ")
        lines.append(
            f"source_id={source_id}: {text_block}"
        )

    evidence_str = "\n".join(lines)
    return evidence_str, used_source_ids

def make_rag_chain(llm: HuggingFacePipeline):
    parser = StrOutputParser()
    chain = ANSWER_PROMPT | llm | parser
    return chain

# --------------------------------------------------------------------------------------
# 4. End-to-end answer function
# --------------------------------------------------------------------------------------

def answer_question(
    vectordb: Chroma,
    llm: HuggingFacePipeline,
    question: str,
    k: int = 5,
    k_final: int = 3,
    cross_encoder=None,
    gene_tool=None,
    max_dist=0.8,
    max_retries=2,
) -> str:
    question, retrieved, lowest_dist = retrieve_with_feedback(question, vectordb, gene_tool, k=k, max_dist=max_dist, max_retries=max_retries)
    if not retrieved:
       rag_chain = ANSWER_PROMPT2 | llm | StrOutputParser()
       answer = rag_chain.invoke(
          {"question": question}
       )
       formatted = f"Q: {question}\nA: {answer.strip()}\n---"
       return formatted

    if cross_encoder:
        ce_scores = cross_encoder.predict([(question, t[0].page_content) for t in retrieved])
        ce_scores = np.asarray(ce_scores, dtype=float)
        keep = np.argsort(-ce_scores)[:k_final]
        retrieved = [retrieved[i] for i in keep]

    segments = expand_with_context(retrieved, corpus_index, window=1)
    evidence_str, used_source_ids = build_evidence_from_segments(segments)

    print(evidence_str)

    rag_chain = make_rag_chain(llm)
    answer = rag_chain.invoke(
        {"question": question, "evidence": evidence_str}
    )
    citation = '['+"][ ".join(sorted(used_source_ids))+']'
    answer += citation
    formatted = f"Q: {question}\nA: {answer.strip()} \n---"
    return formatted

class GeneSynonymTool:
    def __init__(self, csv_path: str):
        self.canonical_to_syns = defaultdict(list)
        self.name_to_canonical = {}

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                canon = row["canonical"].strip()
                syn = row["synonym"].strip()
                self.canonical_to_syns[canon].append(syn)
                # map canonical and synonym to canonical (case-insensitive)
                self.name_to_canonical[canon.upper()] = canon
                self.name_to_canonical[syn.upper()] = canon

    def get_replacement(self, token: str) -> str | None:
        """
        If token matches a known gene name or synonym, return
        a preferred synonym to replace it with; else None.
        For example, 'APOE' -> 'apolipoprotein E'.
        """
        canon = self.name_to_canonical.get(token.upper())
        if canon is None:
            return None
        syns = self.canonical_to_syns.get(canon, [])
        if not syns:
            return None
        # heuristics: prefer longest synonym (full name)
        return max(syns, key=len)

WORD_RE = re.compile(r"(\w+|\W+)")  # captures words *and* non-word chunks

def replace_query_tokens_with_synonyms(query: str, gene_tool: GeneSynonymTool) -> str:
    parts = WORD_RE.findall(query)
    replaced_any = False
    new_parts = []

    for part in parts:
        if part.isalnum():  # word-like
            repl = gene_tool.get_replacement(part)
            if repl:
                new_parts.append(repl)
                replaced_any = True
            else:
                new_parts.append(part)
        else:
            # punctuation/space
            new_parts.append(part)

    if not replaced_any:
        return query
    return "".join(new_parts)

def retrieve_with_feedback(question, vectordb, gene_tool, k=5, max_dist=0.85, max_retries=2):
    """Retrieve docs, and if average_dist > max_dist, try re-querying."""
    attempt = 0
    retrieved = None

    for attempt in range(max_retries):
        retrieved = vectordb.similarity_search_with_score(
            question,
            k=k,
        )
        dists = [t[1] for t in retrieved]
        avg_dist = sum(dists)/len(dists)

        if avg_dist <= max_dist or attempt == max_retries:
            break

        # 2) Reformulate query if too low
        print(f"[Feedback] High distance ({avg_dist:.2f}); reformulating query...")

        # Basic heuristic: ask the LLM to rewrite or expand the query
        new_question = replace_query_tokens_with_synonyms(question, gene_tool)
        if new_question == question:
            # No possible synonym replacement; no point looping
            break
        question = new_question

    if avg_dist > max_dist:  # give up retrieving and answer directly
      return question, None, avg_dist
    return question, retrieved, avg_dist

llm = create_llm()
gene_tool = GeneSynonymTool("gene_synonyms.csv")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

for query in queries:
    out = answer_question(vectordb, llm, query, cross_encoder=cross_encoder,k_final=2,gene_tool=gene_tool,max_dist=threshold)
    print(out)

