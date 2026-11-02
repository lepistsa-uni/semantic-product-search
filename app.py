
# --- LIBRARIES --- #

import json
from pathlib import Path

import pandas as pd
import gradio as gr

# --- MODULES --- #

from src.cleaning import clean_text
from src.retrievers import (
    DB_DIR,
    load_model,
    open_products_table,
    build_bm25,
    search_bm25,
    search_dense,
    fuse_rrf,
    load_reranker,
    rerank_cross_encoder,
    select_llm
)

# --- SETTINGS --- #

SEARCH_DEPTH = 20 
SHOW_TOP = 5
RESULTS_DIR = "data/clean"


# --- HELPER FUNCTIONS --- #

def add_rank(results):
    # add rank column to results dataframe
    results = results.copy()
    results.insert(0, "rank", range(1, len(results) + 1))
    return results


def load_method_ranking():
    # read all results from RESULTS_DIR and combine into a single dataframe
    all_results = []
    for path in sorted(Path(RESULTS_DIR).glob("*_results.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for row in data["results"]:
            all_results.append(row)

    ranking = pd.DataFrame(all_results)
    ranking = ranking.drop_duplicates(subset="Method", keep="last")

    ranking = ranking[["Method", "Hit@1", "Recall@5", "Recall@10", "MRR"]]
    ranking = ranking.sort_values("Hit@1", ascending=False).round(3)
    return ranking


# --- LOADING --- #

print("Loading model...")
model = load_model()
print("Loading products table...")
tbl = open_products_table(DB_DIR)
print("Converting lancedb table to pandas dataframe...")
products = pd.DataFrame(tbl.to_arrow().to_pandas())
print("Building BM25 index...")
bm25 = build_bm25(products)
method_ranking = load_method_ranking()
print("Loading reranker...")
reranker = load_reranker()
print("Done loading.")


# --- RETRIEVAL --- #

def run_search(user_text):
    
    query = clean_text(user_text)

    if query == "": # if empty query, return empty results
        empty = pd.DataFrame()
        return "", empty, empty, empty, empty, empty, method_ranking
    
    # bm25 results
    bm25_results = search_bm25(query, bm25, products, SEARCH_DEPTH)

    # dense results
    dense_results = search_dense(query, model, tbl, SEARCH_DEPTH)

    # fuse results
    hybrid_results = fuse_rrf(bm25_results, dense_results, SHOW_TOP)

    # rerank hybrid results using cross-encoder
    reranked_results = rerank_cross_encoder(query, hybrid_results, reranker, products, SHOW_TOP)

    # select product using LLM
    llm_selection = select_llm(query, hybrid_results, products)


    return (
        query, 
        bm25_results.head(SHOW_TOP), 
        dense_results.head(SHOW_TOP), 
        hybrid_results.head(SHOW_TOP),
        reranked_results, 
        llm_selection,
        method_ranking
    )

# --- GRADIO INTERFACE --- #


interface = gr.Interface(
    fn=run_search,
    inputs=gr.Textbox(label="Query Text"),
    outputs=[
        gr.Textbox(label="Cleaned Query"), 
        gr.Dataframe(label="BM25"), 
        gr.Dataframe(label="Dense"), 
        gr.Dataframe(label="Hybrid (RRF)"),
        gr.Dataframe(label="Cross-Encoder Reranked (Hybrid)"), 
        gr.Dataframe(label="LLM Selection (Qwen 3.5 4B)"),
        gr.Dataframe(label="Method Ranking (dev-set)")],
    title="Semantic Product Search",
    description="Enter query for semantic product search.",
    flagging_mode="manual",
)
interface.launch()