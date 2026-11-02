
# --- LIBRARIES --- #

import numpy as np
import pandas as pd
import torch
import lancedb
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from src.cleaning import tokenize_stem
from src.cleaning import clean_text


# --- SETTINGS --- #

DB_DIR = "data/clean/indexes/lancedb" # notebook 05
MODEL_NAME = "intfloat/multilingual-e5-base" # notebook 05
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3" # notebook 07
TABLE_NAME = "products" # notebook 05
MAX_SEQ_LENGTH = 256 # notebook 05
RRF_K = 60 # update after testing


# --- BUILD FUNCTIONS --- #

def load_model():
    '''multilingual-e5-base model for queries'''
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = MAX_SEQ_LENGTH
    return model
    
def load_reranker():
    '''load cross-encoder model for reranking'''
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    reranker = CrossEncoder(RERANKER_MODEL, device=device, max_length=512)
    return reranker


def open_products_table(db_dir):
    '''open products table from lancedb'''
    db = lancedb.connect(db_dir) # open route to lancedb index folder
    tbl = db.open_table(TABLE_NAME) # open products table
    print(f"Table {TABLE_NAME} opened: {len(tbl):,} rows")
    return tbl

def build_bm25(products):
    '''build BM25 index from search_text column of clean products table'''
    corpus = []

    for text in products['search_text']: # tokenize each product's search_text column and append to corpus
        tokens = tokenize_stem(text)
        corpus.append(tokens)
        
    bm25 = BM25Okapi(corpus) # build BM25 index from corpus

    return bm25 # return BM25 index object
    

# --- SEARCH FUNCTIONS --- #

def search_bm25(query, bm25, products, top_k):
    '''search BM25 index for query and return top_n results as product numbers'''
    tokens = tokenize_stem(query) # tokenize query

    if len(tokens) == 0: # if query is empty, return empty list
        return []
    
    scores = bm25.get_scores(tokens)

    sorted_indices = np.argsort(scores) # ascending order, so best scores are at the end
    best_first = sorted_indices[::-1] # reverse to get descending order
    top_indices = best_first[:top_k] # get top_k indices

    top_products = products[["product_number", "clean_label"]].iloc[top_indices] # get product numbers of top_k products
    
    top_products = top_products.reset_index(drop=True) # reset index of top_products dataframe, that is, the index will be 0, 1, 2, ... instead of the original index from products dataframe
    
    top_products["bm25_score"] = scores[top_indices].round(2) # dedicated column for bm25 scores

    return top_products




def search_dense(query, model, tbl, top_k):
    '''search dense index for query and return top_n results as product numbers. needs "query:" prefix for query to work'''
    query_text = "query: " + query # add prefix to query
    query_vector = model.encode(query_text, normalize_embeddings=True) # encode query to vector, normalization makes vector length 1 thus making cosine similarity equivalent to dot product
    results = tbl.search(query_vector).limit(top_k).to_pandas() # search lancedb table for query vector and return top_k results as pandas dataframe
    top_products = results[["product_number", "clean_label", "_distance"]]
    top_products = top_products.rename(columns={"_distance": "dense_distance"}) # rename _distance column to dense_score
    top_products["dense_distance"] = top_products["dense_distance"].round(2) # round dense_score to 2 decimal places
    return top_products

def fuse_rrf(bm25_results, dense_results, top_k):
    '''fuse two lists of product numbers using Reciprocal Rank Fusion (RRF). 1 / (k + rank). K-value comes from notebook 06'''
    rrf_scores = {}
    clean_labels = {}

    rank = 1
    for _, row in bm25_results.iterrows():
        product_number = row["product_number"]
        rrf_scores[product_number] = rrf_scores.get(product_number, 0.0) + 1.0 / (RRF_K + rank)
        clean_labels[product_number] = row["clean_label"]
        rank += 1

    rank = 1
    for _, row in dense_results.iterrows():
        product_number = row["product_number"]
        rrf_scores[product_number] = rrf_scores.get(product_number, 0.0) + 1.0 / (RRF_K + rank)
        clean_labels[product_number] = row["clean_label"]
        rank += 1

    ranked_products = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
    ranked_products = ranked_products[:top_k] # get top_k product numbers

    rows = []
    for product_number in ranked_products:
        rows.append({
            "product_number": product_number,
            "clean_label": clean_labels[product_number],
            "rrf_score": round(rrf_scores[product_number], 4),
        })

    top_products = pd.DataFrame(rows)
    return top_products


def rerank_cross_encoder(query, candidates, reranker, products, top_k=5):
    '''rerank candidate products using cross-encoder'''
    if candidates.empty:
        return pd.DataFrame()
    
    # look up search_text for each candidate product_number
    cand_df = candidates.merge(
        products[['product_number', 'search_text']], 
        on='product_number', 
        how='left'
    )
    cand_df['search_text'] = cand_df['search_text'].fillna(cand_df['clean_label']).fillna('')
    
    # create pairs of (query, candidate search_text) for reranking
    pairs = [(query, text) for text in cand_df['search_text']]
    cand_df['reranker_score'] = reranker.predict(pairs)
    
    # sort candidates by reranker_score and return top_k
    reranked = cand_df.sort_values(by='reranker_score', ascending=False).head(top_k).reset_index(drop=True)
    reranked['reranker_score'] = reranked['reranker_score'].round(2)
    
    return reranked[['product_number', 'clean_label', 'reranker_score']]

if __name__ == "__main__":

    # --- TEST FUNCTIONS --- #
    
    # load model
    model = load_model()
    
    # open products table
    tbl = open_products_table(DB_DIR)
    
    # convert lancedb table to pandas dataframe
    products = pd.DataFrame(tbl.to_arrow().to_pandas())
    
    # build BM25 index
    bm25 = build_bm25(products)
    
    # test search_bm25 function
    query = clean_text("kuusiokoloruuvi din7984 a4 m8x40")
    top_k = 5
    top_products = search_bm25(query, bm25, products, top_k)
    print(top_products)


    tokens = tokenize_stem(query)
    print("Query tokens:", tokens)

    for product_number in top_products["product_number"]:
        row = products.index[products["product_number"] == product_number][0]
        print("\n", products["clean_label"].iloc[row])
        for token in tokens:
            token_score = bm25.get_scores([token])[row]
            print(f"   {token:20} {token_score:.2f}")

    # test search_dense function
    print("\nTesting search_dense function:")
    dense_products = search_dense(query, model, tbl, top_k)
    print(dense_products)
    
    # test fuse_rrf function

    print("RRF_K =", RRF_K)

    print("\nTesting fuse_rrf function:")
    bm25_20 = search_bm25(query, bm25, products, 20)
    dense_20 = search_dense(query, model, tbl, 20)
    print(bm25_20[["product_number", "clean_label"]])
    print(dense_20[["product_number", "clean_label"]])
    print(fuse_rrf(bm25_20, dense_20, top_k=5))