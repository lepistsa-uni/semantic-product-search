
# --- LIBRARIES --- #

import numpy as np
import pandas as pd
import torch
import lancedb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from src.cleaning import tokenize_stem
from src.cleaning import clean_text


# --- SETTINGS --- #

DB_DIR = "data/clean/indexes/lancedb" # notebook 05
MODEL_NAME = "intfloat/multilingual-e5-base" # notebook 05
TABLE_NAME = "products" # notebook 05
MAX_SEQ_LENGTH = 256 # notebook 05
RRF_K = 60 # update after testing


# --- BUILD FUNCTIONS --- #

def load_model():
    '''multilingual-e5-base model for queries'''
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = MAX_SEQ_LENGTH
    return model
    

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

    top_products = products[["clean_label", "product_number"]].iloc[top_indices] # get product numbers of top_k products
    
    top_products = top_products.reset_index(drop=True) # reset index of top_products dataframe, that is, the index will be 0, 1, 2, ... instead of the original index from products dataframe
    
    top_products["bm25_score"] = scores[top_indices].round(2) # dedicated column for bm25 scores

    return top_products




def search_dense(query, model, tbl, top_k):
    '''search dense index for query and return top_n results as product numbers. needs "query:" prefix for query to work'''
    pass

def fuse_rrf(keyword_list, dense_list, top_k, k=60):
    '''fuse two lists of product numbers using Reciprocal Rank Fusion (RRF). 1 / (k + rank). K-value comes from notebook 06'''
    pass

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
    query = clean_text("kuusiokoloruuvi m8 x 40 a4")
    top_k = 5
    top_products = search_bm25(query, bm25, products, top_k)
    print(top_products)