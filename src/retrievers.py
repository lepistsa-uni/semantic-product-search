


def load_model():
    '''multilingual-e5-base model for queries'''
    pass

def open_products_table(db_dir):
    '''open products table from lancedb'''
    pass

def build_bm25(products):
    '''build BM25 index from search_text column of clean products table'''
    pass

def search_bm25(query, bm25, products, top_k):
    '''search BM25 index for query and return top_n results as product numbers'''
    pass

def search_dense(query, model, tbl, top_k):
    '''search dense index for query and return top_n results as product numbers. needs "query:" prefix for query to work'''
    pass

def fuse_rrf(keyword_list, dense_list, top_k, k=60):
    '''fuse two lists of product numbers using Reciprocal Rank Fusion (RRF). 1 / (k + rank)'''
    pass

