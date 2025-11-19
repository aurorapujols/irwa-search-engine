import os, sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join('..', '..')))

from collections import defaultdict, Counter
from dotenv import load_dotenv

from myapp.search import load_corpus as lc
from project_progress.part_1.data_prep import corpus_df_loading, build_terms, join_build_terms
from project_progress.part_2.index_tf_idf import get_index_and_metrics

load_dotenv()  # take environment variables from .env

# Preprocessing -------------------------------------------------------------------------#
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
products_filepath = os.path.join(BASE_DIR, "..", "..", "data", "products.json")
products_numeric_data_filepath = os.path.join(BASE_DIR, "..", "..", "data", "products_numeric_data.json")

def dump_data(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_or_create(filepath, compute_data_function):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        data = compute_data_function()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return data
    
def get_numerical_info_by_product(corpus):
    """
    Function to process the corpus and get only the preprocessed numerical values for each product.

    :param corpus: Corpus of products.
    :return result_dict: dictionary of pid -> dict [with "out_of_stock", "average_rating", "selling_price", and "discount"]
    """

    products = [doc.model_dump() for doc in corpus.values()]
    corpus_df = pd.DataFrame(products)

    selected_attr = corpus_df[["pid", "out_of_stock", "average_rating", "selling_price", "discount"]].copy()

    # Input the value of average price to products without a price
    mean_price = selected_attr["selling_price"].mean()
    selected_attr.loc[:, "selling_price"] = (
        corpus_df["selling_price"].fillna(corpus_df["actual_price"]).fillna(mean_price)
    )

    # Give a 0 rating to non-rated products
    selected_attr.loc[:, "average_rating"] = corpus_df["average_rating"].fillna(0.0)

    # Give a discount of 0 to products without discount info
    selected_attr.loc[:, "discount"] = corpus_df["discount"].fillna(0.0)

    # Convert to dict
    result_dict = selected_attr.set_index("pid").to_dict(orient="index")
    
    return result_dict

def process_products(corpus):
    """
    Function that loads the products of the corpus as a dictionary of pid -> list (of categorical data as a list of processed tokens)

    :param corpus: Corpus with all the products and all their data.
    :return products: (dict) pid -> tokens (all preprocessed terms of the categorical data)
    """
    products = {}
    for product in list(corpus.values()):
        
        # Process the documents categorical fields as in the creation of the inverted index, but concatenate all the terms in a single string (in this case we do not care about the fields)
        products[product.pid] = join_build_terms([product.title, 
                                            product.description,
                                            product.brand,
                                            product.category,
                                            product.sub_category,
                                            product.seller,
                                            " ".join([detail for detail in product.product_details.values()])])

    return products
# ---------------------------------------------------------------------------------------#

# TF-IDF ranking ------------------------------------------------------------------------#
def rank_tf_idf_cosine_similarity(query_terms, products, index, tf, idf, weights):
    """
    Perform the ranking of the results of a search based on the tf-idf weights and using cosine similarity

    :param terms: list of query terms
    :param products: list of products to rank that match the query
    :param index: inverted index data structure
    :param tf: term frequencies
    :param idf: inverted document frequencies
    :param weights: weights to average the fields to compute the rank

    :return product_scores: (list) of sorted decreasingly value pairs [score, pid] with the score of each pid in the ranking
    """
    # For the docs, take only the components for the query terms
    product_vectors = defaultdict(lambda: [0] * len(query_terms))
    query_vector = [0] * len(query_terms)

    # Compute the norm for the query tf
    query_term_counts = Counter(query_terms)
    query_norm = np.linalg.norm(list(query_term_counts.values()))

    # Compute tf-idf for each document and query
    for term_idx, q_term in enumerate(query_terms):
        if q_term not in index: # or q_term not in idf:    # just to control
            continue

        # tf*idf (normalize TF)
        query_vector[term_idx] = (query_term_counts[q_term] / query_norm) * idf[q_term]

        # Compute the document vectors
        second_loop = False
        for field, postings in index[q_term].items():
            for pid, positions in postings:
                if pid in products:
                    product_vectors[pid][term_idx] += (
                        tf[pid][q_term][field] * idf[q_term] * weights[field]
                    )

    # Compute cosine similarity scores (we do not use the query norm because it is a constant)
    product_scores = []

    for pid, current_prod_vec in product_vectors.items():
        doc_norm = np.linalg.norm(current_prod_vec)
        if doc_norm == 0:
            score = 0.0
        else:
            score = np.dot(current_prod_vec, query_vector) / doc_norm
        product_scores.append([score, pid])

    # Sort by score descending
    product_scores.sort(key=lambda x: x[0], reverse=True)

    return product_scores
# ---------------------------------------------------------------------------------------#

# BM25 ranking --------------------------------------------------------------------------#
def compute_prod_data(products):
    """
    Compute the general metrics for all documents to use in BM25

    :param products: (dict) pid -> processed text from each product in corpus
    :return df: (dict) term -> document frequency
    :return idf: (dict) term -> inverted document frequency
    :return Ld: (dict) pid -> document length
    :return Lave: (float) average of the document length in whole collection
    """

    df = defaultdict(int)
    idf = {}
    Ld = {}
    Lave = 0
    
    N = len(products)

    for pid, terms in products.items():

        # Compute df
        unique_terms = set(terms)
        for term in unique_terms:
            df[term] += 1
            idf[term] = np.log(N/df[term])

        # Compute Ld
        Ld[pid] = len(products[pid])

    # Compute Lave
    Lave = np.sum(list(Ld.values()))/len(Ld.values())

    return df, idf, Ld, Lave

def rank_BM25(query_terms, products, filtered_prod, idf, Ld, Lave, k1=1.2, b=0.75):
    """
    Ranking of specific products using BM25 algorithm.

    :param query_terms: (list) of terms in the query
    :param products: (dict) pid -> (list) of terms in the categorical variables of product pid
    :param filtered_prod: (list) of pids of the products containing all query terms
    :param idf: (dict) pid -> (float) inverse document frequency of all the products
    :param Ld: (dict) pid -> (int) length of all documents
    :param Lave: (float) average document length in the whole corpus
    :param k1: tf tunning parameter, k1=1.2 if not specified
    :param b: document length normalization tunning parameter, b=0.75 if not specified
    :return RSV_scores: (list) of sorted decreasingly value pairs [score, pid] with the score of each pid in the ranking
    """
    tf = defaultdict(dict)
    RSV_scores = []

    for pid in filtered_prod:

        # Compute the tf of the query terms in the filtered documents
        tf[pid] = Counter(products[pid])    # computes the number of times a word appears in the list of words 'terms'

        # Compute the score for the filtered products
        RSVd = 0
        for q_term in query_terms:
            nominator = (k1 + 1) * tf[pid][q_term]
            denominator = k1 * ((1-b) + b*(Ld[pid]/Lave)) + tf[pid][q_term]
            RSVd += idf[q_term] * (nominator/denominator)

        RSV_scores.append([RSVd, pid])

    RSV_scores.sort(key=lambda x: x[0], reverse=True)

    return RSV_scores
# ---------------------------------------------------------------------------------------#

# Custom ranking ------------------------------------------------------------------------#
def rank_custom(query_terms, metadata, products, filtered_prod, idf, Ld, Lave, k1=1.2, b=0.75, \
                lambda_=0.3, popularity=0.3, affordability=0.7, availability=0.01):
    """
    Custom ranking function.

    :param query_terms: (list) of terms in the query
    :param metadata: (dict) pid -> (dict) with the numerical data for each product
    :param products: (dict) pid -> (list) of terms in the categorical variables of product pid
    :param filtered_prod: (list) of pids of the products containing all query terms
    :param idf: (dict) pid -> (float) inverse document frequency of all the products
    :param Ld: (dict) pid -> (int) length of all documents
    :param Lave: (float) average document length in the whole corpus
    :param k1: tf tunning parameter, k1=1.2 if not specified
    :param b: document length normalization tunning parameter, b=0.75 if not specified
    :param lambda_: bm25 score tunning parameter (the higher the biggest influence the score has in the ranking), lambda_=0.6 if not specified
    :param popularity: weight of the products' rating in the final score, popularity=0.3 if not specified
    :param affordability: weight of the products' price and discount in the final score, affordability=0.7 if not specified
    :param availability: penalization for the score of products out of stock, availability=0.01 if not specified
    :return custom_scores: (list) of sorted decreasingly value pairs [score, pid] with the score of each pid in the ranking
    """
    tf = defaultdict(dict)
    custom_scores = []
    numerical_data = metadata

    for pid in filtered_prod:

        # Compute the tf of the query terms in the filtered documents
        tf[pid] = Counter(products[pid])    # computes the number of times a word appears in the list of words 'terms'

        # Compute the score for the filtered products
        RSVd = 0
        for q_term in query_terms:            
            nominator = (k1 + 1) * tf[pid][q_term]
            denominator = k1 * ((1-b) + b*(Ld[pid]/Lave)) + tf[pid][q_term]
            RSVd += idf[q_term] * (nominator/denominator)

        # Modify to obtain custom score
        rating_score = numerical_data[pid]["average_rating"] / 5.0
        discount_score = numerical_data[pid]["discount"] / 100.0
        price_score = 1 / (1 + np.log(numerical_data[pid]["selling_price"]))  # cheaper = higher score
        availability_score = availability if numerical_data[pid]["out_of_stock"] else 1

        score_d = (lambda_ * RSVd) + (1-lambda_) * (popularity * rating_score + affordability * (discount_score + price_score))
        final_score_d = score_d * availability_score

        custom_scores.append([final_score_d, pid])

    custom_scores.sort(key=lambda x: x[0], reverse=True)
    
    return custom_scores
# ---------------------------------------------------------------------------------------#

# Pipeline ------------------------------------------------------------------------------#
def filter(query, products):
    """
    The output is the list of documents that contain ALL query terms.

    :param query: (string) query
    :param products: (Dict) pid -> document text
    :return selected_docs: (List) of documents' ids that contain all query terms
    """

    query_terms = build_terms(query)  # tokenize query
    selected_docs = []

    for pid, prod_terms in products.items():
        # check if ALL query terms are in this document
        if all(term in prod_terms for term in query_terms):
            selected_docs.append(pid)

    return query_terms, selected_docs

def print_ranking(scores, index2title):
    """
    Function to print the ranking for an arbitrary algorithm.

    :param scores: (list) of value pairs [score, pid] with score being the score for the document pid
    :param index2title: (dict) pid -> (string) content of the "title" field of the product
    """
    for idx, (score, pid) in enumerate(scores):
        print(f"{idx+1:4}. [score={score:.3f}] {index2title[pid]}")
# ---------------------------------------------------------------------------------------#

if __name__ == "__main__":

    # Load the corpus
    print("\033[34mLoading corpus...\033[0m")
    json_path = os.getenv("DATA_FILE_PATH")
    corpus = corpus_df_loading(json_path)
    print("\033[34mDONE! Starting preprocessing...\033[0m")

    # Preprocess the products and precompute the inverted index
    products = get_or_create(products_filepath, lambda: process_products(corpus))
    numeric_data = get_or_create(products_numeric_data_filepath, lambda: get_numerical_info_by_product(corpus))
    print("\033[34mDONE! Computing metrics...\033[0m")
    index, index2title, tf, df, idf = get_index_and_metrics(corpus)
    df_bm25, idf_bm25, Ld, Lave = compute_prod_data(products=products)
    print("\033[34mREADY!\033[0m")

    # ENGINE START
    print("\033[33m\nClothing Articles Search Engine\033[0m")
    exit = False
    while(not exit):

        '''STEP 1: take a query as input'''
        query = input("\033[33m\nEnter search terms: \033[0m")

        '''STEP 2: filter the products'''
        query_terms, filtered_prod = filter(query, products=products)

        # Rank with the three methods                
        if (len(filtered_prod) == 0):   # Compute ranking only if we found documents during filtering
            print("\n\033[91mNo results!\033[0m")
        else:

            # TF-IDF + cosine similarity
            print("\n\033[93mTF-IDF + cosine similarity\033[0m")

            WEIGHTS = {
                "title_description": 1.0,
                "brand": 0.5,
                "category": 0.3,
                "sub_category": 0.4,
                "seller_product_details": 0.2,
            } 
            scores_tfidf = rank_tf_idf_cosine_similarity(query_terms=query_terms, 
                                    products=filtered_prod,
                                    index=index,
                                    tf=tf,
                                    idf=idf,
                                    weights=WEIGHTS)
            print_ranking(scores=scores_tfidf, index2title=index2title)

            # BM25
            print("\n\033[93mBM25\033[0m")
            scores_BM25 = rank_BM25(query_terms=query_terms,
                                    products=products,
                                    filtered_prod=filtered_prod,
                                    idf=idf_bm25,
                                    Ld=Ld,
                                    Lave=Lave)
            print_ranking(scores=scores_BM25, index2title=index2title)

            # Custom Ranking
            print("\n\033[93mCustom Ranking\033[0m")
            scores_custom = rank_custom(query_terms=query_terms,
                                        metadata=numeric_data,
                                        products=products,
                                        filtered_prod=filtered_prod,
                                        idf=idf,
                                        Ld=Ld,
                                        Lave=Lave)
            print_ranking(scores=scores_custom, index2title=index2title)


        answer = input("\033[33mWould you like to do another search[Y/n]: \033[0m").strip().lower()
        exit = answer in ("N", "n", "no", "No", "NO")
    # ENGINE END