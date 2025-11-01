import os
import json
import math
import collections
import numpy as np
import pandas as pd

from collections import defaultdict

from project_progress.part_1.data_prep import (
    build_terms,
    join_build_terms,
    corpus_df_loading,
)

from dotenv import load_dotenv

load_dotenv()

# -------------------- FUNCTIONS AND PARAMETERS ----------------------------------------- #
weights = {
        "title_description": 0.5,
        "brand": 0.05,
        "category": 0.2,
        "sub_category": 0.2,
        "seller_product_details": 0.05,
}


def create_index_tf_idf(corpus):
    """
    Implement the inverted index and compute tf, df and idf

    Argument:
    corpus -- list of Documents containing the description of the product

    Returns:
    index - the inverted index (implemented through a Python dictionary) containing terms as keys 
     and the corresponding list of document these keys appears in (and the positions) as values.
    index2title - a mapping of article pid to its title
    tf - normalized term frequency for each term in each document
    df - number of documents each term appear in
    idf - inverse document frequency of each term
    """

    index = defaultdict(dict)
    tf = defaultdict(lambda: defaultdict(dict))
    df = defaultdict(int)
    idf = {}
    index2title = {}
    num_articles = len(corpus)

    for product in list(corpus.values()):
        index2title[product.pid] = product.title
        # For each field to be considered in the index, get its terms
        title_description = join_build_terms(
            [product.title, product.description]
        )  # Pre-process the `title` and `description`
        brand_terms = join_build_terms([product.brand])
        category_terms = join_build_terms([product.category])
        sub_category_terms = join_build_terms([product.sub_category])
        seller_product_details = join_build_terms(
            [
                product.seller,
                " ".join([detail for detail in product.product_details.values()]),
            ]
        )  # Pre-process the `title` and `description`

        # Fields and target terms to process
        fields = [
            "title_description",
            "brand",
            "category",
            "sub_category",
            "seller_product_details",
        ]
        target = [
            title_description,
            brand_terms,
            category_terms,
            sub_category_terms,
            seller_product_details,
        ]

        # Initialize a temporal dictionary to store the index terms for the current article
        current_article_index = defaultdict(dict)

        # Create the index for the current article
        # For each field we consider in the index
        for field_idx, field in enumerate(fields):
            # For each term in the target field
            for position, term in enumerate(target[field_idx]):
                try:
                    # Add the new found term's position to the dict
                    current_article_index[term][field][1].append(position)
                except:
                    # Create the entry for the term and field if it didn't exist
                    current_article_index[term][field] = [product.pid, [position]]

        seen_terms = set()
        for field in fields:
            norm = 0
            for term, postings in current_article_index.items():
                if (
                    field in postings.keys()
                ):  # Check that the term appear in the current field
                    norm += len(postings[field][1]) ** 2
            norm = math.sqrt(norm)

            for term, postings in current_article_index.items():
                if field in postings.keys():
                    pid = postings[field][0]
                    # Compute term frequency and document frequency of each term per category
                    tf[pid][term][field] = np.round(len(postings[field][1]) / norm, 4)

                    # In the practice, the tf/df and the index were in separate loops. Both 
                    # codes are now in one loop to avoid reading the same twice
                    # Join the current article's index with the global index
                    try: # Add the array of positions ("[id, [[0],[1]]]"") in the given term and field
                        index[term][field].append(
                            postings[field]
                        )  
                    except: # Create the entry for the term and the field with the array of positions
                        index[term][field] = [
                            postings[field]
                        ]  

                if term not in seen_terms:
                    df[term] += 1
                    seen_terms.add(term)

    for term in df.keys():
        idf[term] = np.round(np.log(float(num_articles / df[term])), 4)

    return index, index2title, tf, df, idf

def rank_tf_idf(query_terms, products, index, tf, idf, weights):
    """
    Perform the ranking of the results of a search based on the tf-idf weights

    Argument:
    terms -- list of query terms
    products -- list of products to rank that match the query
    index -- inverted index data structure
    tf -- term frequencies
    idf -- inverted document frequencies
    weights -- weights to average the fields to compute the rank

    Returns:
    Print the list of product ids of the ranked articles
    """
    # For the docs, take only the components for the query terms
    product_vectors = defaultdict(lambda: [0] * len(query_terms))
    query_vector = [0] * len(query_terms)

    # Compute the norm for the query tf
    query_term_counts = collections.Counter(query_terms)
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
                        tf[pid][q_term][field] * weights[field]
                    )

    product_scores = [
        [np.dot(current_prod_vec, query_vector), prod]
        for prod, current_prod_vec in product_vectors.items()
    ]
    product_scores.sort(reverse=True)

    return product_scores

def filter(query, index):
    """
    The output is the list of documents that contain ALL query terms.

    :param query: (string) query
    :param index: (Dict) inverted index dictinary
    :return selected_docs: (List) of documents' ids that contain all query terms
    """

    query_terms = build_terms(query)
    docs = None
    for term in query_terms:
        try:
            # Get all doc ids from the term
            all_doc_ids = [
                doc_id
                for categories in index[term].values()
                for doc_id, _ in categories
            ]

            # Get intersection of documents with ALL the terms
            if docs is None:  # First time, set is empty
                docs = set(all_doc_ids)  # Initiallize with first term's doc ids
            else:
                docs &= set(all_doc_ids)

        except:
            pass

    return query_terms, list(docs)

def engine_search(query, index, tf, idf, weights):
    query_terms, filtered_products = filter(query=query, index=index)
    if (len(filtered_products) == 0):
        return None
    scores = rank_tf_idf(
        query_terms=query_terms,
        products=filtered_products,
        index=index,
        tf=tf,
        idf=idf,
        weights=weights,
    )
    return scores

def get_index_and_metrics(corpus):
    file_path = os.path.join(os.getcwd(), "data", "index_and_metrics.json")

    if os.path.exists(file_path):
        print("\033[34mLoading index and measures...\033[0m")
        with open(file_path, "r") as f:
            data = json.load(f)
        return data["index"], data["index2title"], data["tf"], data["df"], data["idf"]
    else:
        print("\033[34mComputing index and measures... (this might take a while)\033[0m")
        index, index2title, tf, df, idf = create_index_tf_idf(corpus)
        data = {
            "index": index,
            "index2title": index2title,
            "tf": tf,
            "df": df,
            "idf": idf
        }
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        return index, index2title, tf, df, idf

# -------------------- SEARCH ENGINE ---------------------------------------------------- #

if __name__ == "__main__":
    # Load the corpus
    print("\033[34mLoading corpus...\033[0m")
    json_path = os.getenv("DATA_FILE_PATH")
    corpus = corpus_df_loading(json_path)

    # Create index
    index, index2title, tf, df, idf = get_index_and_metrics(corpus)

    print("\033[33m\nClothing Articles Search Engine\033[0m")
    exit = False
    while(not exit):
        query = input("\033[33m\nEnter search terms: \033[0m")
        scored_products = engine_search(query, index, tf, idf, weights)
        if scored_products != None:
            print(f"\033[32mProducts found (10/{len(scored_products)}):\033[0m")
            for idx, (score, pid) in enumerate(scored_products):
                if(idx >= 10):
                    break
                print(f"{idx+1}. [idf={score:.3f}] {index2title[pid]}")
        else:
            print("\033[32mNo results found!\033[0m")

        answer = input("\033[33mWould you like to do another search[Y/n]: \033[0m").strip().lower()
        exit = answer in ("N", "n", "no", "No", "NO")
        
    