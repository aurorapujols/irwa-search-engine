import os
import sys
import random
import string
import json
import math
import numpy as np

import  nltk
nltk.download('stopwords')

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from project_progress.part_1.data_prep import build_terms, join_build_terms, corpus_df_loading
from collections import defaultdict
from array import array

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env

def create_index_tf_idf(corpus):
    """
    Implement the inverted index and compute tf, df and idf

    Argument:
    corpus -- 

    #TODO: adapt to our version

    Returns:
    index - the inverted index (implemented through a Python dictionary) containing terms as keys and the corresponding
    list of document these keys appears in (and the positions) as values.
    index2title - a mapping of article pid to its title
    tf - normalized term frequency for each term in each document
    df - number of documents each term appear in
    idf - inverse document frequency of each term
    """

    index = defaultdict(dict)
    tf = defaultdict(dict)
    df = defaultdict(dict)
    idf = defaultdict(dict)
    index2title = {}
    num_articles = len(corpus)

    for doc in list(corpus.values()):
        index2title[doc.pid] = doc.title
        # For each field to be considered in the index, get its terms
        title_description = join_build_terms([doc.title, doc.description])  # Pre-process the `title` and `description`
        brand_terms = join_build_terms([doc.brand])
        category_terms = join_build_terms([doc.category])
        sub_category_terms = join_build_terms([doc.sub_category])
        seller_product_details = join_build_terms([doc.seller, " ".join([detail for detail in doc.product_details.values()])])  # Pre-process the `title` and `description`

        # Fields and target terms to process
        fields = ['title_description', 'brand', 'category', 'sub_category', 'seller_product_details']
        target = [title_description, brand_terms, category_terms, sub_category_terms, seller_product_details]

        # Initialize a temporal dictionary to store the index terms for the current article
        current_article_index = defaultdict(dict)

        # Create the index for the current article
        # For each field we consider in the index
        for idx, field in enumerate(fields):    
            # For each term in the target field
            for position, term in enumerate(target[idx]):   
                try:
                    # Add the new found term's position to the dict
                    current_article_index[term][field][1].append(position)  
                except:
                    # Create the entry for the term and field with the term's position if it didn't exist
                    current_article_index[term][field] = [doc.pid, [position]]  

        for field in fields:
            norm = 0
            for term, positions in current_article_index.items():
                norm += len(positions[field][1])**2
            norm = math.sqrt(norm)

            for term, positions in current_article_index.items():
                # Compute term frequency and document frequency of each term per category
                try:
                    tf[term][field].append(np.round(len(positions[field][1])/norm, 4))
                    df[term][field] += 1
                # If it's a term we haven't seen, create a new term frequency and document frequency entry
                except:
                    tf[term][field] = [np.round(len(positions[field][1])/norm, 4)]
                    df[term][field] = 1

                # In the practice, the tf/df and the index were in separate loops. Both codes are now in one 
                # loop to avoid reading the same twice
                # Join the current article's index with the global index
                try:
                    index[term][field].append(positions[field])  # Add the array of positions ("[id, [[0],[1]]]"") in the given term and field
                except:
                    index[term][field] = [positions[field]]      # Create the entry for the term and the field with the array of positions


    for term, posting_fields in df:
        for field, posting in posting_fields:
            idf[term][field] = np.round(np.log(float(num_articles / df[term])), 4)

    return index, index2title, tf, df, idf

def search(query, index):
    """
    The output is the list of documents that contain ALL query terms.

    :param query: (string) query
    :param index: (Dict) inverted index dictinary
    :return selected_docs: (List) of documents' ids that contain all query terms
    """

    query = build_terms(query)
    docs = None
    for term in query:
        try:
            # Get all doc ids from the term
            all_doc_ids = [ doc_id
                            for categories in index[term].values()
                            for doc_id, positions in categories
                        ]
            
            # Get intersection of documents with ALL the terms
            if docs is None:    # First time, set is empty
                docs = set(all_doc_ids)     # Initiallize with first term's doc ids
            else:
                docs &= set(all_doc_ids) 
                
        except:
            pass
    
    return docs

def rank_tf_idf(terms, docs, index, tf, idf):
        """
    Perform the ranking of the results of a search based on the tf-idf weights

    Argument:
    terms -- list of query terms
    docs -- list of products to rank that match the query
    index -- inverted index data structure
    tf -- term frequencies
    idf -- inverted document frequencies

    Returns:
    Print the list of product ids of the ranked articles
    """
    pass

if __name__ == "__main__":

    # Load the corpus
    json_path = os.getenv("DATA_FILE_PATH")
    corpus = corpus_df_loading(json_path)

    # Try to load the index from file
    try:
        with open("inverted_index.json", "r") as f:
            index = json.load(f)
        print("Index loaded from 'inverted_index.json'")
    except (FileNotFoundError, json.JSONDecodeError):
        print("Index file not found or invalid. Computing index...")
        index, index2title, tf, df, idf = create_index_tf_idf(corpus)
        with open("inverted_index.json", "w") as f:
            json.dump(index, f, indent=2)
        print("Inverted index created and saved to 'inverted_index.json'")

    # Perform search
    docs = search("men slim jeans blue", index)
    print(f"Selected docs: {docs}")