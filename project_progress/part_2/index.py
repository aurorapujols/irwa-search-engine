import os
import sys
import random
import string
import json

import  nltk
nltk.download('stopwords')

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from project_progress.part_1.data_prep import build_terms, join_build_terms, corpus_df_loading
from collections import defaultdict
from array import array

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env

def create_index(corpus):
    """
    Implement the inverted index.

    :param corpus: corpus (collection of products' documents)
    :return: the inverted index containing terms as keys and the corresponding list of documents where these keys appear in (and the positions) as values
    """

    # Initialize variables
    index = defaultdict(dict)
    metadata = {}
    info_index = {}  # dictionary to map products `title` to document ids

    # Read each product's article in the corpus
    for doc in list(corpus.values()):

        # We get the information we are gonna need for each document
        pid = doc.pid
        title = doc.title
        description = doc.description
        brand = doc.brand
        category = doc.category
        sub_category = doc.sub_category
        product_details = doc.product_details
        seller = doc.seller

        # Store the categorical data in a dictionary for further retrieval
        info_index[pid] = {
            "title": title,
            "description": description,
            "brand": brand,
            "category": category,
            "sub_category": sub_category,
            "product_details": product_details,
            "seller": seller
        }
        
        # Separate the numerical data, the out of stock and url for future filtering and other purposes
        metadata[pid] = {
            "out_of_stock": doc.out_of_stock,
            "selling_price": doc.selling_price,
            "discount": doc.discount,
            "actual_price": doc.actual_price,
            "average_rating": doc.average_rating,
            "url": doc.url
        }
        

        # For each field to be considered in the index, get its terms
        title_description = join_build_terms([title, description])  # Pre-process the `title` and `description`
        brand_terms = join_build_terms([brand])
        category_terms = join_build_terms([category])
        sub_category_terms = join_build_terms([sub_category])
        seller_product_details = join_build_terms([seller, " ".join([detail for detail in product_details.values()])])  # Pre-process the `title` and `description`

        # Fields and target terms to process
        fields = ['title_description', 'brand', 'category', 'sub_category', 'seller_product_details']
        target = [title_description, brand_terms, category_terms, sub_category_terms, seller_product_details]

        # Initialize a temporal dictionary to store the index terms for the current article
        current_article_index = defaultdict(dict)

        # Create the index for the current article
        for idx, field in enumerate(fields):    # For each field we consider in the index
            for position, term in enumerate(target[idx]):   # For each term in the target field
                try:
                    current_article_index[term][field][1].append(position)  # Add the new found term's position to the dict
                except:
                    current_article_index[term][field] = [pid, [position]]  # Create the entry for the term and field with the term's position if it didn't exist

        # Join the current aarticle's index with the global index
        for term, position_in_fields in current_article_index.items():  # For each term in the current index (and its sub-dictonary of fields)
            for field, posting in position_in_fields.items():   # For each field in the sub-dictionary of fields (and its positions in the article)
                try:
                    index[term][field].append(posting)  # Add the array of positions ("[id, [[0],[1]]]"") in the given term and field
                except:
                    index[term][field] = [posting]      # Create the entry for the term and the field with the array of positions
    
    # Return the final index, the info_index with the categorical values map, and the metadata map
    return index, info_index, metadata


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

# def query_index()

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
        index, info_index, metadata = create_index(corpus)
        with open("inverted_index.json", "w") as f:
            json.dump(index, f, indent=2)
        print("Inverted index created and saved to 'inverted_index.json'")

    # Perform search
    docs = search("men slim jeans blue", index)
    print(f"Selected docs: {docs}")