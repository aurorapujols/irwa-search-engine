import os, sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")

sys.path.append(os.path.abspath(os.path.join('..', '..')))

from collections import defaultdict, Counter
from dotenv import load_dotenv

from myapp.search import load_corpus as lc
from project_progress.part_1.data_prep import corpus_df_loading, build_terms, join_build_terms
from project_progress.part_2.index_tf_idf import get_index_and_metrics
from project_progress.part_3.ranking import (
    get_or_create,

)

from gensim.models import Word2Vec
from gensim.parsing.preprocessing import preprocess_string

load_dotenv()  # take environment variables from .env

# Preprocessing -------------------------------------------------------------------------#
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
product_sentences_filepath = os.path.join(BASE_DIR, "..", "..", "data", "products.json")


def get_product_sentences(corpus, preprocess=preprocess_string):
    """
    Function that loads the products of the corpus as a dictionary of pid -> list (of sentences. A sentence is the joined values of categorical data per affine categories: title + description, brand + category + subcategory, seller + details)

    :param corpus: Corpus with all the products and all their data.
    :return product_sentences: (dict) pid -> lists of tokens (all preprocessed terms of the categorical data per grouped subcategories)
    """
    products_sentences = {}

    for product in list(corpus.values()):
        # Process the documents categorical fields as in the creation of the inverted index, but concatenate all the terms in a single string (in this case we do not care about the fields)
        subgroups = [
            " ".join([product.title, product.description]),
            " ".join([product.brand, product.category, product.sub_category]),
            " ".join([product.seller, 
                      " ".join([detail for detail in product.product_details.values()])]
                      )
            ]
        products_sentences[product.pid] = [preprocess(group) for group in subgroups]
        
    return products_sentences
# ---------------------------------------------------------------------------------------#

# Word2Vec embedding --------------------------------------------------------------------#
def get_training_sentences(product_sentences):
    """
    Function that builds the training sentences from the product sentences.

    :param product_sentences: Dictionary of pid -> list of tokenized sentences.
    :return sentences: List of tokenized sentences for training the Word2Vec model.
    """
    sentences = []

    for prod_sentences in product_sentences.values():
        sentences.extend(prod_sentences)

    return sentences

def train_word2vec_model(sentences, vector_size=100, window=7, min_count=5, negative=10, sg=1, epochs=10):
    """
    Function that trains a Word2Vec model on the given sentences.

    :param sentences: List of tokenized sentences for training the Word2Vec model.
    :param vector_size: Dimensionality of the word vectors.
    :param window: Maximum distance between the current and predicted word within a sentence.
    :param min_count: Ignores all words with total frequency lower than this.
    :param negative: If > 0, negative sampling will be used, the int for the number of negative samples.
    :param sg: Training algorithm: 1 for skip-gram; otherwise CBOW.
    :param epochs: Number of iterations (epochs) over the corpus.
    :return model: Trained Word2Vec model.
    """
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        negative=negative,
        sg=sg, 
    )
    # model.train(sentences, total_examples=len(sentences), epochs=epochs)

    return model

def get_document_embeddings(word2vec_model, product_sentences):
    """
    Function that computes the document embeddings for each product by averaging the Word2Vec embeddings of the words in its sentences.

    :param word2vec_model: Trained Word2Vec model.
    :param product_sentences: Dictionary of pid -> list of tokenized sentences.
    :return doc_embeddings: Dictionary of pid -> document embedding (numpy array).
    """
    doc_embeddings = {}

    for pid, sentences in product_sentences.items():
        all_word_vectors = []

        for word in sentences:
            if word in word2vec_model.wv:
                all_word_vectors.extend(word2vec_model.wv[word])
            else:
                all_word_vectors.extend(np.zeros(word2vec_model.vector_size))

        doc_embeddings[pid] = np.mean(all_word_vectors, axis=0)

    return doc_embeddings
# ---------------------------------------------------------------------------------------#

# Ranking -------------------------------------------------------------------------------#
def cosine_similarity(document_representation, query_representation):
    """
    Function that computes the cosine similarity between a document and a query representation.

    :param document_representation: Numpy array representing the document.
    :param query_representation: Numpy array representing the query.
    :return similarity: Cosine similarity score.
    """
    dot_product = np.dot(document_representation, query_representation)
    norm_document = np.linalg.norm(document_representation)

    if norm_document == 0:
        return 0.0
    
    return dot_product / (norm_document)

def rank_documents(word2vec_model, doc_embeddings, preprocessed_query):
    """
    Function that ranks documents based on their cosine similarity to the query.

    :param word2vec_model: Trained Word2Vec model.
    :param doc_embeddings: Dictionary of pid -> document embedding (numpy array).
    :param query: Query string.
    :return similarities: List of [similarity score, pid] sorted in descending order of similarity.
    """    
    # Compute the query embedding
    query_vectors = []

    for word in preprocessed_query:
        if word in word2vec_model.wv:
            query_vectors.extend(word2vec_model.wv[word])
        else:
            query_vectors.extend(np.zeros(word2vec_model.vector_size))

    if len(query_vectors) == 0:
        query_embedding = np.zeros(word2vec_model.vector_size)
    else:
        query_embedding = np.mean(query_vectors, axis=0)
    
    # Compute similarities and rank documents
    similarities = []
    for pid, doc_embedding in doc_embeddings.items():
        sim = cosine_similarity(doc_embedding, query_embedding)
        similarities.append([sim, pid])
    similarities.sort(key=lambda x: x[0], reverse=True)

    return similarities

# ---------------------------------------------------------------------------------------#

# Pipeline ------------------------------------------------------------------------------#
def filter(query, doc_embeddings, product_sentences):
    """
    Function to filter the products based on the presence of all query terms in the product sentences.

    :param query: (string) The user query. 
    :param doc_embeddings: (dict) pid -> document embedding (numpy array).
    :param product_sentences: (dict) pid -> list of tokenized sentences.
    :return processed_query: (list) of preprocessed query terms.
    :return filtered_doc_embeddings: (dict) pid -> document embedding (numpy array) after filtering.
    """
    # Preprocess the query and extract terms
    processed_query = preprocess_string(query)
    query_terms = set(processed_query)

    # Filter documents that contain all query terms
    filtered_doc_embeddings = {}
    for pid, sentences in product_sentences.items():
        all_terms = set()
        for sentence in sentences:
            all_terms.update(sentence)
        if all_terms.intersection(query_terms) == query_terms:
            filtered_doc_embeddings[pid] = doc_embeddings[pid]  

    return processed_query, filtered_doc_embeddings
        


def print_ranking(scores, index2title):
    """
    Function to print the ranking for an arbitrary algorithm.

    :param scores: (list) of value pairs [score, pid] with score being the score for the document pid
    :param index2title: (dict) pid -> (string) content of the "title" field of the product
    """
    for idx, (score, pid) in enumerate(scores):
        print(f"{idx+1:4}. [score={score:.2f}] {index2title[pid]}")
# ---------------------------------------------------------------------------------------#

if __name__ == "__main__":

    # Load the corpus
    print("\033[34mLoading corpus...\033[0m")
    json_path = os.getenv("DATA_FILE_PATH")
    corpus = corpus_df_loading(json_path)
    print("\033[34mDONE! Starting preprocessing...\033[0m")

    # Preprocess the products and precompute the inverted index
    product_sentences = get_or_create(product_sentences_filepath, lambda: get_product_sentences(corpus))
    print("\033[34mDONE! Computing metrics...\033[0m")
    index, index2title, tf, df, idf = get_index_and_metrics(corpus)
    # Build the embedding space and compute document embeddings
    print("\033[34mDONE! Creating embedding space...\033[0m")
    training_sentences = get_training_sentences(product_sentences)
    word2vec_model = train_word2vec_model(training_sentences)
    print("\033[34mDONE! Generating product embeddings...\033[0m")
    doc_embeddings = get_document_embeddings(word2vec_model, product_sentences)
    print("\033[34mREADY!\033[0m")

    # Show ranking for predetermined queries
    print("\033[34mShowing ranking for predetermined queries...\033[0m")
    # Queries from part 2
    queries = [
        "western leather jacket men",  # context, material, specific cloth, gender
        "cotton innerwear man",  # material, specific cloth, gender
        "yellow black t-shirt women xl",  # adjectives, specific cloth, gender, size
        "casual comfortable blue trousers women",  # context, adjectives, specific cloth, gender
        "breathable sports clothes winter",
    ]  # adjectives, context, general clothes, context
    for query in queries:
        print(f"\n\033[92mQUERY: {query}\033[0m")

        '''STEP 1: filter the products'''
        preprocessed_query, filtered_doc_embeddings = filter(query=query, doc_embeddings=doc_embeddings, product_sentences=product_sentences)
        if (len(filtered_doc_embeddings) == 0):   # Compute ranking only if we found documents during filtering
            print("\n\033[91mNo results!\033[0m")
        else:
            '''STEP 2: rank the filtered products'''
            scores_w2vcossim = rank_documents(word2vec_model, filtered_doc_embeddings, preprocessed_query)
            print_ranking(scores=scores_w2vcossim, index2title=index2title)


