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
from project_progress.part_1.data_prep import corpus_df_loading
from project_progress.part_2.index_tf_idf import get_index_and_metrics
from project_progress.part_3.ranking import (
    get_or_create,
    process_products,
    filter,
    print_ranking,
    )

from gensim.models import Word2Vec
from gensim.parsing.preprocessing import preprocess_string

load_dotenv()  # take environment variables from .env


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
products_filepath = os.path.join(BASE_DIR, "..", "..", "data", "products.json")


# Word2Vec embedding --------------------------------------------------------------------#
def get_training_sentences(products):
    """
    Function that builds the training sentences from the product description as text.

    :param products: Dictionary of pid -> product text description.
    :return sentences: List of tokenized sentences for training the Word2Vec model.
    """
    sentences = []

    for prod_sentences in products.values():
        sentences.extend(prod_sentences)

    return sentences

def train_word2vec_model(sentences, vector_size=100, window=7, min_count=5, negative=10, sg=1):
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

    return model

def build_text_terms(sentences):
    """
    Function that builds a list of terms from the given sentences.
    
    :param sentences: List of tokenized sentences.
    :return term_list: List of terms.
    """
    term_list = []
    for sentence in sentences:
        term_list.extend(sentence)

    return term_list

def get_embedding(text, model:Word2Vec):
    """
    Function that computes the embedding of a document by averaging the embeddings of its terms.

    :param text: List of tokenized sentences representing the document.
    :param model: Trained Word2Vec model.
    :return embedding: Numpy array representing the document embedding.
    """
    doc_terms = build_text_terms(text)
    word_embeddings = [model.wv[word] if word in model.wv else np.zeros(model.vector_size) for word in doc_terms]

    embedding = np.mean(word_embeddings, axis=0)
    return embedding

def get_document_embeddings(word2vec_model, products):
    """
    Function that computes the document embeddings for all products.
    
    :param word2vec_model: Trained Word2Vec model.
    :param products: Dictionary of pid -> text description.
    :return doc_embeddings: Dictionary of pid -> document embedding.
    """
    doc_embeddings = {}
    for pid, sentences in products.items():
        doc_embeddings[pid] = get_embedding(sentences, word2vec_model)
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

def rank_documents(w2v_model, doc2vec, query, preprocess=preprocess_string):
    """
    Function that ranks documents based on their cosine similarity to the query.
    :param w2v_model: Trained Word2Vec model.
    :param doc2vec: Dictionary of pid -> document embedding.
    :param query: Query string.
    :param preprocess: Preprocessing function to apply to the query.
    :return sim_scores: List of [similarity score, pid] sorted in descending order
    """
    query_terms = preprocess_string(query)
    query_embedding = get_embedding(query_terms, w2v_model)

    sim_scores = []
    for pid, doc_embedding in doc2vec.items():
        score = cosine_similarity(doc_embedding,query_embedding)
        sim_scores.append([score, pid])
    
    sim_scores.sort(key = lambda x: x[0], reverse=True)

    return sim_scores

# ---------------------------------------------------------------------------------------#

if __name__ == "__main__":

    # Load the corpus
    print("\033[34mLoading corpus...\033[0m")
    json_path = os.getenv("DATA_FILE_PATH")
    corpus = corpus_df_loading(json_path)
    print("\033[34mDONE! Starting preprocessing...\033[0m")

    # Preprocess the products and precompute the inverted index
    products = get_or_create(products_filepath, lambda: process_products(corpus))
    print("\033[34mDONE! Computing metrics...\033[0m")
    index, index2title, tf, df, idf = get_index_and_metrics(corpus)
    # Build the embedding space and compute document embeddings
    print("\033[34mDONE! Creating embedding space...\033[0m")
    training_sentences = get_training_sentences(products)
    word2vec_model = train_word2vec_model(training_sentences)
    print("\033[34mDONE! Generating product embeddings...\033[0m")
    doc_embeddings = get_document_embeddings(word2vec_model, products)
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
        _, filtered_docs = filter(query=query, products=products)
        filtered_doc_embeddings = {pid: doc_embeddings[pid] for pid in filtered_docs}
        if (len(filtered_doc_embeddings) == 0):   # Compute ranking only if we found documents during filtering
            print("\n\033[91mNo results!\033[0m")
        else:
            '''STEP 2: rank the filtered products'''
            scores_w2vcossim = rank_documents(word2vec_model, filtered_doc_embeddings, query)
            print_ranking(scores=scores_w2vcossim, index2title=index2title)


