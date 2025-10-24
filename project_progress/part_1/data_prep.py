import os
import sys
import random
import string
import json

import  nltk
nltk.download('stopwords')

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from myapp.search.load_corpus import load_corpus
from collections import defaultdict
from array import array

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env


def corpus_df_loading(path):
    """
    In this function we load the corpus as dataframe, and we preprocess the numerical fields.

    :param path: Path to the json file.
    :return corpus: Returns a dictionary List[Document] with the loaded corpus with the numerical fields preprocessed.
    """
    corpus = load_corpus(path)
    return corpus


def build_terms(text):
    """
    Preprocesses the text fields of the document in the corpus (only`title` and `description`) by removing stop words, tokenizing, removing punctuation marks, stemming and [#TODO].

    :param text: (string) text to be processed
    :return text: List of tokens corresponding to the input text after the preprocessing
    """

    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    text = text.lower()  # Transform to lowercase
    text = text.split(" ")  # Tokenize the text, separating by spaces
    text = [
        word.strip(string.punctuation)
        for word in text
        if word.strip(string.punctuation).isalnum() and word not in stop_words
    ]  # eliminate the stop words [`strip` is to separate exclamation signs from words, e.g. "Hello!"" -> "Hello" + "!"]
    text = [stemmer.stem(word) for word in text]  # perform stemming

    # TODO: add more preprocessing if necessary

    return text

def join_build_terms(strings):
    """
    Builds the terms by concatenating the strings in the given list.

    :param strings: List of string. Texts to be concatenated and processed.
    :return terms: List of string, where each item is a word.
    """
    arg = " ".join(strings)
    return build_terms(arg)


def get_articles_info(corpus):
    """
    Get the information from the corpus and return two dictionaries with the info of each one.

    :param corpus: corpus (collection of products' documents)
    :return: two Dict -> `metadata` with the numerical fields in the corpus, and `index_info` with the categorical fields
    """

    # Initialize variables
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

    return metadata, info_index



if __name__ == "__main__":

    # Load the corpus
    json_path = os.getenv("DATA_FILE_PATH")
    corpus = corpus_df_loading(json_path)

    # Get the info of each article
    metadata, info_index = get_articles_info(corpus)

    # Output in json files
    with open("metadata_dict.json", "w") as json_file:
        json.dump(metadata, json_file, indent=2)

    with open("info_index_dict.json", "w") as json_file:
        json.dump(info_index, json_file, indent=2)
