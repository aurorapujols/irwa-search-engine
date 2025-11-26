import random
import numpy as np

from myapp.search.objects import Document
from myapp.search.algorithms import *
from project_progress.part_3.ranking import get_index_and_metrics
from project_progress.part_3.word2vec import *


def dummy_search(corpus: dict, search_id, num_results=20):
    """
    Just a demo method, that returns random <num_results> documents from the corpus
    :param corpus: the documents corpus
    :param search_id: the search id
    :param num_results: number of documents to return
    :return: a list of random documents from the corpus
    """
    res = []
    doc_ids = list(corpus.keys())
    docs_to_return = np.random.choice(doc_ids, size=num_results, replace=False)
    for doc_id in docs_to_return:
        doc = corpus[doc_id]
        res.append(Document(pid=doc.pid, title=doc.title, description=doc.description,
                            url="doc_details?pid={}&search_id={}&param2=2".format(doc.pid, search_id), ranking=random.random()))
    return res

def get_search_results(corpus:dict, results, search_id, num_results=20):
    res = []
    results = np.array(results)
    docs_to_return = results[:min(len(results), num_results), 1]  # get up to num_results products, and get only the pids (not scores as well)
    for rank, doc_id in enumerate(docs_to_return):
        doc = corpus[doc_id]
        res.append(Document(pid=doc.pid, title=doc.title, description=doc.description, selling_price=doc.selling_price, discount=doc.discount,
                            average_rating=doc.average_rating, out_of_stock=doc.out_of_stock, category=doc.category, brand=doc.brand, 
                            sub_category=doc.sub_category, product_details=doc.product_details, url=doc.url,
                            page_url="doc_details?pid={}&search_id={}&param2=2".format(doc.pid, search_id), ranking=rank+1))
    return res


class SearchEngine:
    """Class that implements the search engine logic"""

    def __init__(self, corpus):
        """
        Initializes the search engine with a corpus and computes the TF-IDF matrix.
        
        Args:
            corpus (list of str): List of documents.
        """
        self.corpus = corpus
        self.tf_idf_index, _, self.tf_tfidf, _, self.idf_tfidf = get_index_and_metrics(corpus)
        self.weights_tf_idf = {
            "title_description": 0.5,
            "brand": 0.05,
            "category": 0.2,
            "sub_category": 0.2,
            "seller_product_details": 0.05,
        }
        self.products_dict = process_products(corpus)
        self.metadata = get_numerical_info_by_product(corpus)
        _, self.idf_bm25, self.Ld, self.Lave = compute_prod_data(self.products_dict)

        self.training_sentences = get_training_sentences(self.products_dict)
        self.word2vec_model = train_word2vec_model(self.training_sentences)
        self.prod_embeddings = get_document_embeddings(self.word2vec_model, self.products_dict)

    def search(self, search_query, search_id, search_type='tf-idf'):
        print("Search query:", search_query)

        rankings = []

        # WITH FILTERING:
        # query_terms, filtered_prod = filter(search_query, self.products_dict)

        # WITHOUT FILTERING:
        query_terms = build_terms(search_query)
        filtered_prod = [pid for pid in self.products_dict.keys()]    # all of the products
        
        results = []
        ### You should implement your search logic here:
        # results = dummy_search(corpus, search_id)
        if(query_terms and filtered_prod):
            match search_type:
                case 'tf-idf':
                    rankings = get_tf_idf_ranking(query_terms, filtered_prod, self.tf_idf_index, self.tf_tfidf, self.idf_tfidf, self.weights_tf_idf)
                case 'bm25':
                    rankings = get_bm25_ranking(query_terms, self.products_dict, filtered_prod, self.idf_bm25, self.Ld, self.Lave)
                case 'custom':
                    rankings = get_custom_ranking(query_terms, self.metadata, self.products_dict, filtered_prod, self.idf_bm25, self.Ld, self.Lave)
                case 'word2vec':
                    rankings = get_word2vec_ranking(search_query, self.word2vec_model, self.prod_embeddings, filtered_prod, preprocess=preprocess_string)
            
            results = get_search_results(corpus=self.corpus, results=rankings, search_id=search_id)
        # results = search_in_corpus(search_query)
        return results
