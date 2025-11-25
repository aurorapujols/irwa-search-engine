from project_progress.part_3.ranking import *
from project_progress.part_3.word2vec import *
from gensim.parsing.preprocessing import preprocess_string

def get_tf_idf_ranking(query_terms, filtered_prod, tf_idf_index, tf, idf, weights_tf_idf):
    scores_tf_idf = rank_tf_idf_cosine_similarity(query_terms=query_terms,
                                                  products=filtered_prod,
                                                  index=tf_idf_index,
                                                  tf=tf,
                                                  idf=idf,
                                                  weights=weights_tf_idf)
    return scores_tf_idf

def get_bm25_ranking(query_terms, products, filtered_prod, idf, Ld, Lave, k1=1.2, b=0.75):
    scores_BM25 = rank_BM25(query_terms=query_terms,
                            products=products,
                            filtered_prod=filtered_prod,
                            idf=idf,
                            Ld=Ld,
                            Lave=Lave,
                            k1=k1,
                            b=b)
    return scores_BM25

def get_custom_ranking(query_terms, numeric_data, products, filtered_prod, idf, Ld, Lave, k1=1.2, b=0.75, \
                        lambda_=0.3, popularity=0.3, affordability=0.7, availability=0.01):
    scores_custom = rank_custom(query_terms=query_terms,
                                        metadata=numeric_data,
                                        products=products,
                                        filtered_prod=filtered_prod,
                                        idf=idf,
                                        Ld=Ld,
                                        Lave=Lave,
                                        k1=k1,
                                        b=b,
                                        lambda_=lambda_,
                                        popularity=popularity,
                                        affordability=affordability,
                                        availability=availability)
    return scores_custom

def get_word2vec_ranking(search_query, word2vec_model, product_embeddings, filtered_prods, preprocess=preprocess_string):
    filtered_prod_embeddings = {pid: product_embeddings[pid] for pid in filtered_prods}
    word2vec_rankings = rank_documents(w2v_model=word2vec_model, 
                                       doc2vec=filtered_prod_embeddings,
                                       query=search_query,
                                       preprocess=preprocess)
    return word2vec_rankings