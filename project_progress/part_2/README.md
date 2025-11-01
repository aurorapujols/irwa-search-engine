# PART 2: Indexing and Evaluation

#TODO: introduction

## Functions Description

### 1. ``create_index_tf_idf(corpus)``

**Description**: Constructs an inverted index and computes term frequency (TF), document frequency (DF), and inverse document frequency (IDF) for each term across multiple fields.

**Parameters**:
+ ``corpus`` (dict): Dictionary of product documents.

**Returns**:
+ ``index`` (dict): Inverted index with term positions per field.
+ ``index2title`` (dict): Mapping from product ID to title.
+ ``tf`` (dict): Term frequency per document, term, and field.
+ ``df`` (dict): Document frequency per term.
+ ``idf`` (dict): Inverse document frequency per term.

Example:

```python
index, index2title, tf, df, idf = create_index_tf_idf(corpus)
```

### 2. ``filter(query, index)``

**Description**: Filters the corpus to return only documents that contain all query terms.

**Parameters**:
+ ``query`` (str): User query string.
+ ``index`` (dict): Inverted index.

**Returns**:
+ ``query_terms`` (list): Preprocessed query terms.
+ ``docs`` (list): List of product IDs matching all query terms.

Example:

```python
query_terms, docs = filter("women purple jeans", index)
```

### 3. ``rank_tf_idf(query_terms, products, index, tf, idf, weights)``

**Description**: Ranks the filtered documents using TF-IDF scores across multiple fields, weighted by field importance.

**Parameters**:
+ ``query_terms`` (list): List of query terms.
+ ``products`` (list): List of product IDs to rank.
+ ``index``, ``tf``, ``idf``: Indexing structures.
+ ``weights`` (dict): Field weights for ranking.

**Returns**:
+ ``product_scores`` (list): List of [score, product_id] sorted by relevance.

Example:

```python
scores = rank_tf_idf(query_terms, docs, index, tf, idf, weights)
```

### 4. ``engine_search(query, index)``

**Description**: Combines filtering and ranking to return the most relevant documents for a query.

**Parameters**:
+ ``query`` (str): User query string.
+ ``index`` (dict): Inverted index.

**Returns**:
+ ``scores`` (list): Ranked list of [score, product_id].

Example:

```python
results = engine_search("women purple jeans", index)

```

### 5. ``precision_at_k(query_results, k=10)``

**Description**: Computes the precision at rank 𝑘, i.e., the proportion of relevant documents among the top 𝑘 retrieved.

**Parameters**:
+ ``query_results`` (DataFrame): Sorted search results for a query.
+ ``k`` (int): Number of top documents to consider.

**Returns**:
+ ``precision`` (float): Precision@k value.

### 6. ``recall_at_k(query_results, k=10)``
**Description**: Computes the recall at rank 𝑘, i.e., the proportion of relevant documents retrieved among all relevant ones.

**Parameters**:
+ ``query_results`` (DataFrame): Sorted search results.
+ ``k`` (int): Number of top documents to consider.

**Returns**:
+ ``recall`` (float): Recall@k value.

### 7. ``avg_precision_at_k(query_results, k)``

**Description**: Computes the average precision at 𝑘, approximating the area under the precision-recall curve.

**Parameters**:
+ ``query_results`` (DataFrame): Search results.
+ ``k`` (int): Number of top documents to consider.

**Returns**:
+ ``avg_precision`` (float): Average precision@k.

### 8. ``f1_score_at_k(query_results, k)``

**Description**: Computes the harmonic mean of precision and recall at 𝑘.

**Parameters**:
+ ``query_results`` (DataFrame): Search results.
+ ``k`` (int): Rank threshold.

**Returns**:
+ ``f1_score`` (float): F1 score@k.

### 9. ``map_at_k(results, k)``

**Description**: Computes the Mean Average Precision (MAP) across multiple queries.

**Parameters**:
+ ``results`` (DataFrame): Combined search results for all queries.
+ ``k`` (int): Rank threshold.

**Returns**:
+ ``map`` (float): Mean average precision@k.

### 10. ``rr_at_k(query_results, k)``

**Description**: Computes the Reciprocal Rank (RR) for a single query.

**Parameters**:
+ ``query_results`` (DataFrame): Search results.
+ ``k`` (int): Rank threshold.

**Returns**:
+ ``rr`` (float): Reciprocal rank@k.

### 11. ``mrr(results, k)``

**Description**: Computes the Mean Reciprocal Rank (MRR) across multiple queries.

**Parameters**:
+ ``results`` (DataFrame): Combined search results.
+ ``k`` (int): Rank threshold.

**Returns**:
+ ``mrr`` (float): Mean reciprocal rank.

### 12. ``dcg_at_k(query_results, k)``

**Description**: Computes the Discounted Cumulative Gain (DCG) at 𝑘, measuring relevance with position-based discounting.

**Parameters**:
+ ``query_results`` (DataFrame): Search results.
+ ``k`` (int): Rank threshold.

**Returns**:
+ ``dcg`` (float): DCG@k.

### 13. ``ndcg_at_k(query_results, k)``

**Description**: Computes the Normalized DCG (NDCG) at 𝑘, comparing actual ranking to ideal ranking.

**Parameters**:
+ ``query_results`` (DataFrame): Search results.
+ ``k`` (int): Rank threshold.

**Returns**:
+ ``ndcg`` (float): NDCG@k.

### 14. ``get_precision_recall_curve(query_results)``

**Description**: Computes precision, recall, and F1 score across all ranks for plotting precision-recall curves.

**Parameters**:
+ ``query_results`` (DataFrame): Search results.

**Returns**:
+ ``precision``, ``recall``, ``f1`` (lists): Metric values at each rank.


## Code Execution

TODO(...)