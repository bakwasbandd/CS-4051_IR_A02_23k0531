import math
from collections import defaultdict
from queryProcessor import process_word

def compute_tf_idf(inverted_index, doc_freq, N): # N = total number of documents
    tf_idf = {}

    for term, postings in inverted_index.items():
        df = doc_freq[term]  # number of documents containing the term
        idf = math.log(N / df) # inverse document frequency ==> log(N/df) --> higher for rarer terms 

        tf_idf[term] = {} # term -> {doc_id: tf-idf weight} mapping
        for doc_id, tf in postings.items():
            tf_idf[term][doc_id] = tf * idf

    return tf_idf


def build_doc_vectors(tf_idf): # build document vectors from tf-idf weights for cosine similarity
    doc_vectors = defaultdict(dict) 

    for term, postings in tf_idf.items():
        for doc_id, weight in postings.items():
            doc_vectors[doc_id][term] = weight # doc_id -> {term: tf-idf     weight} mapping for each document

    return doc_vectors


def build_query_vector(query, doc_freq, N):
    terms = query.split()
    tf = {}
# compute term frequency for the query (after processing)
    for t in terms:
        t = process_word(t)
        tf[t] = tf.get(t, 0) + 1

    query_vec = {}
# compute tf-idf for query terms using the same idf as documents
    for term, freq in tf.items():
        if term in doc_freq:
            idf = math.log(N / doc_freq[term])
            query_vec[term] = freq * idf

    return query_vec


def cosine_similarity(vec1, vec2):
    # between two sparse vectors
    dot = 0
    for term in vec1:
        if term in vec2:
            dot += vec1[term] * vec2[term]
    # compute magnitudes
    norm1 = math.sqrt(sum(v*v for v in vec1.values()))
    norm2 = math.sqrt(sum(v*v for v in vec2.values()))

    if norm1 == 0 or norm2 == 0:
        return 0
    # cosine similarity = dot product / (magnitude of vec1 * magnitude of vec2)
    return dot / (norm1 * norm2)

#   rank documents based on cosine similarity with the query vector, and return sorted results above a certain threshold (alpha)
def rank_documents(query, doc_vectors, doc_freq, N, alpha=0.005):
    query_vec = build_query_vector(query, doc_freq, N)

    scores = []

    for doc_id, doc_vec in doc_vectors.items():
        sim = cosine_similarity(query_vec, doc_vec)
        if sim >= alpha: # only consider documents with similarity above score threshold alpha
            scores.append((doc_id, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores