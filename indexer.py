import os
import pickle
from preprocessor import preprocess

def build_indexes(folder, stopwords):

    # inverted index: word -> {doc_id: tf}
    # instead of just storing which documents contain the word,
    # we now also store how many times it appears in each document (term frequency)
    inverted_index = {}

    # positional index: word -> {doc_id: [positions]}
    positional_index = {}

    # doc_mapping: doc_id -> filename
    # mapping of doc ids to their filenames for easy retrieval during search results
    doc_mapping = {}

    files = sorted(os.listdir(folder))  # sorted for consistency

    for doc_id, filename in enumerate(files):

        if not filename.endswith('.txt'):  # skip non-text files if any
            continue

        path = os.path.join(folder, filename)
        doc_mapping[doc_id] = filename

        # preprocess returns list of (word, position)
        tokens = preprocess(path, stopwords)

        for word, pos in tokens:
            # if the word is not in the index, initialize it
            if word not in inverted_index:
                inverted_index[word] = {}

            # if the word has not appeared in this document before, initialize tf = 0
            if doc_id not in inverted_index[word]:
                inverted_index[word][doc_id] = 0

            # increment term frequency count
            inverted_index[word][doc_id] += 1

            # this is for positional indexes! 
            if word not in positional_index:
                positional_index[word] = {}

            if doc_id not in positional_index[word]:
                positional_index[word][doc_id] = []

            # store position of the word in the document
            positional_index[word][doc_id].append(pos)



    # df = number of documents a term appears in
    # compute this by counting the number of doc ids in each posting list
    doc_freq = {}

    for term, postings in inverted_index.items():
        doc_freq[term] = len(postings)

    return inverted_index, positional_index, doc_mapping, doc_freq


# save all indexes including new doc_freq
def save_indexes(inv, pos, doc_mapping, doc_freq):

    with open("inverted_index.pkl", "wb") as f:
        pickle.dump(inv, f)

    with open("positional_index.pkl", "wb") as f:
        pickle.dump(pos, f)

    with open("doc_mapping.pkl", "wb") as f:
        pickle.dump(doc_mapping, f)

    # saving document frequency separately for tf-idf computation later
    with open("doc_freq.pkl", "wb") as f:
        pickle.dump(doc_freq, f)

    print("Indexes saved successfully.")


# load all indexes including doc_freq
def load_indexes():
    with open("inverted_index.pkl", "rb") as f:
        inv = pickle.load(f)

    with open("positional_index.pkl", "rb") as f:
        pos = pickle.load(f)

    with open("doc_mapping.pkl", "rb") as f:
        doc_mapping = pickle.load(f)

    with open("doc_freq.pkl", "rb") as f:
        doc_freq = pickle.load(f)

    return inv, pos, doc_mapping, doc_freq