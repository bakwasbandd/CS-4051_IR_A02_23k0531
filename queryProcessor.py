import re
from nltk.stem import PorterStemmer

# query is preprocessed in the same way as documents
# (lowercase, remove punctuation, stemming) 
ps = PorterStemmer()


def process_word(word):
    # lowercase
    word = word.lower()

    # remove non-alphabetic characters
    word = re.sub(r'[^a-z]', '', word)

    # return stemmed version
    return ps.stem(word)


def process_query(query, stopwords=None):

    # preprocess full query string and return cleaned list of terms
    # cuz of this --> query representation lies in the same feature space as documents

    # extract words only (same regex as the preprocessor)
    words = re.findall(r"[a-z]+", query.lower())

    clean_terms = []

    for w in words:
        # removing stopwordsa from the query as well!
        if stopwords and w in stopwords:
            continue

        processed = process_word(w)

        if processed:  # only add non-empty terms after processing
            clean_terms.append(processed)

    return clean_terms