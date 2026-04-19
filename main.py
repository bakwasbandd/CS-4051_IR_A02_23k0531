from preprocessor import load_stopwords
from indexer import build_indexes, save_indexes, load_indexes
from vsm import compute_tf_idf, build_doc_vectors, rank_documents
from queryProcessor import process_query

SPEECHES_FOLDER = "Speeches"
STOPWORDS_FILE = "Stopword-List.txt"
QUERIES_FILE = "testQueries.txt" 


# first build the indexes (inverted, positional, doc mapping, and doc frequency) and save them to disk!
def build():
    stopwords = load_stopwords(STOPWORDS_FILE)

    # build indexes (now includes doc_freq)
    inv, pos, doc_mapping, doc_freq = build_indexes(SPEECHES_FOLDER, stopwords)

    # save all structures
    save_indexes(inv, pos, doc_mapping, doc_freq)

    print(f"Done! Indexed {len(doc_mapping)} documents, {len(inv)} unique terms.")


# search -> preprocess query, compute tf-idf, rank documents by cosine similarity, and then we return results
def search():
    stopwords = load_stopwords(STOPWORDS_FILE)

    # load indexes
    inv, pos, doc_mapping, doc_freq = load_indexes()

    N = len(doc_mapping)  # total number of documents

    print(f"Loaded {N} documents, {len(inv)} terms.")

    # compute tf-idf and document vectors once (optimization)
    tf_idf = compute_tf_idf(inv, doc_freq, N)
    doc_vectors = build_doc_vectors(tf_idf)

    while True:
        q = input("\nEnter query (or 'exit' to stop): ").strip()

        if q.lower() == "exit":
            break

        # preprocess query
        processed_terms = process_query(q, stopwords)

        if not processed_terms:
            print("Query is empty after preprocessing.")
            continue

        # convert list back to string for ranking function
        processed_query = " ".join(processed_terms)

        # rank documents using cosine similarity
        results = rank_documents(processed_query, doc_vectors, doc_freq, N)

        if results:
            print(f"\nTop {min(10, len(results))} results:")

            for doc_id, score in results[:10]:
                print(f"  [{doc_id}] {doc_mapping[doc_id]} → {score:.4f}")

            print(f"\nLength = {len(results)}") 
            print({doc_id for doc_id, _ in results})

        else:
            print("No relevant documents found.")


# to check the given queries frm the txt file!
def run_queries_from_file():
    stopwords = load_stopwords(STOPWORDS_FILE)

    inv, pos, doc_mapping, doc_freq = load_indexes()
    N = len(doc_mapping)

    tf_idf = compute_tf_idf(inv, doc_freq, N)
    doc_vectors = build_doc_vectors(tf_idf)

    try:
        with open(QUERIES_FILE, "r") as f:
            queries = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("Queries file not found.")
        return

    for i, q in enumerate(queries, 1):
        print(f"\nQuery {i}: {q}")

        processed_terms = process_query(q, stopwords)
        processed_query = " ".join(processed_terms)

        results = rank_documents(processed_query, doc_vectors, doc_freq, N)

        if results:
            print(f"Top {min(10, len(results))} results:")
            for doc_id, score in results[:10]:
                print(f"  [{doc_id}] {doc_mapping[doc_id]} → {score:.4f}")
            # the score is printed for each document
            # it is cosine sim score between the query and document, the higher the scroe the more relevant that document is
            # also scores are ranging from 0 to 1, where 1 means perfect match and 0 means no similarity at all
            # most scores we're getting is arounf 0.06 - 0.05 --> its low because our queries are short and documents are long
            # meaning that query term vector is sparse!!
            print(f"Length = {len(results)}")

        else:
            print("No relevant documents found.")



if __name__ == "__main__":

    while True:
        choice = input(
            "\n1. Build Indexes\n"
            "2. Search (VSM)\n"
            "3. Run Queries from File\n"
            "4. Exit\n\nChoose: "
        )

        if choice == "1":
            build()

        elif choice == "2":
            search()

        elif choice == "3":
            run_queries_from_file()

        elif choice == "4":
            break

        else:
            print("Invalid choice, please enter 1, 2, 3, or 4.")