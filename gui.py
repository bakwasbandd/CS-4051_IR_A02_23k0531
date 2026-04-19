import tkinter as tk
from indexer import load_indexes
from preprocessor import load_stopwords
from vsm import compute_tf_idf, build_doc_vectors, rank_documents
from queryProcessor import process_query

BG         = "#fdf6f9"
PANEL      = "#ffffff"
PINK       = "#f4c0d1"
PINK_DARK  = "#d4537e"
GREEN      = "#c0dd97"
GREEN_DARK = "#3b6d11"
TEXT       = "#2c2c2a"
SUBTEXT    = "#888780"
BORDER     = "#edcfda"
ENTRY_BG   = "#fff0f5"

inv, pos, doc_mapping, doc_freq = load_indexes()
stopwords = load_stopwords("Stopword-List.txt")

N = len(doc_mapping)

# compute tf-idf and doc vectors ONCE (important optimization)
tf_idf = compute_tf_idf(inv, doc_freq, N)
doc_vectors = build_doc_vectors(tf_idf)


#search function triggered by button click or Enter key
def search(event=None):
    q = entry.get().strip()
    if not q:
        return

    # preprocess query (same pipeline as docs)
    processed_terms = process_query(q, stopwords)

    if not processed_terms:
        count_label.config(text="Empty query after preprocessing", fg=PINK_DARK)
        return

    processed_query = " ".join(processed_terms)

    # VSM ranking
    results = rank_documents(processed_query, doc_vectors, doc_freq, N)

    # clear old results
    for widget in results_frame.winfo_children():
        widget.destroy()

    if results:
        count_label.config(
            text=f"{len(results)} relevant document(s)",
            fg=GREEN_DARK
        )

        # show top 10 ranked results
        for i, (doc_id, score) in enumerate(results[:10]):
            row = tk.Frame(results_frame, bg=PANEL, padx=10, pady=6,
                           highlightbackground=BORDER, highlightthickness=1)
            row.pack(fill="x", pady=2)

            # rank number
            tk.Label(row, text=f"{i+1}.", bg=PANEL, fg=PINK_DARK,
                     font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 8))

            # filename
            tk.Label(row, text=doc_mapping[doc_id], bg=PANEL,
                     fg=TEXT, font=("Segoe UI", 10)).pack(side="left")

            # similarity score
            tk.Label(row, text=f"{score:.4f}", bg=PANEL,
                     fg=GREEN_DARK, font=("Segoe UI", 9, "bold")).pack(side="right")

    else:
        count_label.config(text="No relevant documents found", fg=PINK_DARK)
        tk.Label(results_frame, text="Try a different query.",
                 bg=BG, fg=SUBTEXT, font=("Segoe UI", 10)).pack(pady=8)


# GUI setup
root = tk.Tk()
root.title("Vector Space Model IR System")
root.geometry("620x560")
root.configure(bg=BG)
root.resizable(True, True)


tk.Label(root, text="VSM IR System", bg=BG, fg=PINK_DARK,
         font=("Segoe UI", 18, "bold")).pack(anchor="w", padx=28, pady=(24, 2))

tk.Label(root, text="Trump Speeches · Ranked Retrieval (TF-IDF + Cosine Similarity)",
         bg=BG, fg=SUBTEXT, font=("Segoe UI", 9)).pack(anchor="w", padx=28)

tk.Frame(root, bg=BORDER, height=1).pack(fill="x", padx=28, pady=14)


search_frame = tk.Frame(root, bg=BG)
search_frame.pack(fill="x", padx=28)

entry = tk.Entry(search_frame, bg=ENTRY_BG, fg=TEXT, font=("Segoe UI", 12),
                 relief="flat", highlightthickness=1,
                 highlightbackground=BORDER, highlightcolor=PINK,
                 insertbackground=PINK_DARK)
entry.pack(side="left", fill="x", expand=True, ipady=8, ipadx=8)
entry.bind("<Return>", search)

btn = tk.Button(search_frame, text="Search", bg=PINK, fg=PINK_DARK,
                font=("Segoe UI", 10, "bold"), relief="flat",
                bd=0, padx=16, pady=8, cursor="hand2",
                activebackground=GREEN, activeforeground=GREEN_DARK,
                command=search)
btn.pack(side="left", padx=(8, 0))

tk.Label(root, text='e.g.  "america"   |   "hillary clinton"   |   "energy policy"',
         bg=BG, fg=SUBTEXT, font=("Segoe UI", 8)).pack(anchor="w", padx=28, pady=(6, 0))

tk.Frame(root, bg=BORDER, height=1).pack(fill="x", padx=28, pady=14)


results_header = tk.Frame(root, bg=BG)
results_header.pack(fill="x", padx=28, pady=(0, 8))

tk.Label(results_header, text="Ranked Results", bg=BG, fg=SUBTEXT,
         font=("Segoe UI", 9)).pack(side="left")

count_label = tk.Label(results_header, text="", bg=BG,
                       font=("Segoe UI", 9))
count_label.pack(side="right")


canvas = tk.Canvas(root, bg=BG, highlightthickness=0)
scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
canvas.configure(yscrollcommand=scrollbar.set)

scrollbar.pack(side="right", fill="y", padx=(0, 8))
canvas.pack(fill="both", expand=True, padx=28, pady=(0, 20))

results_frame = tk.Frame(canvas, bg=BG)
canvas_window = canvas.create_window((0, 0), window=results_frame, anchor="nw")

results_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width))

root.mainloop()