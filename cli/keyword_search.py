import json
import string
from nltk.stem import PorterStemmer
from tools.tokenization import tokenize
from tools.inverted_index import InvertedIndex
import os

def kw_search(query):
    inv_idx = InvertedIndex()
    try:
        inv_idx.load()
    except FileNotFoundError:
        print("index and/or docmap files do not exist.")
        os.exit(1)
    tokens = tokenize(query, inv_idx.stop_words, inv_idx.translator, inv_idx.stemmer)
    idx_result = []
    done = False
    for t in tokens:
        doc_ids = inv_idx.get_documents(t)
        if doc_ids is not None:
            for idx in doc_ids:
                if idx not in idx_result:
                    idx_result.append(idx)
                if len(idx_result) == 5:
                    done = True
                    break
            if done:
                idx_result.sort()
                break
    result = []
    for idx in idx_result:
        result.append(inv_idx.docmap[idx])
    return result
