from tools.load_data import load_stop_words, load_translator, load_movies
from tools.tokenization import tokenize
import math
from nltk.stem import PorterStemmer
import pickle
import os
from collections import Counter, defaultdict

class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies = defaultdict(Counter)
        self.stop_words = load_stop_words()
        self.translator = load_translator()
        self.stemmer = stemmer = PorterStemmer()
        self.movies = load_movies()
    
    def __add_document(self, doc_id, text):
        words = tokenize(text, self.stop_words, self.translator, self.stemmer)
        for word in set(words):
            self.index[word].add(doc_id)
        self.term_frequencies[doc_id].update(words)
    
    def get_documents(self, term):
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))
    
    def get_tf(self, doc_id: str, term: str) -> int:
        words = tokenize(term, self.stop_words, self.translator, self.stemmer)
        if len(words) != 1:
            raise ValueError("term must be a single token")
        word = words[0]
        return self.term_frequencies[doc_id][word]
    
    def get_idf(self, term: str) -> float:
        words = tokenize(term, self.stop_words, self.translator, self.stemmer)
        if len(words) != 1:
            raise ValueError("term must be a single token")
        word = words[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[word])
        return math.log((doc_count + 1) / (term_doc_count + 1))
    
    def build(self):
        for m in self.movies["movies"]:
            self.docmap[m["id"]] = m
            self.__add_document(m["id"], f"{m['title']} {m['description']}")
    
    def save(self):
        if not os.path.isdir("cache"):
            os.mkdir("cache")
        with open("cache/index.pkl", "wb") as index_file:
            pickle.dump(self.index, index_file)
        with open("cache/docmap.pkl", "wb") as docmap_file:
            pickle.dump(self.docmap, docmap_file)
        with open("cache/term_frequencies.pkl", "wb") as tf_file:
            pickle.dump(self.term_frequencies, tf_file)
    
    def load(self):
        if not os.path.exists("cache/index.pkl") or not os.path.exists("cache/docmap.pkl") or not os.path.exists("cache/term_frequencies.pkl"):
            raise FileNotFoundError()
        with open("cache/index.pkl", "rb") as index_file:
            self.index = pickle.load(index_file)
        with open("cache/docmap.pkl", "rb") as docmap_file:
            self.docmap = pickle.load(docmap_file)
        with open("cache/term_frequencies.pkl", "rb") as tf_file:
            self.term_frequencies = pickle.load(tf_file)
