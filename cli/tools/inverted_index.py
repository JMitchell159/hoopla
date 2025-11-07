from tools.load_data import load_stop_words, load_translator, load_movies, BM25_K1, BM25_B
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
        self.doc_lengths: dict[int, int] = {}
    
    def __add_document(self, doc_id: int, text: str) -> None:
        words = tokenize(text, self.stop_words, self.translator, self.stemmer)
        self.doc_lengths[doc_id] = len(words)
        for word in set(words):
            self.index[word].add(doc_id)
        self.term_frequencies[doc_id].update(words)
    
    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)
    
    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))
    
    def bm25_search(self, query: str, limit: int) -> list[tuple]:
        words = tokenize(query, self.stop_words, self.translator, self.stemmer)
        scores: dict[int, float] = {}
        for word in words:
            movies = self.get_documents(word)
            for m in movies:
                if m in scores:
                    scores[m] += self.bm25(m, word)
                else:
                    scores[m] = self.bm25(m, word)
        sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        result = []
        i = 0
        for s in sorted_scores:
            result.append((self.docmap[s], sorted_scores[s]))
            i += 1
            if i == limit:
                break
        return result
    
    def get_tf(self, doc_id: int, term: str) -> int:
        words = tokenize(term, self.stop_words, self.translator, self.stemmer)
        if len(words) != 1:
            raise ValueError("term must be a single token")
        word = words[0]
        return self.term_frequencies[doc_id][word]
    
    def get_bm25_tf(self, doc_id: int, term: str, k1 = BM25_K1, b = BM25_B) -> float:
        words = tokenize(term, self.stop_words, self.translator, self.stemmer)
        if len(words) != 1:
            raise ValueError("term must be a single token")
        word = words[0]
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        tf = self.term_frequencies[doc_id][word]
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)
    
    def get_idf(self, term: str) -> float:
        words = tokenize(term, self.stop_words, self.translator, self.stemmer)
        if len(words) != 1:
            raise ValueError("term must be a single token")
        word = words[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[word])
        return math.log((doc_count + 1) / (term_doc_count + 1))
    
    def get_bm25_idf(self, term: str) -> float:
        words = tokenize(term, self.stop_words, self.translator, self.stemmer)
        if len(words) != 1:
            raise ValueError("term must be a single token")
        word = words[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[word])
        return math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)
    
    def get_tf_idf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf
    
    def bm25(self, doc_id: int, term: str) -> float:
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)
        return tf * idf
    
    def build(self) -> None:
        for m in self.movies["movies"]:
            self.docmap[m["id"]] = m
            self.__add_document(m["id"], f"{m['title']} {m['description']}")
    
    def save(self) -> None:
        if not os.path.isdir("cache"):
            os.mkdir("cache")
        with open("cache/index.pkl", "wb") as index_file:
            pickle.dump(self.index, index_file)
        with open("cache/docmap.pkl", "wb") as docmap_file:
            pickle.dump(self.docmap, docmap_file)
        with open("cache/term_frequencies.pkl", "wb") as tf_file:
            pickle.dump(self.term_frequencies, tf_file)
        with open("cache/doc_lengths.pkl", "wb") as doc_lengths_file:
            pickle.dump(self.doc_lengths, doc_lengths_file)
    
    def load(self) -> None:
        if not os.path.exists("cache/index.pkl") or not os.path.exists("cache/docmap.pkl") or not os.path.exists("cache/term_frequencies.pkl") or not os.path.exists("cache/doc_lengths.pkl"):
            raise FileNotFoundError()
        with open("cache/index.pkl", "rb") as index_file:
            self.index = pickle.load(index_file)
        with open("cache/docmap.pkl", "rb") as docmap_file:
            self.docmap = pickle.load(docmap_file)
        with open("cache/term_frequencies.pkl", "rb") as tf_file:
            self.term_frequencies = pickle.load(tf_file)
        with open("cache/doc_lengths.pkl", "rb") as doc_lengths_file:
            self.doc_lengths = pickle.load(doc_lengths_file)
