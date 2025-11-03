from tools.load_data import load_stop_words, load_translator, load_movies
from tools.tokenization import tokenize
from nltk.stem import PorterStemmer
import pickle
import os

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.stop_words = load_stop_words()
        self.translator = load_translator()
        self.stemmer = stemmer = PorterStemmer()
        self.movies = load_movies()
    
    def __add_document(self, doc_id, text):
        words = tokenize(text, self.stop_words, self.translator, self.stemmer)
        for word in words:
            if word in self.index:
                self.index[word].append(doc_id)
            else:
                self.index[word] = [doc_id,]
            self.index[word].sort()
    
    def get_documents(self, term):
        if term in self.index:
            return self.index[term.lower()]
        return None
    
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
    
    def load(self):
        if not os.path.exists("cache/index.pkl") or not os.path.exists("cache/docmap.pkl"):
            raise FileNotFoundError()
        with open("cache/index.pkl", "rb") as index_file:
            self.index = pickle.load(index_file)
        with open("cache/docmap.pkl", "rb") as docmap_file:
            self.docmap = pickle.load(docmap_file)
