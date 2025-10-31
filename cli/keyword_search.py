import json
import string
from nltk.stem import PorterStemmer
from tools.tokenization import tokenize

def kw_search(query):
    movies = None
    stop_words = []
    with open("data/movies.json") as movie_file:
        movies = json.load(movie_file)
    with open("data/stopwords.txt") as stop_file:
        content = stop_file.read()
        stop_words = content.splitlines()
    stemmer = PorterStemmer()
    punc_map = {}
    for p in string.punctuation:
        punc_map[p] = None
    translator = str.maketrans(punc_map)
    results = []
    refined = tokenize(query, stop_words, translator, stemmer)
    for m in movies["movies"]:
        m_refined = tokenize(m["title"], stop_words, translator, stemmer)
        movie = " ".join(m_refined)
        for word in refined:
            if word in movie:
                results.append(m)
                break
    results.sort(key=lambda x: x["id"])
    return results[:5]
