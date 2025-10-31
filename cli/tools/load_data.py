import json
import string

def load_movies():
    movies = None
    with open("data/movies.json") as movie_file:
        movies = json.load(movie_file)
    return movies

def load_stop_words():
    stop_words = []
    with open("data/stopwords.txt") as stop_file:
        content = stop_file.read()
        stop_words = content.splitlines()
    return stop_words

def load_translator():
    punc_map = {}
    for p in string.punctuation:
        punc_map[p] = None
    translator = str.maketrans(punc_map)
    return translator
