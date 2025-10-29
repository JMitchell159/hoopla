import json
import string

def kw_search(query):
    movies = None
    stop_words = []
    with open("data/movies.json") as movie_file:
        movies = json.load(movie_file)
    with open("data/stopwords.txt") as stop_file:
        content = stop_file.read()
        stop_words = content.splitlines()
    punc_map = {}
    for p in string.punctuation:
        punc_map[p] = None
    translator = str.maketrans(punc_map)
    results = []
    split = query.lower().translate(translator).split(" ")
    refined = []
    idx = 0
    for s in split:
        if len(s) > 1 and s not in stop_words:
            refined.append(s)
            idx += 1
    for m in movies["movies"]:
        for word in refined:
            if word in m["title"].lower().translate(translator):
                results.append(m)
                break
    results.sort(key=lambda x: x["id"])
    return results[:5]
