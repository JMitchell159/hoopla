import json

def kw_search(query):
    movie_file = open("data/movies.json")
    movies = json.load(movie_file)
    results = []
    for m in movies["movies"]:
        if query.lower() in m["title"].lower():
            results.append(m)
    results.sort(key=lambda x: x["id"])
    return results[:5]
