import string

def tokenize(text, stop_words, translator, stemmer):
    split = text.lower().translate(translator).split()
    result = []
    for s in split:
        if len(s) > 0 and s not in stop_words:
            result.append(stemmer.stem(s))
    return result
