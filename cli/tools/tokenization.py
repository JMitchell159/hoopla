import string

def tokenize(text, stop_words: list[str], translator, stemmer) -> list[str]:
    split = text.lower().translate(translator).split()
    result = []
    for s in split:
        if len(s) > 0 and s not in stop_words:
            result.append(stemmer.stem(s))
    return result
