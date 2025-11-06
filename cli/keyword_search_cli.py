#!/usr/bin/env python3

import argparse
from keyword_search import kw_search
from tools.inverted_index import InvertedIndex
import math

inv_idx = InvertedIndex()

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Builds the inverted index for fast search")

    tf_parser = subparsers.add_parser("tf", help="Get the term frequency of a single-token term in a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get TF score for")

    idf_parser = subparsers.add_parser("idf", help="Get the inverse document frequency of a single-token term in the dataset")
    idf_parser.add_argument("term", type=str, help="Term to get IDF score for")

    bm25idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Gives the combined term frequency-inverse document frequency of a single-token term in a document")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to get TF-IDF score for")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            result = kw_search(args.query)
            for i, r in enumerate(result):
                print(f"{i+1}. {r["title"]}")
        case "build":
            print("Building inverted index...")
            build()
            print("Finished")
        case "tf":
            inv_idx.load()
            try:
                print(inv_idx.get_tf(args.doc_id, args.term))
            except ValueError as e:
                print(e)
        case "idf":
            inv_idx.load()
            try:
                inv_freq = idf(args.term)
                print(f"Inverse document frequency of '{args.term}': {inv_freq:.2f}")
            except ValueError as e:
                print(e)
        case "bm25idf":
            inv_idx.load()
            try:
                bm25 = bm25idf(args.term)
                print(f"BM25 IDF score of '{args.term}': {bm25:.2f}")
            except ValueError as e:
                print(e)
        case "tfidf":
            inv_idx.load()
            try:
                inv_freq = idf(args.term)
                term_freq = tf(args.doc_id, args.term)
                tf_idf = inv_freq * term_freq
                print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
            except ValueError as e:
                print(e)
        case _:
            parser.print_help()

def build():
    inv_idx.build()
    inv_idx.save()

def idf(term: str) -> float:
    return inv_idx.get_idf(term)

def bm25idf(term: str) -> float:
    return inv_idx.get_bm25_idf(term)

def tf(doc_id: int, term: str) -> int:
    return inv_idx.get_tf(doc_id, term)

if __name__ == "__main__":
    main()
