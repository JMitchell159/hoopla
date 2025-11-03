#!/usr/bin/env python3

import argparse
from keyword_search import kw_search
from tools.inverted_index import InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Builds the inverted index for fast search")

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
        case _:
            parser.print_help()

def build():
    inv_idx = InvertedIndex()
    inv_idx.build()
    inv_idx.save()

if __name__ == "__main__":
    main()
