#!/usr/bin/env python3

import argparse
from keyword_search import kw_search

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            result = kw_search(args.query)
            for i, r in enumerate(result):
                print(f"{i+1}. {r["title"]}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
