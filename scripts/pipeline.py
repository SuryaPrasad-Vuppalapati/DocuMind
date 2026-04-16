import argparse

from rag.pipeline import RAGPipeline


def cmd_build(args: argparse.Namespace) -> None:
    pipeline = RAGPipeline(
        corpus_path=args.corpus_path,
        index_path=args.index_path,
        chunks_path=args.chunks_path,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        embedding_model=args.embedding_model,
        chat_model=args.chat_model,
    )
    total_chunks = pipeline.build()
    print(f"Index built: {args.index_path}")
    print(f"Chunks saved: {args.chunks_path}")
    print(f"Total chunks: {total_chunks}")


def cmd_ask(args: argparse.Namespace) -> None:
    pipeline = RAGPipeline(
        corpus_path=args.corpus_path,
        index_path=args.index_path,
        chunks_path=args.chunks_path,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        embedding_model=args.embedding_model,
        chat_model=args.chat_model,
    )
    answer, results = pipeline.ask(query=args.query, k=args.k)
    print("\nAnswer:")
    print(answer)
    print("\nTop retrieved chunks:")
    for i, chunk in enumerate(results, start=1):
        print(
            f"{i}. score={chunk['score']:.4f} | source={chunk['source']} | page={chunk['page']}"
        )


def cmd_full(args: argparse.Namespace) -> None:
    pipeline = RAGPipeline(
        corpus_path=args.corpus_path,
        index_path=args.index_path,
        chunks_path=args.chunks_path,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        embedding_model=args.embedding_model,
        chat_model=args.chat_model,
    )
    answer, results, total_chunks = pipeline.run_full(query=args.query, k=args.k)
    print(f"Total chunks indexed: {total_chunks}")
    print("\nAnswer:")
    print(answer)
    print("\nTop retrieved chunks:")
    for i, chunk in enumerate(results, start=1):
        print(
            f"{i}. score={chunk['score']:.4f} | source={chunk['source']} | page={chunk['page']}"
        )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG pipeline runner")
    parser.add_argument("--corpus-path", default="data/corpus.json")
    parser.add_argument("--index-path", default="data/my_index.faiss")
    parser.add_argument("--chunks-path", default="data/chunks.json")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--chat-model", default="gpt-4o-mini")

    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build index from corpus")
    build_parser.set_defaults(func=cmd_build)

    ask_parser = subparsers.add_parser("ask", help="Ask against existing index")
    ask_parser.add_argument("--query", required=True)
    ask_parser.add_argument("-k", type=int, default=5)
    ask_parser.set_defaults(func=cmd_ask)

    full_parser = subparsers.add_parser(
        "full", help="Build index then answer query in one run"
    )
    full_parser.add_argument("--query", required=True)
    full_parser.add_argument("-k", type=int, default=5)
    full_parser.set_defaults(func=cmd_full)

    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
