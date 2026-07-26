"""Tests for RAG module."""

from own_agent.context.rag import RagManager, CodeChunker, Bm25Indexer, Retriever


def test_chunker_basic():
    chunker = CodeChunker(chunk_size=10, overlap=3)
    chunks = chunker.chunk_text("line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\nline9\nline10\nline11\nline12", source="test.txt")
    assert len(chunks) >= 2
    assert chunks[0].file_path == "test.txt"
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 10


def test_bm25_basic():
    chunker = CodeChunker(chunk_size=50, overlap=10)
    text = "\n".join(f"line{i} = {i}" for i in range(30))
    text += "\ndef hello_world():\n    return 'hello test'\n"
    chunks = chunker.chunk_text(text, source="test.py")
    indexer = Bm25Indexer()
    indexer.index(chunks)
    results = indexer.search("hello_world", top_k=5)
    assert len(results) > 0
    chunk, score = results[0]
    assert score > 0
    assert "hello_world" in chunk.content


def test_bm25_no_match():
    indexer = Bm25Indexer()
    indexer.index([])
    results = indexer.search("anything", top_k=5)
    assert results == []


def test_retriever_format():
    chunker = CodeChunker(chunk_size=50, overlap=10)
    chunks = chunker.chunk_text("def foo():\n    pass", source="file.py")
    formatted = Retriever.format_context(chunks)
    assert "<project_context>" in formatted
    assert "file.py" in formatted
    assert "</project_context>" in formatted


def test_retriever_empty():
    formatted = Retriever.format_context([])
    assert formatted == ""


def test_rag_manager():
    import asyncio
    mgr = RagManager(project_root="own_agent")
    asyncio.run(mgr.index_project())
    assert mgr.total_chunks > 0
    assert mgr.is_indexed

    ctx = asyncio.run(mgr.retrieve_context("Agent class"))
    assert len(ctx) > 0
    assert "<project_context>" in ctx
