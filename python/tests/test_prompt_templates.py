from unittest.mock import AsyncMock, Mock

import pytest
from flowmcp import Pipeline


def make_pipeline():
    llm = Mock()
    llm.complete.side_effect = lambda prompt: prompt
    llm.acomplete = AsyncMock(side_effect=lambda prompt: prompt)
    return Pipeline(llm=llm)


def test_default_prompt_is_unchanged():
    pipeline = make_pipeline()
    pipeline.rag = Mock()
    pipeline.rag.retrieve.return_value = ["first", "second"]
    pipeline.mcp = Mock()
    pipeline.mcp.run.return_value = "tool evidence"
    assert pipeline.ask("Why?") == (
        "Context from documents:\nfirst\n\nsecond\n\n"
        "Tool results:\ntool evidence\n\nQuestion: Why?"
    )


def test_template_uses_retrieved_context_and_query():
    pipeline = make_pipeline()
    pipeline.rag = Mock()
    pipeline.rag.retrieve.return_value = ["first", "second"]
    pipeline.set_template("Evidence: {context}\nQuestion: {query}")
    assert pipeline.ask("Why?") == "Evidence: first\n\nsecond\nQuestion: Why?"
    pipeline.rag.retrieve.assert_called_once_with("Why?")


def test_query_only_template():
    pipeline = make_pipeline()
    pipeline.set_template("Answer briefly: {query}")
    assert pipeline.ask("Why?") == "Answer briefly: Why?"


def test_no_documents_produces_empty_context():
    pipeline = make_pipeline()
    pipeline.set_template("[{context}] {query}")
    assert pipeline.ask("Why?") == "[] Why?"


def test_literal_braces_and_query_braces_are_preserved():
    pipeline = make_pipeline()
    pipeline.set_template('{{"answer": "..."}} {query}')
    assert pipeline.ask("Explain {context}") == '{"answer": "..."} Explain {context}'


def test_template_keeps_tool_results():
    pipeline = make_pipeline()
    pipeline.mcp = Mock()
    pipeline.mcp.run.return_value = "fresh evidence"
    pipeline.set_template("Question: {query}")
    assert pipeline.ask("Why?") == "Question: Why?\n\nTool results:\nfresh evidence"


@pytest.mark.parametrize(
    "template",
    [
        "{missing}",
        "{}",
        "{0}",
        "{query.__class__}",
        "{query[0]}",
        "{query!r}",
        "{query:>10}",
        "{query:{context}}",
        "{query",
        "query}",
    ],
)
def test_invalid_template_is_rejected_without_calls(template):
    pipeline = make_pipeline()
    pipeline.set_template("Original: {query}")
    with pytest.raises(ValueError):
        pipeline.set_template(template)
    pipeline.llm.complete.assert_not_called()
    assert pipeline.ask("Why?") == "Original: Why?"


@pytest.mark.parametrize("template", [None, 1, ["{query}"]])
def test_non_string_template_is_rejected(template):
    with pytest.raises(TypeError, match="template must be a string"):
        make_pipeline().set_template(template)


def test_template_can_be_replaced():
    pipeline = make_pipeline()
    pipeline.set_template("Old {query}")
    pipeline.set_template("New {query}")
    assert pipeline.ask("Why?") == "New Why?"


@pytest.mark.asyncio
async def test_async_template_uses_context_and_tools():
    pipeline = make_pipeline()
    pipeline.rag = Mock()
    pipeline.rag.retrieve.return_value = ["evidence"]
    pipeline.mcp = Mock()
    pipeline.mcp.arun = AsyncMock(return_value="tool evidence")
    pipeline.set_template("{context} -> {query}")
    assert await pipeline.aask("Why?") == (
        "evidence -> Why?\n\nTool results:\ntool evidence"
    )
    pipeline.rag.retrieve.assert_called_once_with("Why?")
    pipeline.mcp.arun.assert_awaited_once_with("Why?")
    pipeline.llm.acomplete.assert_awaited_once()
