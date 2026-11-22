from unittest.mock import Mock, call

import pytest
from flowmcp import Pipeline


def test_empty_batch_makes_no_calls():
    pipeline = Pipeline(llm=Mock())
    pipeline.rag = Mock()
    pipeline.mcp = Mock()
    assert pipeline.batch([]) == []
    pipeline.rag.retrieve.assert_not_called()
    pipeline.mcp.run.assert_not_called()
    pipeline.llm.complete.assert_not_called()


def test_single_query_matches_ask():
    pipeline = Pipeline(llm=Mock())
    pipeline.llm.complete.side_effect = lambda prompt: prompt
    assert pipeline.batch(["Refund policy?"]) == [pipeline.ask("Refund policy?")]


def test_multiple_queries_preserve_order_and_duplicates():
    pipeline = Pipeline(llm=Mock())
    pipeline.llm.complete.side_effect = ["first", "second", "third"]
    assert pipeline.batch(["A", "B", "A"]) == ["first", "second", "third"]
    assert pipeline.llm.complete.call_args_list == [
        call("Question: A"),
        call("Question: B"),
        call("Question: A"),
    ]


@pytest.mark.parametrize("top_k", [0, 1, 5])
def test_batch_retrieves_and_runs_tools_independently(top_k):
    pipeline = Pipeline(llm=Mock())
    pipeline.rag = Mock()
    pipeline.rag.retrieve.side_effect = [["first context"], ["second context"]]
    pipeline.mcp = Mock()
    pipeline.mcp.run.side_effect = ["first tool result", "second tool result"]
    pipeline.llm.complete.side_effect = lambda prompt: prompt

    answers = pipeline.batch(["A", "B"], top_k=top_k)

    assert pipeline.rag.retrieve.call_args_list == [
        call("A", top_k=top_k),
        call("B", top_k=top_k),
    ]
    assert pipeline.mcp.run.call_args_list == [call("A"), call("B")]
    assert answers == [
        "Context from documents:\nfirst context\n\n"
        "Tool results:\nfirst tool result\n\nQuestion: A",
        "Context from documents:\nsecond context\n\n"
        "Tool results:\nsecond tool result\n\nQuestion: B",
    ]


def test_batch_defaults_to_three_chunks():
    pipeline = Pipeline(llm=Mock())
    pipeline.rag = Mock()
    pipeline.rag.retrieve.return_value = []
    pipeline.batch(["A"])
    pipeline.rag.retrieve.assert_called_once_with("A", top_k=3)


def test_batch_stops_on_error():
    pipeline = Pipeline(llm=Mock())
    pipeline.llm.complete.side_effect = ["first", RuntimeError("provider failed")]
    with pytest.raises(RuntimeError, match="provider failed"):
        pipeline.batch(["A", "B", "C"])
    assert pipeline.llm.complete.call_args_list == [
        call("Question: A"),
        call("Question: B"),
    ]
