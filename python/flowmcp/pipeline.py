from typing import List, Optional, Union

from .llm.base import LLM
from .mcp.client import MCP
from .rag.loader import RAG
from .rag.retriever import DEFAULT_EMBEDDING_MODEL


class Pipeline:
    """
    Main entry point. Combines LLM + RAG + MCP in one simple interface.

    Example:
        pipeline = Pipeline(
            llm="openai/gpt-4",
            docs="./my-docs",
            tools=["filesystem", "github"]
        )
        answer = pipeline.ask("What is our refund policy?")
        answer = await pipeline.aask("What is our refund policy?")
    """

    def __init__(
        self,
        llm: Union[str, LLM],
        docs: Optional[Union[str, List[str]]] = None,
        tools: Optional[List[str]] = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        self.llm = LLM.from_string(llm) if isinstance(llm, str) else llm
        self.rag = RAG(docs, embedding_model=embedding_model) if docs else None
        self.mcp = MCP(tools) if tools else None

    def ask(self, question: str, top_k: int = 3) -> str:
        context = ""

        if self.rag:
            chunks = self.rag.retrieve(question, top_k=top_k)
            context = "\n\n".join(chunks)

        tool_results = ""
        if self.mcp:
            tool_results = self.mcp.run(question)

        prompt = self._build_prompt(question, context, tool_results)
        return self.llm.complete(prompt)

    def batch(self, queries: List[str], top_k: int = 3) -> List[str]:
        """Answer queries sequentially, returning answers in input order.

        Each query runs its own retrieval, tools, and LLM call through ask().
        An empty list makes no calls. Errors propagate immediately, so later
        queries are not processed. This is not a provider-side batch API.

        Example:
            answers = pipeline.batch(
                ["What is the refund policy?", "How do I cancel?"], top_k=5
            )
        """
        return [self.ask(query, top_k=top_k) for query in queries]

    async def aask(self, question: str) -> str:
        import asyncio

        context = ""
        if self.rag:
            chunks = await asyncio.to_thread(self.rag.retrieve, question)
            context = "\n\n".join(chunks)

        tool_results = ""
        if self.mcp:
            tool_results = await self.mcp.arun(question)

        prompt = self._build_prompt(question, context, tool_results)
        return await self.llm.acomplete(prompt)

    def _build_prompt(self, question: str, context: str, tool_results: str) -> str:
        parts = []
        if context:
            parts.append(f"Context from documents:\n{context}")
        if tool_results:
            parts.append(f"Tool results:\n{tool_results}")
        parts.append(f"Question: {question}")
        return "\n\n".join(parts)
