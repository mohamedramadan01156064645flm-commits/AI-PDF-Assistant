"""Chatbot module managing conversational multi-turn RAG, follow-up query contextualization,
chat history, citations, and document summarization.
"""

from typing import List, Dict, Any, Optional
from src.rag_pipeline import RAGPipeline


CONTEXTUALIZE_PROMPT_TEMPLATE = """Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
If the question is in Arabic, keep the standalone question in Arabic.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""


class PDFChatbot:
    """Conversational Chatbot with multi-turn memory and follow-up query resolution."""

    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag_pipeline = rag_pipeline
        self.history: List[Dict[str, Any]] = []

    def _format_history_for_contextualizer(self, max_turns: int = 3) -> str:
        """Formats the last N turns of conversation history."""
        recent_turns = self.history[-max_turns:] if self.history else []
        formatted = []
        for turn in recent_turns:
            user_msg = turn.get("user", "")
            bot_msg = turn.get("assistant", "")
            formatted.append(f"User: {user_msg}\nAssistant: {bot_msg[:200]}")
        return "\n".join(formatted)

    def _contextualize_query(self, question: str) -> str:
        """
        Reformulates follow-up question using recent chat history so the vector retriever
        can find the right chunks.
        """
        if not self.history or len(self.history) == 0:
            return question

        # If question is already self-contained (e.g. longer than 15 words), keep it
        if len(question.split()) > 15:
            return question

        history_str = self._format_history_for_contextualizer()
        prompt = CONTEXTUALIZE_PROMPT_TEMPLATE.format(
            chat_history=history_str,
            question=question,
        )

        try:
            if self.rag_pipeline.provider == "gemini":
                standalone = self.rag_pipeline._call_gemini(prompt)
                clean_standalone = standalone.strip().split("\n")[0].strip('"')
                return clean_standalone if clean_standalone else question
            elif self.rag_pipeline.provider == "ollama":
                standalone = self.rag_pipeline._call_ollama(prompt)
                clean_standalone = standalone.strip().split("\n")[0].strip('"')
                return clean_standalone if clean_standalone else question
        except Exception:
            # Fallback to appending last question context if LLM call fails
            last_turn = self.history[-1]
            return f"{question} (context: {last_turn.get('user', '')})"

        return question

    def ask(
        self,
        question: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        source_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Processes a user question, reformulating if it is a follow-up,
        executing RAG, and updating conversation history.
        """
        # Contextualize query if there is conversation history
        standalone_query = self._contextualize_query(question)

        # Execute RAG pipeline with the reformulated query
        rag_output = self.rag_pipeline.generate_answer(
            question=standalone_query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            source_filter=source_filter,
        )

        turn_entry = {
            "user": question,
            "standalone_query": standalone_query,
            "assistant": rag_output["answer"],
            "citations": rag_output["citations"],
            "retrieved_chunks": rag_output["retrieved_chunks"],
            "error": rag_output.get("error"),
        }

        self.history.append(turn_entry)

        return turn_entry

    def summarize_document(self, source_name: str, max_chunks: int = 6) -> str:
        """
        Generates an executive summary of an indexed PDF document.
        """
        raw_chunks = self.rag_pipeline.retriever.vector_store.query_similarity(
            query="main overview introduction summary conclusion key points",
            n_results=max_chunks,
            where_filter={"source": source_name},
        )

        if not raw_chunks:
            return f"No content found for document '{source_name}' to summarize."

        context_text = "\n\n".join([f"Page {c['metadata'].get('page')}: {c['content']}" for c in raw_chunks])

        prompt = f"""You are an executive research assistant. Please provide a clear, structured summary of the document '{source_name}'.
Highlight:
- Main Topic & Objectives
- Core Concepts and Key Takeaways
- Practical Implications or Conclusions

Support both English and Arabic depending on the document language.

DOCUMENT EXCERPTS:
{context_text}

STRUCTURED SUMMARY:"""

        try:
            if self.rag_pipeline.provider == "gemini":
                return self.rag_pipeline._call_gemini(prompt)
            elif self.rag_pipeline.provider == "ollama":
                return self.rag_pipeline._call_ollama(prompt)
            else:
                return (
                    f"**Summary Preview for {source_name}:**\n\n"
                    f"- Document has {len(raw_chunks)} key sections analyzed.\n"
                    f"- Excerpt: \"{raw_chunks[0]['content'][:300]}...\"\n\n"
                    f"*Connect Gemini API Key for complete executive synthesis.*"
                )
        except Exception as e:
            return f"Summary generation failed: {e}"

    def clear_history(self) -> None:
        """Resets the conversation history."""
        self.history.clear()
