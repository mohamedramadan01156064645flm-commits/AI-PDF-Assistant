"""Chatbot module managing conversational multi-turn context,
follow-up query contextualization, mode selection, and document intelligence workflows.
"""

from typing import List, Dict, Any, Optional
from src.rag_pipeline import RAGPipeline


CONTEXTUALIZE_PROMPT_TEMPLATE = """Given a conversation history and the latest user follow-up question which references or builds upon previous messages, formulate a standalone question that can be understood independently.
Do NOT answer the question. Only reformulate it to make it explicit, or return it as is if it is already standalone.
If the question is in Arabic, keep the standalone question in Arabic.

Conversation History:
{chat_history}

User Follow-up: {question}

Standalone Question:"""


class PDFChatbot:
    """Conversational Assistant managing dual-mode universal AI chat and multi-PDF intelligence."""

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
            formatted.append(f"User: {user_msg}\nAssistant: {bot_msg[:250]}")
        return "\n".join(formatted)

    def _is_greeting_or_meta(self, question: str) -> Optional[str]:
        """Detects greetings, identity, or capability inquiries."""
        q = question.strip().lower().rstrip("?.! ")
        words = q.split()

        greetings = {
            "hi", "hello", "hey", "hola", "sup", "good morning", "good evening", "good afternoon",
            "ازيك", "عامل ايه", "مرحبا", "اهلا", "أهلا", "السلام عليكم", "صباح الخير", "مساء الخير",
            "هاي", "الو", "سلام"
        }
        identity = {
            "who are you", "who r u", "who are u", "what are you", "what is your name",
            "introduce yourself", "tell me about yourself", "who created you",
            "انت مين", "إنت مين", "من انت", "من أنت", "مين انت", "مين إنت",
            "عرف نفسك", "عرفني بيك", "ما هي وظيفتك", "بتعمل ايه", "انت ايه"
        }
        capabilities = {
            "what can you do", "help", "how can you help", "how to use", "what are your features",
            "بتعرف تعمل ايه", "تقدر تساعدني بايه", "مساعدة", "كيف استخدمك", "ازاي استعملك"
        }

        is_greeting = q in greetings or (len(words) <= 2 and any(w in greetings for w in words))
        is_identity = q in identity or any(q.startswith(phrase) for phrase in ["who are you", "who r u", "انت مين", "من انت"])
        is_capability = q in capabilities or any(q.startswith(phrase) for phrase in ["what can you do", "بتعرف تعمل ايه", "تقدر تعمل"])

        if is_greeting:
            return (
                "مرحبًا بك! 👋 أنا مساعدك الذكي المتقدم (Universal & Multi-PDF AI Assistant).\n\n"
                "أنا جاهز للإجابة على جميع أسئلتك العامة في البرمجة، والذكاء الاصطناعي، والعلوم، والرياضيات، وإدارة الأعمال، "
                "وكذلك قراءة وتحليل ومقارنة أي عدد من ملفات الـ PDF المرفوعة بدقة فائقة.\n\n"
                "*Hello! I'm your Universal & Multi-PDF AI Assistant. Ask me anything general or upload PDFs to interact deeply with your documents!*"
            )

        if is_identity:
            return (
                "أنا **Universal & Multi-PDF AI Assistant** — نظام ذكاء اصطناعي هجين يجمع بين:\n\n"
                "1. 🧠 **المساعد الذكي الشامل (Universal AI)**: الإجابة على أي سؤال عام أو تقني في علوم الحاسب، والذكاء الاصطناعي، والرياضيات، والتحليل، ومشاكل البرمجة المختلفة.\n"
                "2. 📄 **مساعد الـ Multi-PDF المتطور (Advanced RAG)**: قراءة وفهم عدة ملفات PDF معاً، والبحث الدلالي عبر جميع المستندات المرفوعة، والمقارنة بينها، والتلخيص الذكي، واستخراج المفاهيم.\n\n"
                "💡 يمكنك سؤالي عن أي موضوع عام، أو رفع ملفاتك وسؤالي عنها مباشرة!"
            )

        if is_capability:
            return (
                "إليك أبرز إمكانياتي: 🚀\n\n"
                "1. 💬 **محادثة عامة ذكية**: مناقشة وحل المسائل البرمجية والرياضية والتقنية دون الحاجة لرفع مستندات.\n"
                "2. 📚 **دعم ملفات متعددة (Multi-PDF)**: البحث والربط بين عدة ملفات PDF في نفس الوقت.\n"
                "3. ⚖️ **المقارنة والتحليل**: مقارنة مستندين أو أكثر لتحديد أوجه التشابه والاختلاف.\n"
                "4. 📝 **التلخيص والملاحظات**: توليد ملخص تنفيذي أو مذكرات دراسية (Study Notes) بضغطة زر.\n"
                "5. 🌐 **ثنائي اللغة**: تحدث بطلاقة بالعربية (بما فيها العامية المصرية) والإنجليزية."
            )

        return None

    def _is_followup(self, question: str) -> bool:
        """Determines if a question is likely a follow-up referencing previous context."""
        q = question.strip().lower()
        words = q.split()

        cues = [
            "it", "its", "they", "them", "their", "this", "that", "these", "those",
            "the former", "the latter", "above", "mentioned", "more details", "explain more",
            "elaborate", "give an example", "why?", "how so?", "what about", "and what", "and how",
            "also", "too", "give code", "show me", "can you clarify",
            "ده", "دي", "دول", "هو", "هي", "هما", "السابق", "المذكور", "اشرح اكتر", "وضح اكتر",
            "طب و", "كمان", "اديني مثال", "اكتبلي الكود"
        ]

        if len(words) <= 2 and any(w in ["why", "how", "what", "more", "ليه", "ازاي", "فين", "طب"] for w in words):
            return True

        return any(cue in words or cue == q for cue in cues)

    def _contextualize_query(self, question: str) -> str:
        """
        Reformulates follow-up question using recent chat history only when necessary.
        """
        if not self.history or len(self.history) == 0:
            return question

        # If question is clearly self-contained, do not alter it
        if not self._is_followup(question):
            return question

        history_str = self._format_history_for_contextualizer()
        prompt = CONTEXTUALIZE_PROMPT_TEMPLATE.format(
            chat_history=history_str,
            question=question,
        )

        try:
            standalone = self.rag_pipeline._call_llm(prompt)
            clean_standalone = standalone.strip().split("\n")[0].strip('"').strip()
            return clean_standalone if clean_standalone else question
        except Exception:
            return question

    def ask(
        self,
        question: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        source_filter: Optional[str] = None,
        forced_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Processes a user question:
        1. Checks for greetings/assistant identity
        2. Contextualizes follow-ups using conversation memory
        3. Routes intelligently to General AI Chat, PDF RAG, or Hybrid Reasoning
        4. Retains turn history
        """
        # 1. Greeting or assistant identity check
        meta_reply = self._is_greeting_or_meta(question)
        if meta_reply:
            turn_entry = {
                "user": question,
                "standalone_query": question,
                "assistant": meta_reply,
                "mode": "GENERAL",
                "citations": [],
                "retrieved_chunks": [],
                "grounded": False,
                "error": None,
            }
            self.history.append(turn_entry)
            return turn_entry

        # 2. Contextualize query if follow-up
        standalone_query = self._contextualize_query(question)

        # 3. Generate answer via RAG Pipeline
        rag_output = self.rag_pipeline.generate_answer(
            question=standalone_query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            source_filter=source_filter,
            forced_mode=forced_mode,
        )

        turn_entry = {
            "user": question,
            "standalone_query": standalone_query,
            "assistant": rag_output["answer"],
            "mode": rag_output.get("mode", "GENERAL"),
            "citations": rag_output.get("citations", []),
            "retrieved_chunks": rag_output.get("retrieved_chunks", []),
            "grounded": rag_output.get("grounded", False),
            "error": rag_output.get("error"),
        }

        self.history.append(turn_entry)
        return turn_entry

    def summarize_document(self, source_name: str) -> str:
        """Generates an executive summary of an indexed PDF document."""
        return self.rag_pipeline.summarize_document(source_name)

    def compare_documents(self, source_a: str, source_b: str) -> str:
        """Compares two uploaded documents."""
        return self.rag_pipeline.compare_documents(source_a, source_b)

    def generate_study_notes(self, source_name: str) -> str:
        """Generates high-yield study notes."""
        return self.rag_pipeline.generate_study_notes(source_name)

    def generate_qa(self, source_name: str) -> str:
        """Generates practice Q&A quiz."""
        return self.rag_pipeline.generate_qa(source_name)

    def extract_key_concepts(self, source_name: str) -> str:
        """Extracts technical glossary."""
        return self.rag_pipeline.extract_key_concepts(source_name)

    def explain_document(self, source_name: str) -> str:
        """Explains the document step by step."""
        return self.rag_pipeline.explain_document(source_name)

    def extract_key_points(self, source_name: str) -> str:
        """Extracts key points and highlights."""
        return self.rag_pipeline.extract_key_points(source_name)

    def explain_difficult_concepts(self, source_name: str) -> str:
        """Simplifies complex concepts with analogies."""
        return self.rag_pipeline.explain_difficult_concepts(source_name)

    def create_flashcards(self, source_name: str) -> str:
        """Generates active recall flashcards."""
        return self.rag_pipeline.create_flashcards(source_name)

    def generate_quiz(self, source_name: str) -> str:
        """Generates a multiple-choice self quiz."""
        return self.rag_pipeline.generate_quiz(source_name)
