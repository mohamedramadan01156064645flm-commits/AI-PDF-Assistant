"""Text splitting module for chunking extracted document pages.
Implements RecursiveCharacterTextSplitter with metadata preservation,
custom chunk size/overlap, and chunk analytics.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from src.document_loader import DocumentPage


@dataclass
class TextChunk:
    """Represents a text chunk with retained metadata."""
    page_content: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_content": self.page_content,
            "metadata": self.metadata,
        }


class DocumentSplitter:
    """Splits documents into overlapping chunks with semantic separator hierarchy."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None,
    ):
        """
        Args:
            chunk_size: Target maximum characters per chunk.
            chunk_overlap: Number of characters to overlap between adjacent chunks.
            separators: Hierarchy of split delimiters (defaults to paragraph, newline, sentence, space).
        """
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly smaller than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Support both standard punctuation and Arabic punctuation delimiters
        self.separators = separators or ["\n\n", "\n", "۔", "؟", ".", "!", "?", "،", ";", " ", ""]

    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Recursively splits text using the separator hierarchy."""
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator else list(text)

        good_splits: List[str] = []
        for s in splits:
            if not s:
                continue
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if new_separators:
                    sub_splits = self._split_text_recursive(s, new_separators)
                    good_splits.extend(sub_splits)
                else:
                    # Hard cut if no more separators exist
                    good_splits.append(s)

        # Merge splits respecting chunk_size and chunk_overlap
        current_chunk: List[str] = []
        current_len = 0

        for piece in good_splits:
            piece_len = len(piece) + (len(separator) if current_chunk else 0)
            if current_len + piece_len > self.chunk_size and current_chunk:
                merged = separator.join(current_chunk).strip()
                if merged:
                    final_chunks.append(merged)

                # Keep overlap pieces
                overlap_pieces: List[str] = []
                overlap_len = 0
                for back_piece in reversed(current_chunk):
                    if overlap_len + len(back_piece) <= self.chunk_overlap:
                        overlap_pieces.insert(0, back_piece)
                        overlap_len += len(back_piece) + len(separator)
                    else:
                        break
                current_chunk = overlap_pieces
                current_len = sum(len(p) for p in current_chunk) + (
                    len(separator) * (len(current_chunk) - 1) if current_chunk else 0
                )

            current_chunk.append(piece)
            current_len += len(piece) + (len(separator) if len(current_chunk) > 1 else 0)

        if current_chunk:
            final = separator.join(current_chunk).strip()
            if final:
                final_chunks.append(final)

        return final_chunks

    def split_pages(self, pages: List[DocumentPage]) -> List[TextChunk]:
        """
        Split a list of DocumentPages into TextChunks, preserving and expanding metadata.
        """
        chunks: List[TextChunk] = []
        global_chunk_idx = 0

        for page in pages:
            raw_text = page.page_content.strip()
            if not raw_text or raw_text.startswith("[Empty page"):
                continue

            page_splits = self._split_text_recursive(raw_text, self.separators)

            for local_idx, split_text in enumerate(page_splits):
                if not split_text.strip():
                    continue

                chunk_metadata = dict(page.metadata)
                chunk_metadata.update(
                    {
                        "chunk_index": local_idx,
                        "global_chunk_id": f"{chunk_metadata.get('source', 'doc')}_p{chunk_metadata.get('page', 0)}_c{local_idx}",
                        "chunk_char_count": len(split_text),
                    }
                )

                chunks.append(
                    TextChunk(page_content=split_text, metadata=chunk_metadata)
                )
                global_chunk_idx += 1

        return chunks

    def get_split_stats(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """Calculates statistical overview of the chunks."""
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
            }

        sizes = [len(c.page_content) for c in chunks]
        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": round(sum(sizes) / len(sizes), 1),
            "min_chunk_size": min(sizes),
            "max_chunk_size": max(sizes),
        }
