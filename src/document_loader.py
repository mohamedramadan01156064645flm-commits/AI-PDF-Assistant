"""Document loader module using PyMuPDF (fitz) for extracting text and metadata from PDFs.
Supports single and multiple PDF files, byte streams, local file paths,
scanned page detection, duplicate detection via content hashing, and resilient error recovery.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import io
import hashlib
import unicodedata

try:
    import pymupdf  # PyMuPDF 2.0+
except ImportError:
    import fitz as pymupdf


@dataclass
class DocumentPage:
    """Represents a single extracted page from a document with rich metadata."""
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_content": self.page_content,
            "metadata": self.metadata,
        }


class PDFDocumentLoader:
    """Enterprise-grade loader for PDF documents using PyMuPDF."""

    def __init__(self, extract_images_text: bool = False):
        self.extract_images_text = extract_images_text

    @staticmethod
    def calculate_file_hash(data: Union[bytes, Path, str]) -> str:
        """Computes SHA-256 hash to identify duplicate document uploads."""
        hasher = hashlib.sha256()
        if isinstance(data, (str, Path)):
            with open(data, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
        elif isinstance(data, bytes):
            hasher.update(data)
        elif isinstance(data, io.BytesIO):
            pos = data.tell()
            data.seek(0)
            while chunk := data.read(65536):
                hasher.update(chunk)
            data.seek(pos)
        return hasher.hexdigest()

    def load_pdf(
        self,
        source: Union[str, Path, bytes, io.BytesIO],
        file_name: Optional[str] = None
    ) -> List[DocumentPage]:
        """
        Loads a single PDF file from path or memory buffer with metadata extraction.
        Handles scanned pages, empty pages, and encrypted PDFs gracefully.
        """
        doc = None
        extracted_pages: List[DocumentPage] = []
        source_name = file_name or "uploaded_document.pdf"

        try:
            doc_hash = ""
            if isinstance(source, (str, Path)):
                path_obj = Path(source)
                if not path_obj.exists():
                    raise FileNotFoundError(f"PDF file not found at: {path_obj}")
                source_name = file_name or path_obj.name
                doc_hash = self.calculate_file_hash(path_obj)
                doc = pymupdf.open(str(path_obj))
            elif isinstance(source, bytes):
                doc_hash = self.calculate_file_hash(source)
                doc = pymupdf.open(stream=source, filetype="pdf")
            elif isinstance(source, io.BytesIO):
                source.seek(0)
                content = source.read()
                doc_hash = self.calculate_file_hash(content)
                doc = pymupdf.open(stream=content, filetype="pdf")
            else:
                raise TypeError(f"Unsupported source type for PDF: {type(source)}")

            # Handle encrypted PDFs
            if doc.is_encrypted:
                # Try default empty password
                unlocked = doc.authenticate("")
                if not unlocked:
                    raise PermissionError(f"PDF '{source_name}' is password-protected and cannot be read.")

            total_pages = len(doc)
            if total_pages == 0:
                raise ValueError(f"PDF '{source_name}' contains 0 pages.")

            # Document-level metadata
            doc_meta = doc.metadata or {}
            title = doc_meta.get("title") or source_name
            author = doc_meta.get("author") or ""

            for page_index in range(total_pages):
                page = doc[page_index]
                text = page.get_text("text") or ""
                # Normalize Unicode and clean up repeated trailing whitespace/null chars
                clean_text = unicodedata.normalize("NFKC", text.replace("\x00", "")).strip()

                is_scanned = False
                # If page text is very short but page contains images, mark as scanned
                if len(clean_text) < 30:
                    image_list = page.get_images()
                    if image_list:
                        is_scanned = True
                        if not clean_text:
                            clean_text = (
                                f"[Scanned/Image-only page on page {page_index + 1}. "
                                f"Contains {len(image_list)} embedded image(s).]"
                            )

                if not clean_text:
                    clean_text = f"[Empty page on page {page_index + 1}]"

                metadata = {
                    "source": source_name,
                    "title": title,
                    "author": author,
                    "page": page_index + 1,  # 1-indexed for humans
                    "total_pages": total_pages,
                    "char_count": len(clean_text),
                    "file_hash": doc_hash,
                    "is_scanned": is_scanned,
                }

                extracted_pages.append(
                    DocumentPage(page_content=clean_text, metadata=metadata)
                )

        except Exception as e:
            raise RuntimeError(f"Error reading PDF '{source_name}': {str(e)}") from e
        finally:
            if doc is not None:
                doc.close()

        return extracted_pages

    def load_multiple_pdfs(
        self,
        sources: List[Union[str, Path, bytes, io.BytesIO]],
        file_names: Optional[List[str]] = None
    ) -> List[DocumentPage]:
        """
        Load and aggregate pages from multiple PDF sources.
        """
        all_pages: List[DocumentPage] = []

        for idx, src in enumerate(sources):
            name = file_names[idx] if (file_names and idx < len(file_names)) else None
            pages = self.load_pdf(src, file_name=name)
            all_pages.extend(pages)

        return all_pages
