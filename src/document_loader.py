"""Document loader module using PyMuPDF (fitz) for extracting text and metadata from PDFs.
Supports single and multiple PDF files, byte streams, and local file paths.
Preserves page numbers (1-indexed) and source metadata.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import io
import unicodedata
try:
    import pymupdf  # PyMuPDF 2.0+
except ImportError:
    import fitz as pymupdf


@dataclass
class DocumentPage:
    """Represents a single extracted page from a document."""
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_content": self.page_content,
            "metadata": self.metadata,
        }


class PDFDocumentLoader:
    """Loader for PDF documents using PyMuPDF (fitz)."""

    def __init__(self, extract_images_text: bool = False):
        """
        Initialize the PDF loader.
        Args:
            extract_images_text: Future extension for OCR on scanned pages if needed.
        """
        self.extract_images_text = extract_images_text

    def load_pdf(
        self,
        source: Union[str, Path, bytes, io.BytesIO],
        file_name: Optional[str] = None
    ) -> List[DocumentPage]:
        """
        Load a single PDF file from path or memory buffer.
        
        Args:
            source: File path (str/Path) or raw bytes or BytesIO stream.
            file_name: Optional explicit name for source metadata.

        Returns:
            List of DocumentPage objects containing text and page metadata.
        """
        doc = None
        extracted_pages: List[DocumentPage] = []
        source_name = file_name or "uploaded_document.pdf"

        try:
            if isinstance(source, (str, Path)):
                path_obj = Path(source)
                if not path_obj.exists():
                    raise FileNotFoundError(f"PDF file not found at: {path_obj}")
                source_name = file_name or path_obj.name
                doc = pymupdf.open(str(path_obj))
            elif isinstance(source, bytes):
                doc = pymupdf.open(stream=source, filetype="pdf")
            elif isinstance(source, io.BytesIO):
                source.seek(0)
                doc = pymupdf.open(stream=source.read(), filetype="pdf")
            else:
                raise TypeError(f"Unsupported source type for PDF: {type(source)}")

            total_pages = len(doc)

            for page_index in range(total_pages):
                page = doc[page_index]
                text = page.get_text("text") or ""
                # Normalize Unicode and clean up repeated trailing whitespace/null chars
                clean_text = unicodedata.normalize("NFKC", text.replace("\x00", "")).strip()

                if not clean_text:
                    # Keep note of empty/scanned pages
                    clean_text = f"[Empty page or image-only content on page {page_index + 1}]"

                metadata = {
                    "source": source_name,
                    "page": page_index + 1,  # 1-indexed for humans
                    "total_pages": total_pages,
                    "char_count": len(clean_text),
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

        Args:
            sources: List of file paths or byte buffers.
            file_names: Optional matching list of file names.

        Returns:
            Combined list of DocumentPage objects.
        """
        all_pages: List[DocumentPage] = []

        for idx, src in enumerate(sources):
            name = file_names[idx] if (file_names and idx < len(file_names)) else None
            pages = self.load_pdf(src, file_name=name)
            all_pages.extend(pages)

        return all_pages
