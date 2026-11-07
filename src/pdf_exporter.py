"""PDF Export Module for Document Intelligence Reports.
Generates beautifully formatted, multi-page, publication-grade PDFs from
Markdown Intelligence outputs (Summaries, Study Notes, Key Concepts, Quizzes, etc.).
Uses PyMuPDF (fitz) for native vector layout, headers, footers, pagination,
and Arabic/English bilingual font rendering.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple
import pymupdf


def _get_system_font() -> Optional[str]:
    """Finds an optimal TrueType font supporting Unicode, Latin, and Arabic."""
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def export_intelligence_to_pdf(
    content: str,
    title: str = "Document Intelligence Report",
    action: str = "Document Analysis",
    document_name: str = "Uploaded Document",
) -> bytes:
    """
    Renders structured Markdown content into a multi-page PDF document.
    """
    doc = pymupdf.open()
    font_path = _get_system_font()

    # Page Geometry (Standard A4)
    page_width, page_height = 595.0, 842.0
    margin_x = 50.0
    printable_width = page_width - (2 * margin_x)
    bottom_limit = 760.0

    current_date = datetime.now().strftime("%B %d, %Y • %I:%M %p")
    clean_action = action.replace("_", " ").title()

    def create_page():
        return doc.new_page(width=page_width, height=page_height)

    # First Page
    page = create_page()
    y = 45.0

    # ─────────────────────────────────────────────────────────────
    # FIRST PAGE HEADER BANNER
    # ─────────────────────────────────────────────────────────────
    header_height = 70.0
    header_rect = pymupdf.Rect(margin_x, y, page_width - margin_x, y + header_height)
    # Background card
    page.draw_rect(
        header_rect,
        color=(0.10, 0.34, 0.86),
        fill=(0.95, 0.97, 1.00),
        width=1.2,
    )
    # Brand Top accent line
    page.draw_rect(
        pymupdf.Rect(margin_x, y, page_width - margin_x, y + 4),
        color=(0.10, 0.34, 0.86),
        fill=(0.10, 0.34, 0.86),
    )

    # Header Titles
    page.insert_text(
        (margin_x + 16, y + 28),
        title,
        fontsize=16,
        fontname="Arial",
        fontfile=font_path,
        color=(0.10, 0.34, 0.86),
    )
    meta_line = f"Action: {clean_action}  |  Source: {document_name}  |  {current_date}"
    page.insert_text(
        (margin_x + 16, y + 48),
        meta_line,
        fontsize=9.5,
        fontname="Arial",
        fontfile=font_path,
        color=(0.35, 0.40, 0.50),
    )

    y += header_height + 25.0

    # ─────────────────────────────────────────────────────────────
    # CONTENT PARSER & MULTI-PAGE FLOW
    # ─────────────────────────────────────────────────────────────
    lines = content.split("\n")

    for raw_line in lines:
        line = raw_line.strip()

        # Handle empty lines
        if not line:
            y += 8.0
            continue

        # Check page break threshold
        if y > bottom_limit:
            page = create_page()
            y = 50.0
            # Running Top Header on secondary pages
            page.draw_line(
                (margin_x, y),
                (page_width - margin_x, y),
                color=(0.85, 0.88, 0.94),
                width=0.8,
            )
            page.insert_text(
                (margin_x, y - 8),
                f"AI PDF Assistant  •  {title}",
                fontsize=8.5,
                fontname="Arial",
                fontfile=font_path,
                color=(0.45, 0.50, 0.60),
            )
            y += 20.0

        is_arabic = any("\u0600" <= c <= "\u06FF" for c in line)
        align = 2 if is_arabic else 0

        # 1. Heading 1 (# ...)
        if line.startswith("# "):
            text = line[2:].replace("**", "").replace("*", "")
            h_rect = pymupdf.Rect(margin_x, y, page_width - margin_x, y + 36)
            page.insert_textbox(
                h_rect,
                text,
                fontsize=16,
                fontfile=font_path,
                color=(0.10, 0.34, 0.86),
                align=align,
            )
            y += 34.0
            continue

        # 2. Heading 2 (## ...)
        if line.startswith("## "):
            text = line[3:].replace("**", "").replace("*", "")
            h_rect = pymupdf.Rect(margin_x, y, page_width - margin_x, y + 30)
            page.insert_textbox(
                h_rect,
                text,
                fontsize=13.5,
                fontfile=font_path,
                color=(0.06, 0.11, 0.24),
                align=align,
            )
            y += 28.0
            continue

        # 3. Heading 3 (### ...)
        if line.startswith("### "):
            text = line[4:].replace("**", "").replace("*", "")
            h_rect = pymupdf.Rect(margin_x, y, page_width - margin_x, y + 26)
            page.insert_textbox(
                h_rect,
                text,
                fontsize=12,
                fontfile=font_path,
                color=(0.15, 0.20, 0.32),
                align=align,
            )
            y += 24.0
            continue

        # 4. Heading 4 (#### ...)
        if line.startswith("#### "):
            text = line[5:].replace("**", "").replace("*", "")
            h_rect = pymupdf.Rect(margin_x, y, page_width - margin_x, y + 22)
            page.insert_textbox(
                h_rect,
                text,
                fontsize=11,
                fontfile=font_path,
                color=(0.20, 0.25, 0.35),
                align=align,
            )
            y += 20.0
            continue

        # 5. Blockquote (> ...)
        if line.startswith("> "):
            text = line[2:].replace("**", "")
            q_height = 45.0
            # Left accent vertical bar
            page.draw_rect(
                pymupdf.Rect(margin_x, y, margin_x + 3.5, y + q_height - 6),
                color=(0.10, 0.34, 0.86),
                fill=(0.10, 0.34, 0.86),
            )
            b_rect = pymupdf.Rect(margin_x + 12, y, page_width - margin_x, y + q_height)
            unused = page.insert_textbox(
                b_rect,
                text,
                fontsize=9.5,
                fontfile=font_path,
                color=(0.25, 0.30, 0.42),
                align=align,
            )
            consumed = max(q_height - unused, 18.0)
            y += consumed + 8.0
            continue

        # 6. Bullet Items (* , - , • )
        if line.startswith(("- ", "* ", "• ")):
            clean_item = "•  " + line[2:].replace("**", "")
            b_height = 55.0
            b_rect = pymupdf.Rect(margin_x + 10, y, page_width - margin_x, y + b_height)
            unused = page.insert_textbox(
                b_rect,
                clean_item,
                fontsize=10,
                fontfile=font_path,
                color=(0.12, 0.16, 0.26),
                align=align,
            )
            consumed = max(b_height - unused, 16.0)
            y += consumed + 5.0
            continue

        # 7. Standard Paragraph
        clean_para = line.replace("**", "").replace("`", "")
        p_height = 95.0
        p_rect = pymupdf.Rect(margin_x, y, page_width - margin_x, y + p_height)
        unused = page.insert_textbox(
            p_rect,
            clean_para,
            fontsize=10,
            fontfile=font_path,
            color=(0.12, 0.16, 0.26),
            align=align,
        )
        consumed = max(p_height - unused, 16.0)
        y += consumed + 6.0

    # ─────────────────────────────────────────────────────────────
    # FOOTERS ON EVERY PAGE (Page X of Y & Branding)
    # ─────────────────────────────────────────────────────────────
    total_pages = len(doc)
    footer_y = 800.0

    for idx, p in enumerate(doc, 1):
        # Divider line
        p.draw_line(
            (margin_x, footer_y),
            (page_width - margin_x, footer_y),
            color=(0.85, 0.88, 0.94),
            width=0.8,
        )
        # Left Branding
        p.insert_text(
            (margin_x, footer_y + 16),
            "AI PDF Assistant  •  Document Intelligence",
            fontsize=8.5,
            fontname="Arial",
            fontfile=font_path,
            color=(0.50, 0.55, 0.65),
        )
        # Right Page Number
        page_str = f"Page {idx} of {total_pages}"
        p.insert_text(
            (page_width - margin_x - 65, footer_y + 16),
            page_str,
            fontsize=8.5,
            fontname="Arial",
            fontfile=font_path,
            color=(0.50, 0.55, 0.65),
        )

    # Optimize and subset embedded fonts
    try:
        doc.subset_fonts()
    except Exception:
        pass

    return doc.tobytes()
