from __future__ import annotations

import base64
from pathlib import Path
from typing import Callable, Optional, cast

import streamlit as st
import streamlit.components.v1 as components

DialogDecorator = Callable[[str], Callable[[Callable[[], None]], Callable[[], None]]]


def render_manual(manual_html_path: Path, manual_pdf_path: Path, fallback_markdown: str) -> None:
    if manual_html_path.exists():
        html = manual_html_path.read_text(encoding="utf-8")
        components.html(html, height=640, scrolling=True)
        return
    if manual_pdf_path.exists():
        pdf_bytes = manual_pdf_path.read_bytes()
        b64 = base64.b64encode(pdf_bytes).decode("ascii")
        pdf_html = (
            "<iframe "
            f"src=\"data:application/pdf;base64,{b64}\" "
            "width=\"100%\" height=\"640\" style=\"border:0;\" "
            "></iframe>"
        )
        components.html(pdf_html, height=640, scrolling=True)
        return
    st.markdown(fallback_markdown)


def render_quick_manual_eng(project_root: Path) -> None:
    render_manual(
        project_root / "docs" / "user-guide" / "manual.html",
        project_root / "docs" / "user-guide" / "manual.pdf",
        """
**Manual not available**

Please check that `docs/user-guide/manual.html` (or `manual.pdf`) exists.
        """,
    )


def render_quick_manual_el(project_root: Path) -> None:
    render_manual(
        project_root / "docs" / "user-guide" / "manual-el.html",
        project_root / "docs" / "user-guide" / "manual-el.pdf",
        """
**Το εγχειρίδιο δεν είναι διαθέσιμο**

Ελέγξτε ότι υπάρχει το `docs/user-guide/manual-el.html` (ή `manual-el.pdf`).
        """,
    )


def render_info(project_root: Path) -> None:
    info_html_path = project_root / "docs" / "user-guide" / "info.html"
    if info_html_path.exists():
        html = info_html_path.read_text(encoding="utf-8")
        components.html(html, height=520, scrolling=True)
        return
    st.markdown(
        """
**Info not available**

Please check that `docs/user-guide/info.html` exists.
        """
    )


def get_dialog_decorator() -> Optional[DialogDecorator]:
    dialog = getattr(st, "dialog", None)
    if callable(dialog):
        return cast(DialogDecorator, dialog)
    dialog = getattr(st, "experimental_dialog", None)
    if callable(dialog):
        return cast(DialogDecorator, dialog)
    return None
