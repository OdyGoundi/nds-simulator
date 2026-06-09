from __future__ import annotations

import base64
from pathlib import Path
from typing import List, Optional

import streamlit as st


def _image_path_to_data_uri(image_path: Path) -> Optional[str]:
    if not image_path.exists():
        return None
    suffix = image_path.suffix.lower()
    mime = "image/png"
    if suffix in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"
    data_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data_b64}"


def _pick_latest_existing_path(paths: List[Path]) -> Optional[Path]:
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


def _get_runtime_theme_base() -> str:
    context_obj = getattr(st, "context", None)
    if context_obj is not None:
        theme_obj = getattr(context_obj, "theme", None)
        if isinstance(theme_obj, dict):
            base_val = theme_obj.get("base")
            if base_val is not None:
                return str(base_val).strip().lower()
        elif theme_obj is not None:
            base_attr = getattr(theme_obj, "base", None)
            if base_attr is not None:
                return str(base_attr).strip().lower()
    return ""


def _logo_candidates(project_root: Path) -> List[Path]:
    return [
        project_root / "docs" / "thesis" / "figures" / "new_logo.png",
        project_root / "docs" / "assets" / "new_logo.png",
    ]


def render_header_logo(project_root: Path, width_px: int = 196, align: str = "center") -> bool:
    candidates = _logo_candidates(project_root)
    light_logo_path = _pick_latest_existing_path(candidates)
    dark_logo_path = _pick_latest_existing_path(candidates)
    light_logo_uri = _image_path_to_data_uri(light_logo_path) if light_logo_path is not None else None
    dark_logo_uri = _image_path_to_data_uri(dark_logo_path) if dark_logo_path is not None else None
    if light_logo_uri is None and dark_logo_uri is None:
        return False
    if light_logo_uri is None:
        light_logo_uri = dark_logo_uri
    if dark_logo_uri is None:
        dark_logo_uri = light_logo_uri
    if light_logo_uri is None or dark_logo_uri is None:
        return False

    css_align = "left" if str(align).strip().lower() == "left" else "center"
    runtime_theme = _get_runtime_theme_base()
    light_default_display = "none" if runtime_theme == "dark" else "inline-block"
    dark_default_display = "inline-block" if runtime_theme == "dark" else "none"

    st.markdown(
        f"""
<style>
.dynasim-header-logo-wrap {{
  width: 100%;
  text-align: {css_align};
}}
.dynasim-header-logo-wrap img {{
  width: {int(width_px)}px;
  height: auto;
}}
.dynasim-header-logo-dark {{
  display: {dark_default_display};
}}
.dynasim-header-logo-light {{
  display: {light_default_display};
}}
html[data-theme="dark"] .dynasim-header-logo-light,
html[theme="dark"] .dynasim-header-logo-light,
body[data-theme="dark"] .dynasim-header-logo-light,
body[theme="dark"] .dynasim-header-logo-light,
body.dark .dynasim-header-logo-light {{
  display: none !important;
}}
html[data-theme="dark"] .dynasim-header-logo-dark,
html[theme="dark"] .dynasim-header-logo-dark,
body[data-theme="dark"] .dynasim-header-logo-dark,
body[theme="dark"] .dynasim-header-logo-dark,
body.dark .dynasim-header-logo-dark {{
  display: inline-block !important;
}}
</style>
<div class="dynasim-header-logo-wrap">
  <img class="dynasim-header-logo-light" src="{light_logo_uri}" alt="dynaSim logo">
  <img class="dynasim-header-logo-dark" src="{dark_logo_uri}" alt="dynaSim logo dark">
</div>
        """,
        unsafe_allow_html=True,
    )
    return True
