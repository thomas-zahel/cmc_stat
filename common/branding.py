from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import streamlit as st


@dataclass(frozen=True)
class BrandingConfig:
    app_title: str
    header_title: Optional[str] = None
    header_subtitle: Optional[str] = None
    # External links are intentionally omitted for this repo.
    link_url: Optional[str] = None
    # Optional explicit favicon path. If None, we use static/common/brands/favicon-light-mode.ico if present.
    page_icon_path: Optional[str] = None
    # Layout mode: 'centered' or 'wide'
    layout: str = "centered"


def _repo_root_from_this_file() -> Path:
    # common/branding.py -> repo root is parent of common/
    return Path(__file__).resolve().parents[1]


def _b64(path: Path) -> Optional[str]:
    try:
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _roboto_css() -> str:
    root = _repo_root_from_this_file()
    fonts_dir = root / "static" / "common" / "fonts"

    # Keep it light: 3 weights, normal style.
    weights = {
        400: "roboto-latin-400-normal.woff2",
        500: "roboto-latin-500-normal.woff2",
        600: "roboto-latin-600-normal.woff2",
    }

    font_faces: list[str] = []
    for weight, filename in weights.items():
        payload = _b64(fonts_dir / filename)
        if not payload:
            continue
        font_faces.append(
            "\n".join(
                [
                    "@font-face {",
                    "  font-family: 'Roboto';",
                    "  font-style: normal;",
                    f"  font-weight: {weight};",
                    "  font-display: swap;",
                    f"  src: url(data:font/woff2;base64,{payload}) format('woff2');",
                    "}",
                ]
            )
        )

    # Apply Roboto broadly.
    base = "\n".join(
        [
            ":root {",
            "  --brand-header-bg: #FFFFFF;",
            "  --brand-header-fg: #0B2B3A;",
            "  --brand-muted: rgba(11,43,58,0.75);",
            "  --brand-border: rgba(11,43,58,0.12);",
            "}",
            "html, body, [class*='css'], [data-testid='stAppViewContainer'] {",
            "  font-family: 'Roboto', system-ui, -apple-system, 'Segoe UI', Arial, sans-serif;",
            "}",
            ".app-header {",
            "  position: sticky;",
            "  top: 0;",
            "  z-index: 100;",
            "  background: var(--brand-header-bg);",
            "  color: var(--brand-header-fg);",
            "  padding: 0.75rem 1rem;",
            "  margin: 0 0 1rem 0;",
            "  border: 1px solid var(--brand-border);",
            "  border-radius: 12px;",
            "  box-shadow: 0 1px 10px rgba(0,0,0,0.04);",
            "}",
            ".app-header-inner {",
            "  display: flex;",
            "  align-items: center;",
            "  gap: 1rem;",
            "}",
            ".app-header-title {",
            "  font-size: 1.1rem;",
            "  font-weight: 600;",
            "  line-height: 1.2;",
            "}",
            ".app-header-subtitle {",
            "  font-size: 0.85rem;",
            "  color: var(--brand-muted);",
            "  margin-top: 0.15rem;",
            "}",
            ".app-header-spacer { flex: 1; }",
        ]
    )

    return "\n\n".join(font_faces + [base])


def apply_branding(config: BrandingConfig) -> None:
    """Apply minimal shared branding: page config and Körber logo.

    This intentionally avoids injecting custom CSS so apps use Streamlit's default styling.
    Call this before any other Streamlit commands (including st.session_state usage).
    """

    root = _repo_root_from_this_file()
    default_favicon = root / "static" / "common" / "brands" / "favicon-light-mode.ico"
    page_icon = config.page_icon_path or (str(default_favicon) if default_favicon.exists() else None)

    # Must be first Streamlit call in the app.
    if page_icon is not None:
        st.set_page_config(page_title=config.app_title, layout=config.layout, page_icon=page_icon)
    else:
        st.set_page_config(page_title=config.app_title, layout=config.layout)

    logo_path = root / "static" / "common" / "brands" / "logo.svg"

    try:
        # If link_url is None, omit it entirely.
        if config.link_url:
            st.logo(
                str(logo_path),
                size="large",
                link=config.link_url,
                icon_image=str(logo_path),
            )
        else:
            st.logo(
                str(logo_path),
                size="large",
                icon_image=str(logo_path),
            )
    except Exception:
        # st.logo is available in newer Streamlit; if unavailable, ignore.
        pass

    # Use Streamlit's default typography (no custom HTML/CSS).
    if config.header_title or config.header_subtitle:
        st.title(config.header_title or config.app_title)
        if config.header_subtitle:
            st.caption(config.header_subtitle)
