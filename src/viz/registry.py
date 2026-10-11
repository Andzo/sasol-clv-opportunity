"""Registry and output manager for EDA figures and quantitative findings."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import plotly.graph_objects as go

logger = logging.getLogger("src.viz.registry")


@dataclass
class FigureMetadata:
    fig_id: str
    section: str
    title: str
    is_interactive: bool
    findings: dict[str, Any] = field(default_factory=dict)
    path: str = ""


class VisualRegistry:
    """Manages plotting routines, automated figure persistence, and finding logs."""

    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        self.registry: dict[str, tuple[Callable[..., Any], FigureMetadata]] = {}
        self.manifest: list[FigureMetadata] = []

    def register(self, fig_id: str, section: str, title: str, is_interactive: bool = False):
        """Decorator to register a figure generation function."""
        def decorator(func: Callable[..., Any]):
            meta = FigureMetadata(
                fig_id=fig_id,
                section=section,
                title=title,
                is_interactive=is_interactive,
            )
            self.registry[fig_id] = (func, meta)
            return func
        return decorator

    def run_and_save(self, fig_id: str, *args, **kwargs) -> tuple[Any, dict[str, Any]]:
        """Execute a registered plot, save artifact, and log findings."""
        if fig_id not in self.registry:
            raise KeyError(f"Plot {fig_id!r} is not registered")

        func, meta = self.registry[fig_id]
        sec_dir = self.output_root / meta.section
        sec_dir.mkdir(parents=True, exist_ok=True)

        res = func(*args, **kwargs)
        if isinstance(res, tuple) and len(res) == 2:
            fig, findings = res
        else:
            fig, findings = res, {}

        meta.findings = findings

        if meta.is_interactive and isinstance(fig, go.Figure):
            out_file = sec_dir / f"{meta.fig_id}.html"
            fig.write_html(str(out_file), include_plotlyjs="cdn")
            meta.path = str(out_file.relative_to(self.output_root))
            logger.info("Saved interactive figure %s to %s", meta.fig_id, out_file)
        elif isinstance(fig, plt.Figure):
            out_file = sec_dir / f"{meta.fig_id}.png"
            fig.savefig(out_file, dpi=150, bbox_inches="tight")
            plt.close(fig)
            meta.path = str(out_file.relative_to(self.output_root))
            logger.info("Saved static figure %s to %s", meta.fig_id, out_file)
        else:
            logger.warning("Unrecognized figure type for %s: %s", meta.fig_id, type(fig))

        self.manifest.append(meta)
        return fig, findings

    def save_manifest(self) -> Path:
        """Write manifest.json containing metadata and quantitative insights."""
        manifest_path = self.output_root / "manifest.json"
        manifest_data = [asdict(m) for m in self.manifest]
        manifest_path.write_text(json.dumps(manifest_data, indent=2))
        logger.info("Saved EDA manifest with %d figures to %s", len(self.manifest), manifest_path)
        return manifest_path
