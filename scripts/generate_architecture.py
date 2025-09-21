"""
Generate a high-level tool architecture diagram for this project.

Outputs:
- docs/project_architecture.dot
- docs/project_architecture.png (if Graphviz is available)
- docs/project_architecture.svg (if Graphviz is available)

The diagram groups components into clusters similar to the provided example:
- UI (Streamlit app)
- Core Library (dataset, preprocess, bias, pipeline, models, eval)
- External ML Backends (scikit-learn, LightGBM, CatBoost)

This script has no hard dependency on the Python graphviz package; it will
render via the `dot` CLI if present or just write the DOT file otherwise.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import List


@dataclass
class RenderResult:
    wrote_dot: bool
    wrote_png: bool
    wrote_svg: bool
    output_dir: str
    base_name: str


def _ensure_output_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def _project_has(path: str) -> bool:
    return os.path.exists(path)


def build_dot_detailed(dpi: int) -> str:
    """Return DOT source describing the project's architecture (detailed)."""
    # Detect optional modules/files to tailor the diagram
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    src = os.path.join(root, "src")
    has_dataset = _project_has(os.path.join(src, "dataset.py"))
    has_preprocess = _project_has(os.path.join(src, "preprocess.py"))
    has_pipeline = _project_has(os.path.join(src, "pipeline.py"))
    has_models = _project_has(os.path.join(src, "models.py"))
    has_eval = _project_has(os.path.join(src, "eval.py"))
    has_visualize = _project_has(os.path.join(src, "visualize.py"))
    has_app = _project_has(os.path.join(root, "app.py"))

    dot: List[str] = []
    dot.append("digraph G {")
    dot.append("  rankdir=LR;")
    dot.append(f"  graph [dpi={dpi}];")
    dot.append("  compound=true;")
    dot.append('  node [shape=box, style=filled, fontname="Helvetica", color="#1f2937", fillcolor="#f3f4f6"];')

    # UI cluster (top lane analogue)
    dot.append('  subgraph cluster_ui {')
    dot.append('    label="Streamlit UI";')
    dot.append('    color="#9ca3af";')
    dot.append('    style="rounded";')
    if has_app:
        dot.append('    ui_data [label="Data UI\n(Data Generation / Upload)", fillcolor="#e0f2fe"];')
        dot.append('    ui_builder [label="Pipeline Builder\n(Steps & Model)", fillcolor="#e0f2fe"];')
        dot.append('    ui_results [label="Result Viewer\n(Metrics / Plots)", fillcolor="#e0f2fe"];')
        dot.append('    ui_compare [label="Comparison View", fillcolor="#e0f2fe"];')
    else:
        dot.append('    ui_placeholder [label="UI", fillcolor="#e0f2fe"];')
    dot.append('  }')

    # Core cluster (controller/API client analogue)
    dot.append('  subgraph cluster_core {')
    dot.append('    label="Core Library (src/*)";')
    dot.append('    color="#10b981";')
    dot.append('    style="rounded,dashed";')
    if has_dataset:
        dot.append('    ds [label="Dataset\n(src/dataset.py)", fillcolor="#dcfce7"];')
        dot.append('    bias [label="Bias Injectors\n(Representation / Measurement / Sampling / Label)", fillcolor="#fef9c3"];')
    if has_preprocess:
        dot.append('    pre [label="Preprocess\n(MissingImputer / Skewness / Binner)", fillcolor="#dbeafe"];')
    if has_pipeline:
        dot.append('    pipe [label="Pipeline\n(src/pipeline.Pipeline)", fillcolor="#ede9fe"];')
    if has_models:
        dot.append('    mdl [label="Models\n(Ensemble / Stacking / sklearn)", fillcolor="#fae8ff"];')
    if has_eval:
        dot.append('    ev [label="Evaluation\n(metrics, curves)", fillcolor="#ffe4e6"];')
    if has_visualize:
        dot.append('    viz [label="Visualization\n(matplotlib / seaborn)", fillcolor="#f5f5f4"];')
    dot.append('  }')

    # External backends cluster (external API analogue)
    dot.append('  subgraph cluster_ext {')
    dot.append('    label="External ML Backends";')
    dot.append('    color="#a78bfa";')
    dot.append('    style="rounded,dashed";')
    dot.append('    skl [label="scikit-learn", fillcolor="#ffffff"];')
    dot.append('    lgbm [label="LightGBM", fillcolor="#ffffff"];')
    dot.append('    cat [label="CatBoost", fillcolor="#ffffff"];')
    dot.append('  }')

    # Flow edges (use labels to mirror the example diagram semantics)
    if has_app and has_dataset:
        dot.append('  ui_data -> ds [label="dataset config / upload"];')
    if has_app and has_preprocess:
        dot.append('  ui_builder -> pre [label="preprocess steps"];')
    if has_app and has_dataset:
        dot.append('  ui_builder -> bias [label="bias steps"];')
    if has_preprocess and has_pipeline:
        dot.append('  pre -> pipe [label="transformations", lhead=cluster_core];')
    if has_dataset and has_pipeline:
        dot.append('  ds -> pipe [label="dataframe", lhead=cluster_core];')
    if has_dataset and has_preprocess:
        dot.append('  ds -> pre [style=dashed, label="optional"];')
    if has_pipeline and has_models:
        dot.append('  pipe -> mdl [label="fit/predict"];')
    if has_models and has_eval:
        dot.append('  mdl -> ev [label="predictions"];')
    if has_eval and has_app:
        dot.append('  ev -> ui_results [label="metrics / plots"];')
    if has_pipeline and has_app:
        dot.append('  pipe -> ui_results [style=dashed, label="pipeline diagram"];')
    if has_visualize and has_app:
        dot.append('  viz -> ui_results [style=dotted, label="charts"];')

    # Dependencies to external backends
    if has_models:
        dot.append('  mdl -> skl [style=dotted, label="estimators"];')
        dot.append('  mdl -> lgbm [style=dotted];')
        dot.append('  mdl -> cat [style=dotted];')

    dot.append("}")
    return "\n".join(dot)


def build_dot_simple(dpi: int) -> str:
    """Return a minimal, paper-ready DOT diagram similar to the example."""
    dot: List[str] = []
    dot.append("digraph G {")
    dot.append("  rankdir=LR;")
    dot.append('  node [shape=box, style=filled, fontname="Helvetica", color="#111827", fillcolor="#ffffff"];')
    dot.append(f"  graph [dpi={dpi}];")

    # Top lane: UI (simple)
    dot.append('  subgraph cluster_ui {')
    dot.append('    label="Tool UI";')
    dot.append('    color="#9ca3af";')
    dot.append('    style="rounded";')
    dot.append('    ui [label="Streamlit App", fillcolor="#eef2ff"];')
    dot.append('    proc [label="Processor", fillcolor="#e0f2fe"];')
    dot.append('    viewer [label="Result Viewer", fillcolor="#fef9c3"];')
    dot.append('    ui -> proc -> viewer;')
    dot.append('  }')

    # Bottom lane: Core Engine
    dot.append('  subgraph cluster_core {')
    dot.append('    label="Core Engine";')
    dot.append('    color="#10b981";')
    dot.append('    style="rounded,dashed";')
    dot.append('    ctrl [label="Controller", fillcolor="#dcfce7"];')
    dot.append('    pipe [label="Pipeline", fillcolor="#ede9fe"];')
    dot.append('    client [label="Model / Eval", fillcolor="#fae8ff"];')
    dot.append('    ctrl -> pipe -> client;')
    dot.append('  }')

    # External API/backends cluster (single block)
    dot.append('  subgraph cluster_ext {')
    dot.append('    label="ML Backends";')
    dot.append('    color="#a78bfa";')
    dot.append('    style="rounded,dashed";')
    dot.append('    back [label="scikit-learn / LightGBM / CatBoost", fillcolor="#ffffff"];')
    dot.append('  }')

    # Cross-lane flows
    dot.append('  ui -> ctrl [ltail=cluster_ui, lhead=cluster_core, label="config + data"];')
    dot.append('  client -> viewer [ltail=cluster_core, lhead=cluster_ui, label="metrics / images"];')
    dot.append('  client -> back [style=dotted, label="fit / predict"];')

    dot.append("}")
    return "\n".join(dot)


def render(dot_src: str, output_dir: str, base_name: str) -> RenderResult:
    _ensure_output_dir(output_dir)
    dot_path = os.path.join(output_dir, f"{base_name}.dot")
    with open(dot_path, "w", encoding="utf-8") as f:
        f.write(dot_src)

    wrote_png = False
    wrote_svg = False

    # Prefer python-graphviz if installed
    try:
        import graphviz  # type: ignore

        g = graphviz.Source(dot_src, filename=base_name, directory=output_dir, format="png")
        g.render(cleanup=True)
        wrote_png = True
        g = graphviz.Source(dot_src, filename=base_name, directory=output_dir, format="svg")
        g.render(cleanup=True)
        wrote_svg = True
    except Exception:
        # Fallback to dot CLI if available
        dot_bin = shutil.which("dot")
        if dot_bin:
            try:
                png_path = os.path.join(output_dir, f"{base_name}.png")
                svg_path = os.path.join(output_dir, f"{base_name}.svg")
                subprocess.run([dot_bin, "-Tpng", dot_path, "-o", png_path], check=True)
                subprocess.run([dot_bin, "-Tsvg", dot_path, "-o", svg_path], check=True)
                wrote_png = True
                wrote_svg = True
            except Exception:
                pass

    return RenderResult(True, wrote_png, wrote_svg, output_dir, base_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate architecture diagram for the project")
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), os.pardir, "docs"))
    parser.add_argument("--base-name", default="project_architecture")
    parser.add_argument("--mode", choices=["detailed", "simple"], default="detailed")
    parser.add_argument("--dpi", type=int, default=200, help="Raster output DPI for PNG (e.g., 300 or 600)")
    args = parser.parse_args()

    dot_src = build_dot_simple(args.dpi) if args.mode == "simple" else build_dot_detailed(args.dpi)
    res = render(dot_src, os.path.abspath(args.output_dir), args.base_name)

    print(f"DOT written: {os.path.join(res.output_dir, res.base_name + '.dot')}")
    if res.wrote_png:
        print(f"PNG written: {os.path.join(res.output_dir, res.base_name + '.png')}")
    else:
        print("PNG not generated (graphviz not installed). You can run: dot -Tpng docs/project_architecture.dot -o docs/project_architecture.png")
    if res.wrote_svg:
        print(f"SVG written: {os.path.join(res.output_dir, res.base_name + '.svg')}")


if __name__ == "__main__":
    main()


