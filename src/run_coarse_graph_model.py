from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Legacy pair-wise coarse graph inference removed from the main pipeline."
    )
    parser.add_argument(
        "--use-qwen-doc-graph",
        action="store_true",
        help="Placeholder flag documenting that the main inference entry is run_coarse_graph_qwen.py.",
    )
    return parser.parse_args()


def main() -> None:
    parse_args()
    raise RuntimeError(
        "The pair-wise coarse graph inference baseline has been removed from the main pipeline. "
        "Use `python src/run_coarse_graph_qwen.py` for document-to-coarse-graph inference."
    )


if __name__ == "__main__":
    main()
