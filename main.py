#!/usr/bin/env python3
"""
qguard.py
Run guard evaluation (single prompt) with optional images using Yes-probability PageRank aggregation.

Usage:
  python qguard.py \
    --model_path OpenGVLab/InternVL2_5-4B \
    --guard_questions_json guard_questions.json \
    --prompt "Does the content encourage harm?" \
    --images img1.jpg img2.png \
    --out_json result.json
"""

import argparse
import json
from typing import Dict, List, Any
import torch

from qguard.config import RunConfig
from qguard.seed import set_seed
from qguard.modeling import load_internvl_text_or_mm
from qguard.pipeline import evaluate_prompt_with_pagerank, DEFAULT_ALLOWED_GROUPS
from qguard.vision import load_images_batch


def main():
    parser = argparse.ArgumentParser(
        description="Guard evaluation via Yes probability + PageRank (single prompt, text + preprocessed images)."
    )
    parser.add_argument("--model_path", type=str, default="OpenGVLab/InternVL2_5-4B")
    parser.add_argument("--guard_questions_json", type=str, required=True,
                        help="Path to guard-questions JSON: {category: [question, ...], ...}")
    parser.add_argument("--prompt", type=str, required=True, help="Single prompt to evaluate")
    parser.add_argument("--images", type=str, nargs="*", default=[],
                        help="Optional image file paths to include as context")
    parser.add_argument("--image_input_size", type=int, default=448, help="Tile size for dynamic preprocessing")
    parser.add_argument("--image_max_num", type=int, default=12, help="Max tiles per image (thumbnail appended if >1)")
    parser.add_argument("--allowed_groups", type=str, nargs="*", default=DEFAULT_ALLOWED_GROUPS,
                        help="Categories used for PageRank aggregation")
    parser.add_argument("--threshold", type=float, default=0.50, help="Risk threshold for 'harmful'")
    parser.add_argument("--out_json", type=str, default="pagerank_guard_eval_output.json")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    cfg = RunConfig(
        model_path=args.model_path,
        guard_questions_json=args.guard_questions_json,
        output_json=args.out_json,
        threshold=args.threshold,
        seed=args.seed,
    )
    cfg.apply_env()
    set_seed(cfg.seed)

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Loading model/tokenizer...")
    model, tokenizer = load_internvl_text_or_mm(cfg.model_path)
    print("Ready.")

    with open(cfg.guard_questions_json, "r", encoding="utf-8") as f:
        guard_qs: Dict[str, List[str]] = json.load(f)

    # Preprocess images into (N, 3, H, W) with dynamic tiling + ImageNet normalization
    images_tensor = load_images_batch(
        args.images, input_size=args.image_input_size, max_num=args.image_max_num
    )

    result: Dict[str, Any] = evaluate_prompt_with_pagerank(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        guard_questions=guard_qs,
        allowed_groups=args.allowed_groups,
        threshold=cfg.threshold,
        images_tensor=images_tensor,
    )

    with open(cfg.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nPrediction: {result['prediction']}")
    print(f"Risk score: {result['risk_score']:.4f} (threshold={result['threshold']})")
    print(f"Saved: {cfg.output_json}")


if __name__ == "__main__":
    main()
