from typing import Dict, Any, List, Tuple, Optional
import torch
from .token_utils import yes_no_probs_from_logits

def _model_device_dtype(model):
    try:
        p = next(model.parameters())
        return p.device, p.dtype
    except StopIteration:
        dev = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        dt = getattr(model, "dtype", torch.bfloat16 if torch.cuda.is_available() else torch.float32)
        return dev, dt

_PATCHED = False
def _ensure_tuple_generate(model):
    global _PATCHED
    if _PATCHED:
        return

    lm = getattr(model, "language_model", None)
    if lm is None or not hasattr(lm, "generate"):
        return

    orig_generate = lm.generate

    def wrapped_generate(*args, **kwargs):
        kwargs.setdefault("return_dict_in_generate", True)
        kwargs.setdefault("output_scores", True)
        out = orig_generate(*args, **kwargs)

        if hasattr(out, "sequences") and hasattr(out, "scores"):
            sequences = out.sequences
            scores = out.scores
            return sequences, scores

        if isinstance(out, (tuple, list)) and len(out) == 2:
            return out[0], out[1]

        sequences = getattr(out, "sequences", out)
        scores = getattr(out, "scores", None)
        return sequences, scores

    lm.generate = wrapped_generate
    _PATCHED = True

@torch.no_grad()
def _first_step_logits_via_chat(
    model,
    tokenizer,
    question_text: str,
    images: torch.Tensor, 
    max_new_tokens: int = 1,
) -> Tuple[torch.Tensor, str]: 
    _ensure_tuple_generate(model)

    gen_cfg = dict(max_new_tokens=max_new_tokens, do_sample=False)

    model_device, model_dtype = _model_device_dtype(model)
    if model_device.type == "cpu" and model_dtype == torch.bfloat16:
        model_dtype = torch.float32
    images = images.to(device=model_device, dtype=model_dtype).contiguous()

    response, logits = model.chat(
        tokenizer,
        images,
        question_text,
        gen_cfg,
        history=None
    )

    step0 = logits[0][0] if isinstance(logits, (list, tuple)) else logits[0]
    if not isinstance(step0, torch.Tensor):
        step0 = torch.tensor(step0, device=model_device, dtype=torch.float32)

    return step0, (response or "")


@torch.no_grad()
def _first_step_logits_text_only(
    model,
    tokenizer,
    question_text: str,
    max_new_tokens: int = 1,
) -> Tuple[torch.Tensor, str]: 
    _ensure_tuple_generate(model)

    gen_cfg = dict(max_new_tokens=max_new_tokens, do_sample=False)

    model_device, model_dtype = _model_device_dtype(model)
    if model_device.type == "cpu" and model_dtype == torch.bfloat16:
        model_dtype = torch.float32
    images = None
    response, logits = model.chat(
        tokenizer,
        images,
        question_text,
        gen_cfg,
        history=None
    )

    step0 = logits[0][0] if isinstance(logits, (list, tuple)) else logits[0]
    if not isinstance(step0, torch.Tensor):
        step0 = torch.tensor(step0, device=model_device, dtype=torch.float32)

    return step0, (response or "")


def score_guard_question_yes_prob(
    model,
    tokenizer,
    guard_q: str,
    prompt_text: str,
    yes_ids: List[int],
    no_ids: List[int],
    images: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    q_full = f"{guard_q} (You must answer with only Yes or No).\n\nprompt: {prompt_text}"

    if images is not None:
        step0, decoded = _first_step_logits_via_chat(model, tokenizer, q_full, images, max_new_tokens=4)
    else:
        step0, decoded = _first_step_logits_text_only(model, tokenizer, q_full, max_new_tokens=4)

    y_prob, n_prob = yes_no_probs_from_logits(step0, yes_ids, no_ids)

    return {
        "question": q_full,
        "yes_prob": y_prob,
        "no_prob": n_prob,
        "response": decoded,
    }