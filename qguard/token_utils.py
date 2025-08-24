from typing import List, Tuple
import torch

def gather_yes_no_ids(tok) -> Tuple[List[int], List[int]]:
    candidates = [
        " Yes", "Yes", " yes", "Yes.", " Yes.", " yes.",
        " No",  "No",  " no",  "No.",  " No.",  " no."
    ]
    id_map = {t: tok.encode(t, add_special_tokens=False) for t in candidates}
    yes_ids = [ids[0] for t, ids in id_map.items()
               if t.strip().lower().startswith("yes") and len(ids) == 1]
    no_ids  = [ids[0] for t, ids in id_map.items()
               if t.strip().lower().startswith("no")  and len(ids) == 1]
    yes_ids, no_ids = sorted(set(yes_ids)), sorted(set(no_ids))
    if not yes_ids or not no_ids:
        raise RuntimeError(f"Failed to find single-token Yes/No IDs: yes={yes_ids}, no={no_ids}")
    return yes_ids, no_ids


def yes_no_probs_from_logits(
    logits_step0: torch.Tensor, yes_ids: List[int], no_ids: List[int]
) -> Tuple[float, float]:
    logits_step0 = logits_step0.float()
    y = torch.logsumexp(logits_step0[yes_ids], dim=0)
    n = torch.logsumexp(logits_step0[no_ids],  dim=0)
    probs = torch.softmax(torch.stack([y, n]), dim=0)
    return float(probs[0].item()), float(probs[1].item())
