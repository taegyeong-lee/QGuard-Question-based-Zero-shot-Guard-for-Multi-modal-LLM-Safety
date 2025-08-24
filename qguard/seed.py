def set_seed(seed: int = 1234) -> None:
    """Best-effort seeding across common libs."""
    try:
        import random, numpy as np, torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
