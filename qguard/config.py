from dataclasses import dataclass

@dataclass
class RunConfig:
    """Runtime configuration with sensible defaults."""
    model_path: str = "OpenGVLab/InternVL2_5-4B"
    guard_questions_json: str = "guard_questions.json"
    output_json: str = "pagerank_guard_eval_output.json"
    threshold: float = 0.50  # tuned lower for probability weights
    seed: int = 1234

    # environment knobs
    cuda_launch_blocking: str = "1"
    pytorch_cuda_alloc_conf: str = "max_split_size_mb:512"

    def apply_env(self) -> None:
        import os
        os.environ["CUDA_LAUNCH_BLOCKING"] = self.cuda_launch_blocking
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = self.pytorch_cuda_alloc_conf
