from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ModelConfig:
    model_name: str = "meta-llama/Llama-2-7b-chat-hf" # Placeholder
    max_length: int = 512
    device: str = "cuda" # Placeholder, ensure it doesn't try to access hw
    load_in_8bit: bool = True # Placeholder
    trust_remote_code: bool = True

@dataclass 
class LoRAConfig:
    r: int = 8
    lora_alpha: int = 16
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM" # Placeholder

@dataclass
class TrainingConfig:
    output_dir: str = "./models/lora_adapters"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    learning_rate: float = 2e-4
    fp16: bool = True # Placeholder
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    evaluation_strategy: str = "steps"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    report_to: str = "wandb" # Placeholder, ensure no actual wandb calls
    run_name: str = "nhs-navigator-lora"
