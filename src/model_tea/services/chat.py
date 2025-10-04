import logging
import torch
from pathlib import Path
from typing import Optional, Dict, Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, models_dir: str = "iterative_models"):
        self.models_dir = Path(models_dir)
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.default_max_length = 250
        self.default_temperature = 0.5
        self.default_top_p = 0.85
        self.default_top_k = 30
        self.default_repetition_penalty = 1.4

    def list_available_models(self, show_all: bool = False) -> List[str]:
        if not self.models_dir.exists():
            return []

        models = []
        final_models = []

        for item in self.models_dir.iterdir():
            if item.is_dir():
                if (item / "config.json").exists() or (item / "pytorch_model.bin").exists() or (item / "model.safetensors").exists():
                    models.append(str(item.relative_to(self.models_dir)))
                else:
                    for subitem in item.iterdir():
                        if subitem.is_dir():
                            if (subitem / "config.json").exists() or (subitem / "pytorch_model.bin").exists() or (subitem / "model.safetensors").exists():
                                rel_path = str(subitem.relative_to(self.models_dir))
                                if subitem.name == "final":
                                    final_models.append(rel_path)
                                elif show_all:
                                    models.append(rel_path)

        all_models = sorted(final_models) + sorted(models)
        return all_models if all_models else []

    def load_model(self, model_name: str) -> bool:
        model_path = self.models_dir / model_name

        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return False

        try:
            logger.info(f"Loading model from {model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.float32
            )
            self.model.to(self.device)
            self.model.eval()
            self.model_name = model_name
            logger.info(f"Model loaded successfully on {self.device}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None
    ) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        max_length = max_length or self.default_max_length
        temperature = temperature or self.default_temperature
        top_p = top_p or self.default_top_p
        top_k = top_k or self.default_top_k
        repetition_penalty = repetition_penalty or self.default_repetition_penalty

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        model_path = self.models_dir / model_name
        info = {
            "name": model_name,
            "path": str(model_path),
            "exists": model_path.exists()
        }

        if model_path.exists():
            total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
            info["size_mb"] = round(total_size / (1024 * 1024), 2)

            config_path = model_path / "config.json"
            if config_path.exists():
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    info["model_type"] = config.get("model_type", "unknown")
                    info["vocab_size"] = config.get("vocab_size", 0)

        return info

    def unload_model(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
