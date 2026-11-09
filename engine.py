import threading
import ctypes
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .config import MODEL_ID, MAX_NEW_TOKENS, MAX_HISTORY

class LiveVisionEngine:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.lock = threading.Lock()
        self.loading_error = None
        self.cpp_helper = self._load_cpp_helper()

    def _load_cpp_helper(self):
        """Load the compiled C++ native helper (fast_ops.cpp) for local performance."""
        try:
            # Look for the compiled shared library in the current directory or cpp folder
            for path in ["libfast_ops.so", "cpp/libfast_ops.so", "./libfast_ops.so"]:
                if os.path.exists(path):
                    helper = ctypes.CDLL(os.path.abspath(path))
                    helper.count_words.argtypes = [ctypes.c_char_p]
                    helper.count_words.restype = ctypes.c_int
                    print(f"[Engine] Successfully loaded C++ native module from {path}")
                    return helper
        except Exception as exc:
            print(f"[Engine] C++ native module load skipped/failed: {exc}")
        return None

    def load(self):
        if self.model is not None:
            return
        with self.lock:
            if self.model is not None:
                return
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
                self.model = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    low_cpu_mem_usage=True,
                )
                if not torch.cuda.is_available():
                    self.model.to("cpu")
            except Exception as exc:
                self.loading_error = str(exc)
                raise

    def count_text_words(self, text: str) -> int:
        """Uses the C++ native module if available, otherwise falls back to Python."""
        if self.cpp_helper:
            return self.cpp_helper.count_words(text.encode('utf-8'))
        return len(text.split())

    def reply(self, history, message, image_context=None):
        self.load()
        
        # System prompt tailored for LiveVision's image-to-script specialization
        system_content = (
            "You are LiveVision, a modular, zero-cost AI assistant specialized in "
            "image-to-script generation and interactive character workflows. "
            "Analyze inputs accurately and write clear, engaging scripts."
        )
        
        messages = [{"role": "system", "content": system_content}]
        messages.extend(history[-MAX_HISTORY:])
        
        # If an image context or upload path is provided, inject it into the prompt workflow
        user_prompt = message
        if image_context:
            user_prompt = f"[Attached Image Context: {image_context}] {message}"
            
        messages.append({"role": "user", "content": user_prompt})
        
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True, temperature=0.7, top_p=0.9,
                repetition_penalty=1.08,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

ai = LiveVisionEngine()
      
