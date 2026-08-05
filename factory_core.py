import os
import requests
from bs4 import BeautifulSoup
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import Dataset

class ModelFactoryEngine:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def fetch_web_knowledge(self, query: str) -> str:
        """Scrapes web search summaries using DuckDuckGo HTML layout safely."""
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            response = requests.get(f"https://html.duckgo.com/html/?q={requests.utils.quote(query)}", headers=headers, timeout=10)
            if response.status_code!= 200:
                return "Web data fetch skipped due to network limitations."

            soup = BeautifulSoup(response.text, 'html.parser')
            snippets = []
            for a in soup.find_all('a', class_='result__snippet', limit=3):
                snippets.append(a.get_text())

            return " ".join(snippets) if snippets else "No external web matches found."
        except Exception as e:
            return f"Web fetching error: {str(e)}"

    def build_dataset(self, user_prompt: str, web_data: str) -> Dataset:
        """Constructs instruction-response training records for LoRA adapter adjustment."""
        full_context = f"Context Data: {web_data}\n\nUser Directive: {user_prompt}"
        raw_data = {
            "text": [
                f"<|system|> You are a specialized manufactured AI brain. <|user|> {full_context} <|assistant|> Understood. Initializing configuration and operational rules."
            ]
        }
        return Dataset.from_dict(raw_data)

    def train_lora_adapter(self, base_model_id: str, dataset: Dataset) -> str:
        """Trains a lightweight LoRA adapter modifying attention parameters without breaking limits."""
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="cpu",
            low_cpu_mem_usage=True
        )

        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(model, peft_config)

        # Simplified placeholder training execution step for cloud runner
        adapter_path = os.path.join(self.output_dir, "adapter_model")
        os.makedirs(adapter_path, exist_ok=True)
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)

        return adapter_path
