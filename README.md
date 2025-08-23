# 🧑‍⚕️ Fine-Tuning Phi-3.5-Mini-Instruct as a Medical Assistant

This repository contains the code and Colab notebook used to fine-tune **[Phi-3.5-Mini-Instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)** with **LoRA (Low-Rank Adaptation)** for a domain-specific task: building a lightweight **medical assistant**.

## 📌 Project Overview
- **Base model:** Phi-3.5-Mini-Instruct (1.3B parameters)
- **Fine-tuning method:** LoRA + 4-bit quantization (memory efficient, Colab-friendly)
- **Dataset:** [medalpaca/medical_meadow_medqa](https://huggingface.co/datasets/medalpaca/medical_meadow_medqa)
- **Task:** Instruction tuning for medical question-answering
- **Environment:** Google Colab (T4 GPU)

The resulting model generates **concise, factual responses** to medical-style instructions.

⚠️ **Disclaimer:** This model is for **educational purposes only**. It is **not intended for clinical use** or for making medical decisions.

## 🚀 Fine-Tuned Model
The fine-tuned model is available on Hugging Face Hub:  

👉 [Matanvinkler18/phi-3.5-finetuned-medqa](https://huggingface.co/Matanvinkler18/phi-3.5-finetuned-medqa)

## 🛠️ Features
- End-to-end training pipeline in **Colab notebook**
- **Instruction → Response** dataset formatting
- LoRA fine-tuning with Hugging Face **TRL `SFTTrainer`**
- Automatic push to Hugging Face Hub
- Inference examples (pipeline & generate API)

## 📂 Repository Structure
```

├── notebook.ipynb       # Colab notebook with training + inference
├── README.md            # This file

```

## 📜 Usage

### 1. Training
Run the Colab notebook to:
- Load and format the dataset
- Fine-tune with LoRA
- Push results to Hugging Face Hub

### 2. Inference
Quick test with Hugging Face `pipeline`:

```python
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "microsoft/Phi-3.5-mini-instruct"
ADAPTER = "Matanvinkler18/phi-3.5-finetuned-medqa"

tokenizer = AutoTokenizer.from_pretrained(BASE)
base_model = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto", load_in_4bit=True)
model = PeftModel.from_pretrained(base_model, ADAPTER)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

prompt = """### Instruction:
List the common symptoms of iron deficiency anemia.

### Response:"""

print(pipe(prompt, max_new_tokens=256)[0]["generated_text"])
```

## ✅ Example Prompts

* Explain the difference between Type 1 and Type 2 diabetes
* List the common symptoms of iron deficiency anemia
* Give a step-by-step approach to evaluating a patient with chest pain
* What are safe first-aid steps for treating a minor burn?
* Define hypertension and provide the normal blood pressure range

## 📚 Citations

* Abdin, Mona et al. *Phi-3 Technical Report*. arXiv:2404.14219 (2024).
* Hu, Edward J. et al. *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685 (2021).
* Jin, Di, and Pan et al. *What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams*. arXiv preprint arXiv:2009.13081 (2020).
