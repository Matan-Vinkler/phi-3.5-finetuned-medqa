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
