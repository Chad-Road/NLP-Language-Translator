from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom")

model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom",
    device_map="auto",
    torch_dtype="auto"
    )