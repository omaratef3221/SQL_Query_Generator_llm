from transformers import AutoTokenizer, AutoModelForCausalLM



def get_qwen_model(model_id = "Qwen/Qwen2-1.5B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    return tokenizer, model