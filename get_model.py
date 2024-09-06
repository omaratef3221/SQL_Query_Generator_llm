from transformers import AutoTokenizer, AutoModelForCausalLM
from CustomQwen2ForCausalLM import CustomQwen2ForCausalLM


def get_qwen_model(model_id = "Qwen/Qwen2-0.5B-Instruct"):
    
    
    if model_id == "Omaratef3221/qwen-0.5-rbf-mlp":
        model = CustomQwen2ForCausalLM.from_pretrained("Omaratef3221/qwen-0.5-rbf-mlp")
        tokenizer = AutoTokenizer.from_pretrained("Omaratef3221/qwen-0.5-rbf-mlp", force_download = True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
    print("======= DOWNLOADED MODEL =======")
    print(model)
    print("="*30)
    return tokenizer, model