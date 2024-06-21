import warnings
warnings.filterwarnings("ignore")

import argparse
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
from get_model import get_qwen_model
from data import get_dataset, prepare_data
from huggingface_hub import login
import requests

def main(args):
    tokenizer, model = get_qwen_model(model_id = args.model_id)
    data = get_dataset(dataset_id=args.dataset_id)
    
    training_params = TrainingArguments(
    output_dir="./model",
    save_strategy="steps",
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    save_steps=1,
    logging_steps=500,
    learning_rate=1e-4,
    push_to_hub = True,
    hub_model_id = f"{args.model_id.split('/')[1]}-SQL-generator",
    push_to_hub_model_id = f"{args.model_id.split('/')[1]}-SQL-generator"
    )
    
    response_template = " ### The response query is:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
    model,
    train_dataset=data,
    formatting_func=prepare_data,
    data_collator=collator,
    )
    
    trainer.train()
    trainer.save_model(f"./{args.model_id.split('/')[1]}")
    
    trainer.push_to_hub(f"omaratef3221/{args.model_id.split('/')[1]}-SQL-generator")
    
    requests.post("https://ntfy.sh/sql_query_generator_llm", data="Model Trained Uploaded to HuggingFace ".encode(encoding='utf-8'))

    
if __name__ == "__main__":
    # model_id, dataset_id, epochs, batch_size
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='Qwen/Qwen2-0.5B-Instruct')
    parser.add_argument('--dataset_id', type=str, default='motherduckdb/duckdb-text2sql-25k')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--hf_login', type=bool, default=True)
    args = parser.parse_args()
    if args.hf_login:
        login()
    else:
        pass
    main(args)