import warnings
warnings.filterwarnings("ignore")


from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
from get_model import get_qwen_model
from data import get_dataset, prepare_data

def main():
    tokenizer, model = get_qwen_model(model_id = "Qwen/Qwen2-1.5B-Instruct")
    data = get_dataset(dataset_id="motherduckdb/duckdb-text2sql-25k")
    
    training_params = TrainingArguments(
    output_dir="./model",
    save_strategy="no",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    optim="paged_adamw_32bit",
    save_steps=0,
    logging_steps=500,
    learning_rate=1e-4,
    push_to_hub_model_id = "omaratef3221/Qwen2-1.5B-Instruct-SQL-generator"
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
    trainer.save_model("./qwen_model_1.5B")
    trainer.save_tokenzer("./qwen_tokenizer_1.5B")
    
    trainer.push_to_hub("omaratef3221/Qwen2-1.5B-Instruct-SQL-generator")
    
if __name__ == "__main__":
    main()