
# Qwen2-0.5B-Instruct-SQL-query-generator

![GitHub stars](https://img.shields.io/github/stars/omaratef3221/SQL_Query_Generator_llm?style=social)
![GitHub forks](https://img.shields.io/github/forks/omaratef3221/SQL_Query_Generator_llm?style=social)
![GitHub issues](https://img.shields.io/github/issues/omaratef3221/SQL_Query_Generator_llm)

This repository contains the fine-tuned model Qwen2-0.5B-Instruct-SQL-query-generator, designed to generate SQL queries from natural language text prompts.

## Model Description

The Qwen2-0.5B-Instruct-SQL-query-generator is a specialized model fine-tuned on the `motherduckdb/duckdb-text2sql-25k` dataset (first 10k rows). This model can convert natural language questions into SQL queries, facilitating data retrieval and database querying through natural language interfaces.

### Intended Uses

- Convert natural language questions to SQL queries.
- Facilitate data retrieval from databases using natural language.
- Assist in building natural language interfaces for databases.

### Limitations

- The model is fine-tuned on a specific subset of data and may not generalize well to all SQL query formats or databases.
- It is recommended to review the generated SQL queries for accuracy and security, especially before executing them on live databases.

## Training and Evaluation Data

### Training Data

The model was fine-tuned on the `motherduckdb/duckdb-text2sql-25k` dataset, specifically using the first 10,000 rows. This dataset includes natural language questions and their corresponding SQL queries.

### Evaluation Data

The evaluation data used for fine-tuning was a subset of the same dataset, ensuring consistency in training and evaluation metrics.

## Training Procedure

The training code is available on [GitHub](https://github.com/omaratef3221/SQL_Query_Generator_llm/).

### Training Hyperparameters

- **Learning Rate:** 1e-4
- **Batch Size:** 8
- **Save Steps:** 1
- **Logging Steps:** 500
- **Number of Epochs:** 5

### Training Frameworks

- **Transformers:** 4.39.0
- **PyTorch:** 2.2.0
- **Datasets:** 2.20.0
- **Tokenizers:** 0.15.2

## Model Performance

Evaluation metrics such as accuracy, precision, recall, and F1-score were used to assess the model's performance.

## Usage

To use this model, load it from the Hugging Face Model Hub and provide natural language text prompts. The model will generate the corresponding SQL queries.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("omaratef3221/Qwen2-0.5B-Instruct-SQL-query-generator")
model = AutoModelForSeq2SeqLM.from_pretrained("omaratef3221/Qwen2-0.5B-Instruct-SQL-query-generator")

inputs = tokenizer("Show me all employees with a salary greater than $100,000", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## How to Train the Model

To train the model, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/omaratef3221/SQL_Query_Generator_llm.git
    cd SQL_Query_Generator_llm
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the training script:
    ```bash
    python train.py --model_id Qwen/Qwen2-0.5B-Instruct --dataset_id motherduckdb/duckdb-text2sql-25k --epochs 5 --batch_size 8
    ```

## Notifications

After training, a notification will be sent to `https://ntfy.sh/sql_query_generator_llm`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For any questions or issues, please open an issue on this GitHub repository or contact [Omar Atef](https://github.com/omaratef3221).

---

⭐️ Don't forget to star the repository if you find it useful!
