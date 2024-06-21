from datasets import load_dataset



def get_dataset(dataset_id = "motherduckdb/duckdb-text2sql-25k", split = "train[:10000]"):
    dataset = load_dataset(dataset_id, split=split)
    return dataset

def prepare_data(example):
    output_texts = []
    for i in range(len(example['query'])):
        text = f""" 
        Act as an assistant to a software engineer who is writing SQL query codes: 
        {example['prompt'][i]}\n ### The response query is: {example['query'][i]}
        """
        output_texts.append(text)
    return output_texts