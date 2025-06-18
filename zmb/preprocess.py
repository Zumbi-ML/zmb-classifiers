from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

def preprocess_data(df, target_column, config):
    text_column = config["data"]["text_column"]
    pretrained_model = config["model"]["pretrained_model"]
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    # Convert label to int if needed
    if df[target_column].dtype != int:
        df[target_column] = df[target_column].astype('category').cat.codes

    train_df, test_df = train_test_split(df, test_size=config["training"]["test_size"], random_state=config["training"]["random_state"])

    def tokenize(example):
        return tokenizer(
            example[text_column],
            padding="max_length",
            truncation=True,
            max_length=512
        )

    train_dataset = Dataset.from_pandas(train_df).map(tokenize, batched=True)
    test_dataset = Dataset.from_pandas(test_df).map(tokenize, batched=True)

    return train_dataset, test_dataset, tokenizer