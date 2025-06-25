from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

def preprocess_data(df, target_column, config):
    from transformers import AutoTokenizer
    from datasets import Dataset
    from sklearn.model_selection import train_test_split

    text_column = config["data"]["text_column"]
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["pretrained_model"])

    def tokenize(batch):
        texts = [str(t) for t in batch[text_column]]

        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=512,
        )

    # Split train/test
    train_df, test_df = train_test_split(
        df,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"]
    )

    train_dataset = Dataset.from_pandas(train_df).map(tokenize, batched=True)
    test_dataset = Dataset.from_pandas(test_df).map(tokenize, batched=True)

    return train_dataset, test_dataset, tokenizer
