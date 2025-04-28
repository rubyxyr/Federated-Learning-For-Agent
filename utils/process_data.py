from datasets import load_dataset


def apply_chat_template(
    example,
    tokenizer,
):
    messages = [
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]},
    ]
    example["text"] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return example


def split_dataset(dataset, num_clients, seed=1234):
    dataset = dataset.shuffle(seed=seed)  # Shuffle the dataset
    local_datasets = []
    for i in range(num_clients):
        local_datasets.append(dataset.shard(num_clients, i))
    return local_datasets


def build_dataset(
    tokenizer, datasetname, num_clients, test_size=0.1, seed=1234, dataset_sample=200
):
    trainset_full = load_dataset(datasetname, split="train")
    train_test = trainset_full.train_test_split(test_size=test_size, seed=seed)
    train_dataset = train_test["train"]
    train_dataset = train_dataset.shuffle(seed=seed)
    if dataset_sample:
        num_sample = min(len(train_dataset), dataset_sample)
        train_dataset = train_dataset.select(range(num_sample))
    test_dataset = train_test["test"]
    column_names = list(train_dataset.features)
    processed_train_dataset = train_dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=column_names,
        desc="Applying chat template to train_sft",
    )

    processed_test_dataset = test_dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=column_names,
        desc="Applying chat template to test_sft",
    )
    train_dataset_split = split_dataset(processed_train_dataset, num_clients, seed)
    return train_dataset_split, processed_test_dataset
