from datasets import load_dataset
from transformers import AutoTokenizer,AutoModelForSequenceClassification,TrainingArguments, Trainer
import random
import pandas as pd
import datasets
from IPython.display import display, HTML
import numpy as np
import evaluate



dataset = load_dataset("yelp_review_full")
print(dataset)




def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))

show_random_elements(dataset["train"])



tokenizer = AutoTokenizer.from_pretrained(r"./bert-base-chinese/", local_files_only=True)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)
show_random_elements(tokenized_datasets["train"], num_examples=1)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))




model = AutoModelForSequenceClassification.from_pretrained(r"./bert-base-chinese/", local_files_only=True, num_labels=5)



model_dir = "models/bert-base-cased-finetune-yelp"

# logging_steps 默认值为500，根据我们的训练数据和步长，将其设置为100
training_args = TrainingArguments(output_dir=model_dir,
                                  per_device_train_batch_size=16,
                                  num_train_epochs=5,
                                  logging_steps=100)
print(training_args)



metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)




training_args = TrainingArguments(output_dir=model_dir,
                                  evaluation_strategy="epoch",
                                  per_device_train_batch_size=16,
                                  num_train_epochs=3,
                                  logging_steps=30)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
small_test_dataset = tokenized_datasets["test"].shuffle(seed=64).select(range(100))

trainer.evaluate(small_test_dataset)


trainer.save_model(model_dir)
trainer.save_state()

# trainer.model.save_pretrained("./")

