# Introduction

This code contains step-by-step guide to setup working environment in Hugging Face.

# Step 1: Setup the environment
We need to visit Meta github to have access to Llama model. Also, use the same email to register a Huggingface account.

##### Obtaining Huggingface access token
Click on top-right icon > Settings > Access Tokens > New Token to create new token. Copy the token values, which will be used in future code.

##### Install required library
Use pip to install transformers
```
pip install transformers datasets torch
```


# Step 2: Run a simple example
This is a bare minium code to run a small LLM 
```
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Load the IMDB dataset
dataset = load_dataset('imdb')

# Split the dataset into train and test sets
train_dataset = dataset['train']
test_dataset = dataset['test']

# Load the tokenizer and model
model_name = "facebook/opt-350m"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=[YourToken])
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, token=[YourToken])

# Tokenize the dataset
def preprocess_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)

# Set up the Trainer
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)

```
