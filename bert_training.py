from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""



# Load pre-trained model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Load the dataset
# Assume we have a CSV file with 'text' and 'label' columns for binary classification task
dataset = load_dataset('csv', data_files={'train': 'train.csv', 'test': 'test.csv'})

# Preprocess the data
def preprocess(data):
    return tokenizer(data['text'], truncation=True, padding=True, max_length=512)

train_dataset = dataset['train'].map(preprocess, batched=True)
test_dataset = dataset['test'].map(preprocess, batched=True)

# Set the format for PyTorch
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    weight_decay=0.01,
    no_cuda=True
)

# Create the trainer and start training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./fine_tuned_model")
