import json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn import metrics
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, EvalPrediction, DataCollatorWithPadding)
from torch.utils.data import Dataset

# Load and preprocess data functions
def load_data(filename):
    with open(filename, 'r') as f:
        datas = [json.loads(json.loads(line.strip())["gpt_answer"]) for line in f]
    datas = [item for sublist in datas for item in sublist]  # Flatten list of lists
    datas = [data for data in datas if data.get('timeliness') in {'year', 'month', 'quarter', 'day', 'none', 'uncertain', 'week'}]
    datas = list({data['query']: data for data in datas}.values())
    return datas

def preprocess_data(data, tokenizer, label_dict):
    inputs = {'query': [item['query'] for item in data],
              'labels': [label_dict[item['timeliness']] for item in data]}
    return inputs

# Custom Dataset
class MyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx] + '<s>现在是2023年9月6日',
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {key: val.squeeze(0) for key, val in encoding.items()}, torch.tensor(self.labels[idx], dtype=torch.long)
    def __len__(self):
        return len(self.labels)

# Compute metrics function
def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    probs = torch.nn.functional.softmax(torch.tensor(p.predictions), dim=1).numpy()
    return {
        'accuracy': (preds == p.label_ids).mean(),
        'recall': metrics.recall_score(p.label_ids, preds, average='macro'),
        'precision': metrics.precision_score(p.label_ids, preds, average='macro'),
        'f1': metrics.f1_score(p.label_ids, preds, average='macro'),
        'auc': metrics.roc_auc_score(p.label_ids, probs, average='macro', multi_class='ovo'),
    }

# Process starts
label_dict = {'year': 4, 'month': 2, 'quarter': 3, 'day': 0, 'week': 1, 'none': 5, 'uncertain': 6}

# Initialize tokenizer and model
model_name = ''
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7) # 文本分类模型
# Load and preprocess data
data = load_data('sft_before0830_v2.txt')
inputs = preprocess_data(data, tokenizer, label_dict)
# Generate training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    inputs['query'], inputs['labels'], random_state=42, test_size=0.1
)
# Create datasets
data_collator = DataCollatorWithPadding(tokenizer) # batch化
train_dataset = MyDataset(train_texts, train_labels, tokenizer)
val_dataset = MyDataset(val_texts, val_labels, tokenizer)
# Define Training Arguments and Trainer
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=64,
    warmup_steps=100,
    weight_decay=0.01,
    evaluation_strategy='steps',
    eval_steps=200,
    load_best_model_at_end=True
)

# 用于训练和评估神经网络的高级循环类
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train and Evaluate
trainer.train()
eval_results = trainer.evaluate()
model.save_pretrained('./saved_model_0908_newdata')
tokenizer.save_pretrained('./saved_model_0908_newdata')

# Prediction and save results
predictions = trainer.predict(val_dataset)
preds = np.argmax(predictions.predictions, axis=1)
probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
df = pd.DataFrame({
    'text': val_texts,
    'prediction': preds,
    'score': np.max(probs, axis=1),
    'label': predictions.label_ids,
})

df.to_csv('evaluation_results_0908.csv', index=False)