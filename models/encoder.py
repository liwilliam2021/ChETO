from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
import torch

# Load the dataset
dataset = load_dataset('csv', data_files={'train': 'path/to/train.csv', 'test': 'path/to/test.csv'})

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Freeze all layers except the classification head
for param in model.bert.parameters():
  param.requires_grad = False
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenize the dataset
def tokenize_function(examples):
  return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# Create DataLoader
train_dataloader = torch.utils.data.DataLoader(tokenized_datasets['train'], batch_size=16, shuffle=True)
eval_dataloader = torch.utils.data.DataLoader(tokenized_datasets['test'], batch_size=16)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(3):
  for batch in train_dataloader:
    optimizer.zero_grad()
    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['label'])
    loss = outputs.loss
    loss.backward()
    optimizer.step()
  print(f"Epoch {epoch + 1} completed")

# Evaluation loop
model.eval()
eval_loss = 0
for batch in eval_dataloader:
  with torch.no_grad():
    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['label'])
    loss = outputs.loss
    eval_loss += loss.item()
eval_loss /= len(eval_dataloader)
print(f"Evaluation loss: {eval_loss}")