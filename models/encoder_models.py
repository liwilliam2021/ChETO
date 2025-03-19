import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig
from peft import get_peft_model, LoraConfig, TaskType

class SentenceBERTContrastive(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = BertModel.from_pretrained('bert-base-uncased')

        # Freeze all base model params
        for param in base_model.parameters():
            param.requires_grad = False

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["query", "value"]
        )
        self.bert = get_peft_model(base_model, lora_config)

        self.projection = nn.Linear(self.bert.config.hidden_size, 128)  # still project to embedding space
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        return self.projection(cls_embeddings)
    
class BertSentimentClassifier(nn.Module):
    def __init__(self):
        super(BertSentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.out = nn.Linear(self.bert.config.hidden_size, 5)  # 5 sentiment classes

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output
        return self.out(cls_output)