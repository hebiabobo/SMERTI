from transformers import BartTokenizer, BartForConditionalGeneration
import os
from transformers import TrainingArguments
from transformers import Trainer

# 提供保存检查点
training_args = TrainingArguments("test_trainer")

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

# 数据集
masked_train_path = r"E:\PycharmProjects\SMERTI-master\News_Dataset\masked_train_headlines.txt"
unmasked_train_path = r"E:\PycharmProjects\SMERTI-master\News_Dataset\train_headlines.txt"

masked_val_path = r"E:\PycharmProjects\SMERTI-master\News_Dataset\masked_val_headlines.txt"
unmasked_val_path = r"E:\PycharmProjects\SMERTI-master\News_Dataset\val_headlines.txt"

masked_test_path = r"E:\PycharmProjects\SMERTI-master\News_Dataset\masked_test_headlines.txt"
unmasked_test_path = r"E:\PycharmProjects\SMERTI-master\News_Dataset\test_headlines.txt"

trainer = Trainer(
    model=model, args=training_args, train_dataset=small_train_dataset, eval_dataset=small_eval_dataset
)
