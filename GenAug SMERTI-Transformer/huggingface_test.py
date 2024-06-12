import numpy as np
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import BatchEncoding

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
# TXT = "sat outside and enjoyed a great <mask> . morgan was very nice . manager helped <mask> and greeted us throughout the visit ."
# TXT = "good , <mask> great , food . unfortunately , <mask> dirty <mask> here <mask> , <mask> really off putting ."
# TXT = "UN Chief Says There Is No <mask> in Syria"
TXT = 'i <mask> this <mask>! very nice people run the <mask> and the <mask> is always good. stars!'

model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
print("huggingface_test_input_ids")
print(input_ids)

# generated_ids = model.generate(input_ids, max_length=40, repetition_penalty=0.5, num_beams=4)
generated_ids = model.generate(input_ids, max_length=50, repetition_penalty=0.5, do_sample=True, top_k=50, top_p=0.8)
final_generation = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print("generated_ids")
print(generated_ids)
print("final_generation")
print(final_generation)


