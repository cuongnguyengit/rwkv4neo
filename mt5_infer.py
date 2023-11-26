from transformers import AutoTokenizer, AutoModel
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("FredDYyy/mT5-base-translation-vi-en-jp-cn", token="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")
model = T5ForConditionalGeneration.from_pretrained("FredDYyy/mT5-base-translation-vi-en-jp-cn", token="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")

input_ids = tokenizer("translate English to Vietnamese: The house is wonderful.", return_tensors="pt").input_ids
outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


# tokenizer = AutoTokenizer.from_pretrained("FredDYyy/mT5-base-translation-vi-en-jp-cn")

# model = AutoModel.from_pretrained("FredDYyy/mT5-base-translation-vi-en-jp-cn")


