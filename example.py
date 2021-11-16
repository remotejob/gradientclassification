from transformers import BertTokenizerFast, BertForSequenceClassification

from datasets import load_dataset
model_path = 'remotejob/gradientclassification_v0'
max_length = 512

model = BertForSequenceClassification.from_pretrained(model_path, num_labels=68)
tokenizer = BertTokenizerFast.from_pretrained(model_path)


def get_prediction(text):
    # prepare our text into tokenized sequence
    inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    # perform inference to our model
    outputs = model(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)
    # executing argmax function to get the candidate label
    print("probs-->",probs.argmax().item(),probs.max())
    return probs.argmax().item(),probs.max()
    # return target_names['target'][probs.argmax().item()]


text = """
eik sit tehdä treffit sun sängyysi ja hoidetaan yhteiset himomme pois
"""
print(get_prediction(text))