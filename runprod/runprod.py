from transformers import BertTokenizerFast, BertForSequenceClassification

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
# from flask_sqlalchemy import SQLAlchemy

from pydantic import BaseModel

pretrainedmodel = 'remotejob/gradientclassification_v0'
max_length = 512

import sqlite3
conn = sqlite3.connect('data/mldata.db')

tokenizer = BertTokenizerFast.from_pretrained(pretrainedmodel)
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token

model = BertForSequenceClassification.from_pretrained(pretrainedmodel)

origins = ["*"]


class Itemask(BaseModel):
    ask: str


def get_reply(model,tokenizer, ask):
    inputs = tokenizer(ask, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    # perform inference to our model
    outputs = model(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)

    # executing argmax function to get the candidate label
    print("probs-->",probs.argmax().item(),probs.max().item())

    score = probs.max().item()

    intent = probs.argmax().item()


    return score,intent
    
    # return target_names['target'][probs.argmax().item()]

app = FastAPI()

# app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///example.sqlite"
# db = SQLAlchemy(app)


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/classification")
def predict(item: Itemask):
    ask = item.ask
    score,intent =  get_reply(model,tokenizer, ask)

    stsql = "SELECT template from templatestbl where intentid=" +str(intent)+" ORDER BY random() limit 1"
    print(stsql)

    with sqlite3.connect("data/mldata.db") as con:
        cur = con.cursor()
        cur.execute("SELECT template from templatestbl where intentid=" +str(intent)+" ORDER BY random() limit 1")
        for row in cur:
            print("ph",row[0])
            ans = row[0]

    
    return  {'Score':score,'Answer':ans} 

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info", reload=False)   