

# from sklearn.datasets import fetch_20newsgroups
# from sklearn.model_selection import train_test_split


# dataset = fetch_20newsgroups(subset="test", shuffle=True, remove=("headers", "footers", "quotes"))


from pprint import pprint


# pprint(list(dataset.data[:2]))

# print(dataset)


# def read_20newsgroups(test_size=0.2):
#     # download & load 20newsgroups dataset from sklearn's repos
#     dataset = fetch_20newsgroups(subset="all", shuffle=True, remove=("headers", "footers", "quotes"))
#     documents = dataset.data
#     labels = dataset.target
#     # split into training & testing a return data as well as label names
#     return train_test_split(documents, labels, test_size=test_size), dataset.target_names

# (train_texts, valid_texts, train_labels, valid_labels), target_names = read_20newsgroups()

# pprint(list(valid_labels))


import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from datasets import load_dataset

dataset = load_dataset('csv', data_files={'train': ['data/train.csv'],'valid':['data/val.csv']})


# print(list(dataset['valid']['intent']))

model_name = "bert-base-uncased"
# max sequence length for each document/sentence sample
max_length = 512
# load the tokenizer
tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)

train_encodings = tokenizer(dataset['train']['ask'], truncation=True, padding=True, max_length=max_length)
valid_encodings = tokenizer(dataset['valid']['ask'], truncation=True, padding=True, max_length=max_length)

train_labels = dataset['train']['intent']
valid_labels = dataset['valid']['intent']


class NewsGroupsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = NewsGroupsDataset(train_encodings, train_labels)
valid_dataset = NewsGroupsDataset(valid_encodings, valid_labels)

print(train_dataset[0])