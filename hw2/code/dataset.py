import random
import json

def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

train_set = load_json('./DATA/SST2/train.json')
val_set = load_json('./DATA/SST2/val.json')

print("Train set number: ", len(train_set))
print("Validation set number: ", len(val_set))

print("-----------Show some examples in train set-----------")
print("Here, 0 means Negative, 1 means Positive")
for i in range(10):
    print(random.choice(train_set))


