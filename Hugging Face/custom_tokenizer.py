from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import normalizers
from tokenizers.normalizers import NFD, Lowercase,Strip, Replace
import json


tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(special_tokens=["[UNK]","[PAD]"," "])
tokenizer.pre_tokenizer = Whitespace()
chars_to_remove_regex = '[0-9]|[\(\)\_\,\?\.\!\-\;\:\"\“\%\‘\”\�\']'
normalizer = normalizers.Sequence([NFD(), Lowercase(), Strip(), Replace(chars_to_remove_regex, '')])
tokenizer.normalizer = normalizer
files = [f"data/train.txt" ]
tokenizer.train(files, trainer)
tokenizer.save("data/tokenizer-ewondo.json")

with open("data/tokenizer-ewondo.json", 'r', encoding="UTF-8") as tokenizer_file:
	data = json.load(tokenizer_file)
	vocab_dict = data['model']['vocab']
with open('data/vocab.json', 'w', encoding="UTF-8") as vocab_file:
    json.dump(vocab_dict, vocab_file)
