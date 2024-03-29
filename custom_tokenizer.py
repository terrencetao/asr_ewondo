from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import normalizers
from tokenizers.normalizers import NFD, Lowercase,Strip, Replace
import json
import codecs

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(special_tokens=["[UNK]","[PAD]"," "])
tokenizer.pre_tokenizer = Whitespace()
chars_to_remove_regex = '[\(\)\_\,\?\.\!\-\;\:\"\“\%\‘\”\�\']'
normalizer = normalizers.Sequence([NFD(), Lowercase(), Strip(), Replace(chars_to_remove_regex, '')])
tokenizer.normalizer = normalizer
files = [f"data/testamentEwo.txt" ]
tokenizer.train(files, trainer)
tokenizer.save("data/tokenizer-ewondo.json")

with codecs.open("data/tokenizer-ewondo.json", 'r', encoding="UTF-8") as tokenizer_file:
	data = json.load(tokenizer_file)
	vocab_dict = data['model']['vocab']
with codecs.open('data/vocab.json', 'w', encoding="UTF-8") as vocab_file:
    json.dump(vocab_dict, vocab_file, ensure_ascii=False)
