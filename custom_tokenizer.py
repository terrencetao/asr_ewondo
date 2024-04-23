from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import normalizers
from tokenizers.normalizers import NFD, Lowercase,Strip, Replace
import json
import codecs
import argparse
import sentencepiece as spm
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help ='source of text')
parser.add_argument('--tokenizer', help ='BPE, char, all')
args = parser.parse_args()
text  = args.dataset
token = args.tokenizer
f = "data/" + text + ".txt"
if token=='BPE':
  tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
  trainer = BpeTrainer(special_tokens=["[UNK]","[PAD]"," "])
  tokenizer.pre_tokenizer = Whitespace()
  chars_to_remove_regex = '[\(\)\_\,\?\.\!\-\;\:\"\“\%\‘\”\�\']'
  normalizer = normalizers.Sequence([NFD(), Lowercase(), Strip(), Replace(chars_to_remove_regex, '')])
  tokenizer.normalizer = normalizer
  
  files = [f]
  tokenizer.train(files, trainer)
  tokenizer.save("data/tokenizer-ewondo.json")

  with codecs.open("data/tokenizer-ewondo.json", 'r', encoding="UTF-8") as tokenizer_file:
    data = json.load(tokenizer_file)
    vocab_dict = data['model']['vocab']
  with codecs.open('data/vocab_BPE.json', 'w', encoding="UTF-8") as vocab_file:
    json.dump(vocab_dict, vocab_file, ensure_ascii=False)
elif token == 'all':
   spm.SentencePieceTrainer.train('--input='+f + '  --model_prefix=m_char --vocab_size=100 --model_type=char --normalization_rule_name=nfkc_cf')

   sp = spm.SentencePieceProcessor(model_file='m_char.model')
   vocabs = [sp.id_to_piece(id) for id in range(sp.get_piece_size())]

   rm = ["!",".", "?",",","\n", '-',":",";","'", "!","«","<unk>", "<s>", "</s>","̀", "́", "̂", "̄", "̌"]
   for e in rm:
     if e in vocabs:
       vocabs.remove(e)
   vocab_dict = {v.lower():k for k,v in enumerate(sorted(vocabs))}
   vocab_dict["|"] = vocab_dict["▁"]
   del vocab_dict["▁"]
   ajout = ["á", "é","ə́", "ɛ́", "í", "ó", "ɔ́", "ú","ē", "ə̄","ɛ̄", "ī" ,"ō", "ɔ̄", "ū","ǎ", "ě", "ə̌", "ɛ̌", "ǐ", "ǒ", "ɔ̌", "ǔ", "â", "ê", "ə̂", "ɛ̂", "î" ,"ô" ,"ɔ̂", "û","dz", "gb", "kp","ndz", "a", "e", "ə", "ɛ", "i", "o", "ɔ", "u"]
   for a in ajout:
     if a not in vocab_dict.keys():
       vocab_dict[a]=len(vocab_dict)
   vocab_dict['[UNK]']=len(vocab_dict)
   vocab_dict['[PAD]']=len(vocab_dict)
   with open('data/vocab_all.json', 'w', encoding="UTF-8") as vocab_file:
     json.dump(vocab_dict, vocab_file, ensure_ascii=False)
else:
   spm.SentencePieceTrainer.train('--input='+f + '  --model_prefix=m_char --vocab_size=100 --model_type=char --normalization_rule_name=nfkc_cf')

   sp = spm.SentencePieceProcessor(model_file='m_char.model')
   vocabs = [sp.id_to_piece(id) for id in range(sp.get_piece_size())]
   rm = ["!",".", "?",",","\n", '-',":",";","'", "!","«","<unk>", "<s>", "</s>"]
   for e in rm:
      if e in vocabs:
       vocabs.remove(e)
   vocab_dict = {v.lower():k for k,v in enumerate(sorted(vocabs))}
   vocab_dict["|"] = vocab_dict["▁"]
   del vocab_dict["▁"]
   with open('data/vocab_all.json', 'w', encoding="UTF-8") as vocab_file:
    json.dump(vocab_dict, vocab_file, ensure_ascii=False)
