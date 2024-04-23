import sentencepiece as spm
import json
hparams = {
        'train': 'data/train.txt',
       'vocab_file': 'data/vocab_all.json',
        'tok_model':'m_char.model',
        }
spm.SentencePieceTrainer.train('--input='+hparams['train'] + '  --model_prefix=m_char --vocab_size=100 --model_type=char --normalization_rule_name=nfkc_cf')

sp = spm.SentencePieceProcessor(model_file=hparams['tok_model'])
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
with open(hparams['vocab_file'], 'w', encoding="UTF-8") as vocab_file:
    json.dump(vocab_dict, vocab_file, ensure_ascii=False)

print('vocabulaire created')
