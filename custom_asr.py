import torch
import json
import codecs
from datasets import load_metric
import soundfile as sf
import torchaudio
import argparse
from tokenizers import Tokenizer
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC
from data_prepare import prepare_data
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from tqdm.contrib import tqdm
from transformers import Trainer
from transformers import TrainingArguments
import numpy as np
import sentencepiece as spm
#from speechbrain.lobes.augment import TimeDomainSpecAugment
import random 
#do_augment= TimeDomainSpecAugment (speeds=[80, 110, 120],
#                                   perturb_prob=1.0,
#                                   drop_freq_prob=1.0,
#                                  drop_chunk_length_low=1000,
#                                   drop_chunk_length_high=3000)

parser = argparse.ArgumentParser()
parser.add_argument('--model', help ='name of model')
args = parser.parse_args()
model  = args.model

hparams = {
'output_folder' : "./results",
'repo_name': 'YomieNZ/asr_cm_Ewondo',
'seed': 1986,

# URL for  wav2vec2 model.
'wav2vec2_hub': model,

# Data files
'data_folder': 'datasets/Phrases',
'train_splits': ["train"],
'dev_splits': ["dev"],
'test_splits': ["test"],
'skip_prep': False,
'train_csv': 'results/train.csv',
'valid_csv': 'results/dev.csv',
'test_csv':  'results/test.csv',
'tokenizer_folder': "data/tokenizer-ewondo.json",
'vocab_file': "data/vocab_char.json",
'tok_model': 'data/m_char.model',
'wer_file':"results/wer_test.csv",
'lm':'results/2gram.arpa',
'wer_lm_file':"results/wer_lm_test.csv",
# training parameter
'sample_rate': 16000
}

random.seed(hparams['seed'])
run_on_main(
        prepare_data,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["output_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": "train.csv",
            "skip_prep": hparams["skip_prep"],
            
        },
    )

def prepare_dataset(batch):
        audio = batch["audio"]


        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["input_length"] = len(batch["input_values"])
        
        with processor.as_target_processor():
            batch["labels"] = processor(batch["sentence"]).input_ids
        return batch

def create_batch(path, aug=True):
     """
	    path: path to the csv generate by prepare_data
	          field of csv:
	                  ID: id of wave
	                  wrd: transcript of wav
	                  wav: path to the wav
	                  spk_id: id of speaker
	                  duration: length of wav
	    output: list of dictionay
     """
     data = []
     df = pd.read_csv(path)
     for index, row in tqdm(df.iterrows()):
         entry = {}
         entry["sentence"] = row['wrd']
         info = torchaudio.info(row['wav'])
         sr = info.sample_rate
         w  = sb.dataio.dataio.read_audio(row['wav'])
         length = torch.ones(1)
         if sr!=hparams['sample_rate']:
             sig = torchaudio.transforms.Resample(sr, 
                     hparams['sample_rate'])(w)
         else:
             sig = w
         
         sr = hparams['sample_rate']
         entry["audio"] = {"sampling_rate": sr, 'ID' : row["ID"], "array" : sig}
         if aug==True and random.randint(1,2)==1:
             aug_w = do_augment(sig.unsqueeze(0),length)
             entry_aug = {}
             entry_aug["audio"] = {"sampling_rate": sr, 'ID': row["ID"] + '_aug', "array":aug_w.squeeze()}
             entry_aug['sentence'] = row['wrd']
             data.append(entry_aug)
         entry["path"] = row["wav"]
         entry['ID']= row['ID']
         data.append(entry)
         
     return data


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer= cer_mertric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer, "cer": cer}


def map_to_result(batch):
    with torch.no_grad():
        input_values = torch.tensor(batch["input_values"], device="cpu").unsqueeze(0)
        logits = model(input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    batch['pred_str'] = processor.batch_decode(pred_ids)[0]
    batch["target"] = processor.decode(batch["labels"], group_tokens=False)
    
    return batch

def map_to_result_lm(batch):
    with torch.no_grad():
        input_values = torch.tensor(batch["input_values"], device="cpu").unsqueeze(0)
        logits = model(input_values).logits.cpu().numpy()
    
    batch['pred_str'] = processor_with_lm.batch_decode(logits)[0][0]
    batch["target"] = batch['sentence']

    
    return batch

def store_resutl(results, model, fichier):
    pred_str = []
    wer_step = []
    label_str = []
    id_aud = []
    for res in results:
       pred_str.append(res['pred_str'])
       label_str.append(res['target'])
       id_aud.append(res['ID'])
       wer_step.append(wer_metric.compute(predictions=[res['pred_str']], references=[res['target']]))
               
    data = {'ID':id_aud,'predicitons': pred_str, 'taget': label_str, 'wer':wer_step}
    df = pd.DataFrame(data=data)
    with open(fichier, 'a', newline='') as f:
        df.to_csv(f, header=f.tell()==0, index=False)
    
    print('{} Test wer : {:.3f}'.format(model, wer_metric.compute(predictions=pred_str, references=label_str)))
    print('\n {} Test cer : {:.3f}'.format(model, cer_mertric.compute(predictions=pred_str, references=label_str)))
    return True

    
train_data = create_batch(hparams["train_csv"], aug=False)
dev_data = create_batch(hparams["valid_csv"],aug=False)
test_data = create_batch(hparams["test_csv"],aug=False)

#vocab_train = set(y for x in train_data for y in x['sentence'])
#vocab_dev = set(y for x in dev_data for y in x['sentence'])
#vocab_test =  set(y for x in test_data for y in x['sentence'])
#vocab1 = vocab_train.union(vocab_dev)
#vocab = vocab1.union(vocab_test)
#sp = spm.SentencePieceProcessor(model_file=hparams['tok_model'])
#vocabs = [sp.id_to_piece(id) for id in range(sp.get_piece_size())]

#if "\n" in vocabs:
#    vocab.remove("\n")
#vocab_dict = {v: k for k, v in enumerate(sorted(vocabs))}
#vocab_dict["|"] = vocab_dict[" "]
#del vocab_dict[" "]
#vocab_dict["[UNK]"]= len(vocab_dict)
#vocab_dict["[PAD]"]= len(vocab_dict)
#with codecs.open(hparams['vocab_file'], 'w', encoding="UTF-8") as vocab_file:
#    json.dump(vocab_dict, vocab_file, ensure_ascii=False)

print('vocabulaire created')

tokenizer = Wav2Vec2CTCTokenizer(hparams['vocab_file'], unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)


processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

wer_metric = load_metric("wer")
cer_mertric= load_metric("cer")

train_data = list(map(prepare_dataset, train_data))
dev_data = list(map(prepare_dataset, dev_data))
test_data = list(map(prepare_dataset, test_data))


model = Wav2Vec2ForCTC.from_pretrained(
        hparams['wav2vec2_hub'], 
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.1,
        mask_time_prob=0.075,
        layerdrop=0.1,
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,
    )

model.freeze_feature_extractor()



training_args = TrainingArguments(
  output_dir=hparams['output_folder'],
  group_by_length=True,
  per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=60,
    gradient_checkpointing=True,
    fp16=False,
    save_steps=100,
    eval_steps=30,
    logging_steps=50,
    learning_rate=3e-4,
    warmup_steps=500,
    save_total_limit=1,
    push_to_hub=False,

)


trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=dev_data,
    tokenizer=processor.feature_extractor,
)

print('starting training')
trainer.train()
model.save_pretrained(hparams['output_folder'])
tokenizer.save_pretrained(hparams['output_folder'])
processor.save_pretrained(hparams['output_folder'])

results = map(map_to_result, test_data)




store_resutl(results, hparams['wer_file'])



vocab_dict = processor.tokenizer.get_vocab()
sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}

from pyctcdecode import build_ctcdecoder
from transformers import Wav2Vec2ProcessorWithLM

decoder = build_ctcdecoder(
    labels=list(sorted_vocab_dict.keys()),
    kenlm_model_path=hparams['lm'],
)



processor_with_lm = Wav2Vec2ProcessorWithLM(
    feature_extractor=processor.feature_extractor,
    tokenizer=processor.tokenizer,
    decoder=decoder
)

result_lm = map(map_to_result_lm, test_data)
store_resutl(result_lm, hparams['wer_lm_file'])
