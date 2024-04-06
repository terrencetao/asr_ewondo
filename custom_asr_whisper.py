
import torch
import json
import codecs
from datasets import load_metric
import soundfile as sf
import torchaudio
import argparse
from tokenizers import Tokenizer

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


from dataclasses import dataclass
from transformers import WhisperProcessor
from transformers import WhisperTokenizer
from transformers import WhisperFeatureExtractor
from transformers import WhisperForConditionalGeneration
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
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
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
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

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
    
    
train_data = create_batch(hparams["train_csv"], aug=False)
dev_data = create_batch(hparams["valid_csv"],aug=False)
test_data = create_batch(hparams["test_csv"],aug=False)
  
    
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Hindi", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Hindi", task="transcribe")
    
wer_metric = load_metric("wer")
cer_mertric= load_metric("cer")

train_data = list(map(prepare_dataset, train_data))
dev_data = list(map(prepare_dataset, dev_data))
test_data = list(map(prepare_dataset, test_data))    
    
    


model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")


model.generation_config.language = "hindi"
model.generation_config.task = "transcribe"

model.generation_config.forced_decoder_ids = None


from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
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

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=dev_data,
    tokenizer=processor.feature_extractor,
)



trainer.train()



