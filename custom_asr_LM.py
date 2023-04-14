
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pyctcdecode import build_ctcdecoder
from transformers import Wav2Vec2ProcessorWithLM
import pandas as pd
from tqdm.contrib import tqdm
from datasets import load_metric
import torchaudio
import speechbrain as sb
import torch
hparams ={

        'wav2vec_hub': 'results/',
        'pretrained': 'results/',
        'test_csv':  'results/test.csv',
        'lm'      : 'results/2gram.arpa',
        'wer_file': 'results/wer_test.csv',
        'wer_lm_file': 'results/wer_test_lm.csv',
        'sample_rate': 16000,
        } 
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids

    return batch

def create_batch(path):
    data=[]
    df = pd.read_csv(path)
    for index, row in tqdm(df.iterrows()):
        entry = {}
        entry["sentence"] = row['wrd']
        info = torchaudio.info(row['wav'])
        sr = info.sample_rate
        w = sb.dataio.dataio.read_audio(row['wav'])
        if sr!=hparams['sample_rate']:
            sig = torchaudio.transforms.Resample(sr, hparams['sample_rate'])(w)
        else:
            sig = w
        sr = hparams['sample_rate']
        entry["audio"] = {'sampling_rate': sr, 'array': sig}
        entry['path'] = row['wav']
        data.append(entry)
    return data

def store_result(results, fichier):
    pred_str = []
    wer_step = []
    label_str = []

    for res in results:
        pred_str.append(res['pred_str'])
        label_str.append(res['target'])
        wer_step.append(wer_metric.compute (predictions=[res['pred_str']],
            references=[res['target']]))

    data = {'predictions': pred_str, 'target': label_str, 'wer':wer_step}
    df = pd.DataFrame(data=data)
    df.to_csv(fichier)
    print('Test wer : {:.3f}'.format(wer_metric.compute(predictions=pred_str, references=label_str)))
    print('Test wer : {:.3f}'.format(wer_metric.compute(predictions=pred_str, references=label_str)))
    return True

processor = Wav2Vec2Processor.from_pretrained(hparams["wav2vec_hub"])

def map_to_result(batch):
    with torch.no_grad():
        input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
        logits = model(input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    batch['pred_str'] = processor.batch_decode(pred_ids)[0]
    batch["target"] = processor.decode(batch["labels"], group_tokens=False)
    
    return batch

def map_to_result_lm(batch):
   with torch.no_grad():
        input_values = torch.tensor(batch['input_values'], device='cuda').unsqueeze(0)
        model.cuda()
        logits = model(input_values).logits.cpu().numpy()

   batch['pred_str'] = processor_with_lm.batch_decode(logits)[0][0]
   batch['target'] = batch['sentence']

   return batch

wer_metric = load_metric("wer")
cer_metric = load_metric("cer")

model = Wav2Vec2ForCTC.from_pretrained(hparams["wav2vec_hub"])

vocab_dict = processor.tokenizer.get_vocab()
sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}


decoder = build_ctcdecoder(
    labels=list(sorted_vocab_dict.keys()),
    kenlm_model_path=hparams['lm'],
)



processor_with_lm = Wav2Vec2ProcessorWithLM(
    feature_extractor=processor.feature_extractor,
    tokenizer=processor.tokenizer,
    decoder=decoder
)

test_data = create_batch(hparams["test_csv"])
test_data = list(map(prepare_dataset, test_data))

result = map(map_to_result, test_data)
result_lm = map(map_to_result_lm, test_data)

store_result(result_lm, hparams['wer_lm_file']) 
store_result(result, hparams['wer_file'])
