from tqdm.contrib import tqdm
import pandas as pd
import torch

def prepare_dataset(batch):
        audio = batch["audio"]


        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["input_length"] = len(batch["input_values"])
        
        with processor.as_target_processor():
            batch["labels"] = processor(batch["sentence"]).input_ids
        return batch


def create_batch(path):
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
        info = torchaudio.info(row["wav"])
        sr = info.sample_rate
        w  = sb.dataio.dataio.read_audio(row['wav'])
        if sr!=hparams['sample_rate']:
            sig = torchaudio.transforms.Resample(
                sr, hparams['sample_rate'])(w)
        else:
            sig = w
        sr = hparams['sample_rate']
        entry["audio"] = {"sampling_rate": sr, 'path' : row["ID"], "array" : sig}
        entry["path"] = row["wav"]
        data.append(entry)

    return data


    def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def map_to_result(batch):
    with torch.no_grad():
        input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
        logits = model(input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    batch['pred_str'] = processor.batch_decode(pred_ids)[0]
    batch["target"] = processor.decode(batch["labels"], group_tokens=False)
    
    return batch