
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pyctcdecode import build_ctcdecoder
from transformers import Wav2Vec2ProcessorWithLM
from utils import *



hparams ={

        'wav2vec_hub': 'YomieNZ/asr_cm_Ewondo'
        'test_csv':  'results/test.csv',
        'lm'      : 'results/3grams.csv',
        }        
processor = Wav2Vec2Processor.from_pretrained(hparams["wav2vec_hub"])
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
results = map(map_to_result, test_data)
pred_str = []
wer_step = []
label_str = []

for res in results:
   pred_str.append(res['pred_str'])
   label_str.append(res['target'])
   wer_step.append(wer_metric.compute(predictions=[res['pred_str']], references=[res['target']]))
           
data = {'predicitons': pred_str, 'taget': label_str, 'wer':wer_step}
df = pd.DataFrame(data=data)
df.to_csv(hparams['wer_file'])
print('Test wer : {:.3f}'.format(wer_metric.compute(predictions=pred_str, references=label_str)))

