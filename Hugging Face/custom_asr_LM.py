
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

hparams ={

        'wav2vec_hub': 'results/checkpoint-500'
        }        
processor = Wav2Vec2Processor.from_pretrained(hparams["wav2vec_hub"])
model = Wav2Vec2ForCTC.from_pretrained(hparams["wav2vec_hub"])

