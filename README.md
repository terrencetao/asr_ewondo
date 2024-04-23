asr_ewondo

 This is the model for Wav2Vec2-Large-XLSR-Ewondo, a fine-tuned facebook/wav2vec2-large-xlsr-53 model on the dataset collected by the IDASCO Team
### 0. Setup Environment :
 torch
 json
 codecs
 datasets 
 soundfile
 torchaudio
 argparse
 tokenizers 
 transformers
 speechbrain 
 pandas 
 dataclasses 
 typing 
 tqdm
 numpy
 sentencepiece

All the python packages can be installed via `pip install`

### 1. Description

#### 1.2 Dataset
We constructed a corpus from 103 sentences
read by 5 speakers. We randomly selected 11
sentences for the test (2min30s) and 92 remaining sentences for the
training (21min51s)

#### 1.3 running:
- Define the tokenization method:
 All: letter + tone ,  char: tone separate to letter and BPE
 
`python3 custom_tokenizer.py --dataset train --tokenizer all`

- Running asr

`python3 custom_asr.py --model $model --$tok`
