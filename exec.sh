models=("facebook/wav2vec2-xls-r-300m" "facebook/wav2vec2-xls-r-1b" "facebook/wav2vec2-xls-r-2b")
tokens=("all" "char" "BPE")

for model in "${models[@]}"; do
  for tok in "${tokens[@]}"; do
    python3 custom_tokenizer.py --dataset train --tokenizer $tok
    python3 custom_asr.py --model $model --token $tok 
  done
done


