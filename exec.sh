models=("facebook/wav2vec2-xls-r-300m" "facebook/wav2vec2-large-xlsr-53" "lucio/wav2vec2-large-xlsr-kinyarwanda" "LeBenchmark/wav2vec2-FR-7K-large", "Akashpb13/Swahili_xlsr")
tokens=("all" "char" "BPE")

for model in "${models[@]}"; do
  for tok in "${tokens[@]}"; do
    python3 custom_tokenizer.py --dataset train --tokenizer $tok
    python3 custom_asr.py --model $model --token $tok 
  done
done


