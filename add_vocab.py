import os 

ext = ".arpa"
ngram_source = "3gram"+ext
ngram_target = ngram_source + "_correct" + ext
add_word = ["</s>"]
with open(os.path.join("results",ngram_source), "r", encoding="UTF-8") as read_file, open(os.path.join("results",ngram_target), "w", encoding="UTF-8") as write_file:
    for w in add_word:
       has_added_eos = False
       for line in read_file:
           if not has_added_eos and "ngram 1=" in line:
               count=line.stip().split("=")[-1]
               write_file.write(line.replace(f"{count}", f"{int(count)+1}"))
           #elif not has_added_eos and w
