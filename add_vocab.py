import os 

ext = ".arpa"
ngram_source = "2gram"+ext
ngram_target = ngram_source + "_correct" + ext
add_word = ["</s>"]
with open("5gram.arpa", "r") as read_file, open("5gram_correct.arpa", "w") as write_file:
  has_added_eos = False
  for line in read_file:
    if not has_added_eos and "ngram 1=" in line:
      count=line.strip().split("=")[-1]
      write_file.write(line.replace(f"{count}", f"{int(count)+1}"))
    elif not has_added_eos and "<s>" in line:
      write_file.write(line)
      write_file.write(line.replace("<s>", "</s>"))
      has_added_eos = True
    else:
      write_file.write(line)

