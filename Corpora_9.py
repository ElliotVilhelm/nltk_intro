from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize

# mac corpus at Users/null/nltk_data

sample = gutenberg.raw("bible-kjv.txt")
tok = sent_tokenize(sample)

print(tok[5:15])
