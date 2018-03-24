from nltk.tokenize import sent_tokenize, word_tokenize

# tokenizing
# word tokenizers
# sentence tokenizers
# lexicon and corporas
# corpora - body of text. ex: medical journals, presidential speeches
# lexicon - words and their meanings, "dictionary", investor speak

# investor speak - bull someone who is positive about the market
# english speak - bull is a real bull with horns

example_text = "Hello Mr. Smith, how are you doing today? The weather is great " \
               "and Python is awesome. The sky is great and the weather is " \
               "beautiful. Hello Mr. Smith."

print(sent_tokenize(example_text))
print(word_tokenize(example_text))


for word in word_tokenize(example_text):
    print(word)

for sentence in sent_tokenize(example_text):
    print(sentence)
