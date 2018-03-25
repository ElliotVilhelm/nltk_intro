"""
Chinking: Chunk EXCEPT .. 
"""
import nltk
from nltk.corpus import state_union # state of the union addresses form last 70 years
# unsupervised ML tokenizer, comes pre trained, can be trained
from nltk.tokenize import PunktSentenceTokenizer

# loading in corpora
train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

# Train tokenizing network on sample speech
custom_sent_tokenizer = PunktSentenceTokenizer(train_text) # train on text

tokenized = custom_sent_tokenizer.tokenize(sample_text)


"""
POS tag list:

CC	coordinating conjunction
CD	cardinal digit
DT	determiner
EX	existential there (like: "there is" ... think of it like "there exists")
FW	foreign word
IN	preposition/subordinating conjunction
JJ	adjective	'big'
JJR	adjective, comparative	'bigger'
JJS	adjective, superlative	'biggest'
LS	list marker	1)
MD	modal	could, will
NN	noun, singular 'desk'
NNS	noun plural	'desks'
NNP	proper noun, singular	'Harrison'
NNPS	proper noun, plural	'Americans'
PDT	predeterminer	'all the kids'
POS	possessive ending	parent's
PRP	personal pronoun	I, he, she
PRP$	possessive pronoun	my, his, hers
RB	adverb	very, silently,
RBR	adverb, comparative	better
RBS	adverb, superlative	best
RP	particle	give up
TO	to	go 'to' the store.
UH	interjection	errrrrrrrm
VB	verb, base form	take
VBD	verb, past tense	took
VBG	verb, gerund/present participle	taking
VBN	verb, past participle	taken
VBP	verb, sing. present, non-3d	take
VBZ	verb, 3rd person sing. present	takes
WDT	wh-determiner	which
WP	wh-pronoun	who, what
WP$	possessive wh-pronoun	whose
WRB	wh-abverb	where, when
"""


def process_content():
    try:
        for i in tokenized[5:]:
            # print(i, "\n\n\n")
            words = nltk.word_tokenize(i)
            # print(words, "\n\n\n")
            tagged = nltk.pos_tag(words)

            # . is any character
            # ? is 0 or 1
            # all rb part of speech tags at no longer than 3 characters
            # * any adverb 0 or more
            # all adverbs, all verb, a noun, 0 or 1 proper noun

            # the chink HAS to be on a seperate line in comparison to the chunk

            chunkGram = r"""Chunk: {<.*>+}
                                    }<VB.?|IN|DT|TO>+{"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            # print(chunked)
            chunked.draw()
            # Tuples of words with part of speeches

    except Exception as e:
        print(str(e))

process_content()
