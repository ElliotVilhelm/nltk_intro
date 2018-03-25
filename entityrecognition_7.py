"""
Entity recognition with Noun search combined can be useful
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
NE Type	Examples
ORGANIZATION	Georgia-Pacific Corp., WHO
PERSON	Eddy Bonte, President Obama
LOCATION	Murray River, Mount Everest
DATE	June, 2008-06-29
TIME	two fifty a m, 1:30 p.m.
MONEY	175 million Canadian Dollars, GBP 10.40
PERCENT	twenty pct, 18.75 %
FACILITY	Washington Monument, Stonehenge
GPE	South East Asia, Midlothian
"""

def process_content():
    try:
        for i in tokenized[10:]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            # Binary = true .. put different entity types together
            namedEnt = nltk.ne_chunk(tagged, binary=True)

            namedEnt.draw()



    except Exception as e:
        print(str(e))

process_content()
