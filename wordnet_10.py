"""
Intro to nltk wordnet
"""
from nltk.corpus import wordnet

syns = wordnet.synsets("program")

# print(syns)
# print(syns[0])
# Just the word
# print(syns[0].lemmas()[0].name())

# definition
# print(syns[0].definition())

# examples
# print(syns[0].examples())

synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        # print(l)
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

# print(set(synonyms))
# print(set(antonyms))


# semantic similarity

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("boat.n.01")
# woo and palmer semantic similarity
print(w1.wup_similarity(w2))

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("car.n.01")
# woo and palmer semantic similarity
print(w1.wup_similarity(w2))

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("cat.n.01")
# woo and palmer semantic similarity
print(w1.wup_similarity(w2))
