"""
Lemmatizing - will create a synonym to the word
"""

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("cats"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("water"))
print(lemmatizer.lemmatize("ducks"))
print(lemmatizer.lemmatize("rocks"))

# default parameter for lematize pos=Noun!!
# we need to pass pos tag for non nouns
print(lemmatizer.lemmatize("better"))
print(lemmatizer.lemmatize("better", pos="a"))

print(lemmatizer.lemmatize("run"))
print(lemmatizer.lemmatize("run", pos='v'))

