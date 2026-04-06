import nltk
nltk.download("punkt_tab")

from nltk.tokenize import word_tokenize, sent_tokenize


sentences = "Her cat's name is Luna. Her dog's name it Max"

result = sent_tokenize(sentences)

print(result)

sentence_1 = "her cat's name is luna"

result_1 = word_tokenize(sentence_1)

print(result_1)

sentence_2 = "Her cat's name is Luna and her dog's name is Max"

result_2 = word_tokenize(sentence_2)

print(result_2)