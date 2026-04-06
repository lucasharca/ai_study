# Stemming is making the text more uniform by reducing words to their "base form"
# so that different forms of the same word don't confuse the analysis
# connecting / connected => connect


from nltk.stem import PorterStemmer

ps = PorterStemmer()

connect_tokens = ['connecting', 'connected', 'connectivity', 'connect', 'connects']

for t in connect_tokens:
    print(t, ": ", ps.stem(t))
    # The result here is -> 'connect', 'connect', 'connect', 'connect', 'connect'

learn_tokens = ['learned', 'learning', 'learn', 'learns', 'learner', 'learners']

for t in learn_tokens:
    print(t, ": ", ps.stem(t))


likes_tokens = ['likes', 'better', 'worse']

for t in likes_tokens:
    print(t, ": ", ps.stem(t))
    # Here we can see that worse was stemmed to wors -> Not what we wanted