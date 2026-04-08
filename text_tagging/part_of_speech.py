# POS tagging: labeling each word in a sentence with its grammatical role
# Noun, verb, adjective, etc.

import spacy
# spaCy: NLP (Natural language processor) lib that comes with pretrained models capable of 
# recognizing parts of speech, named entities, and other linguistic features
import pandas as pd

nlp = spacy.load("en_core_web_sm") #en_core_web_sm: lightweigh and fast

emma_ja = "emma woodhouse handsome clever and rich with a comfortable home and happy disposition seemed to unite some of the best blessings of existence and had lived nearly twentyone years in the world with very little to distress or vex her she was the youngest of the two daughters of a most affectionate indulgent father and had in consequence of her sisters marriage been mistress of his house from a very early period her mother had died too long ago for her to have more than an indistinct remembrance of her caresses and her place had been supplied by an excellent woman as governess who had fallen little short of a mother in affection sixteen years had miss taylor been in mr woodhouses family less as a governess than a friend very fond of both daughters but particularly of emma between them it was more the intimacy of sisters even before miss taylor had ceased to hold the nominal office of governess the mildness of her temper had hardly allowed her to impose any restraint and the shadow of authority being now long passed away they had been living together as friend and friend very mutually attached and emma doing just what she liked highly esteeming miss taylors judgment but directed chiefly by her own"

# a spacy document: an object that stores the text 
# along with all the linguistic information spacy generates such as tokens and part of speech tags
# also analyzes grammar and looks for named entities

spacy_doc = nlp(emma_ja)
pos_df = pd.DataFrame(columns=['token', 'pos_tag'])

for token in spacy_doc:
    # for each token in the spacy doc
    # we concat in a pandas dataframe
    pos_df = pd.concat([pos_df, 
                        pd.DataFrame.from_records([{'token': token.text, 'pos_tag': token.pos_}])], ignore_index=True)

# say we want to find the most common tokens and their associated POS tag
# we group by both token and pos_tag
# .size -> to count how many rows belong to each group
# reset index -> important because after grouping, pandas uses the grouped columns as the index of the table
# resetting the index turns them back into regular columns, so the table looks clean and easy to read
# we also give this new column the name 'counts'
# finally we sort the values with ascending set to false, so the most significant shows first

pos_df_counts = pos_df.groupby(['token', 'pos_tag']).size().reset_index(name='counts').sort_values(by='counts', ascending=False)


print(pos_df_counts.head(10))

# now let's find out how many different words fall under each POS tag
# here each row will represent one part of speech tag, such as noun, verb, adjective
# Along with the number of unique tokens that belong to it
####
# We use the groupby function on pos_df_counts, this time grouping only by the post tag column
# Then for each group, we count the number of tokens using .count 
# -> this gives us the number of different words used for each part of speec (pos)
# Then we sort values using ascending=False

pos_df_poscounts = pos_df_counts.groupby(['pos_tag'])['token'].count().sort_values(ascending=False)

print(pos_df_poscounts.head(10))

# We can be more specific. To look particularly at the most common nouns in our data
# we can filter the pos_df_counts dataframe to include only pos tags that equal noun
# from there we select the top ten nouns by their counts

top_nouns = pos_df_counts[pos_df_counts.pos_tag == 'NOUN'][:10]

print(top_nouns)