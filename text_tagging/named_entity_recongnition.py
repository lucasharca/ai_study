# Named Entity Recognition (NER): Process of finding and labeling essential pieces of information in text
# such as names of people, places, organizations, dates, quantities

import spacy
from spacy import displacy # built in visualization tool that lets us see entities highlighted directly in text
# from spacy import tokenizer # breaks text into tokens so that Spacy can process
from IPython.display import HTML, display
import re 

nlp = spacy.load('en_core_web_sm')

google_text = "Google was founded on September 4, 1998, by computer scientists Larry Page and Sergey Brin while they were PhD students at Stanford University in California. Together they own about 14% of its publicly listed shares and control 56% of its stockholder voting power through super-voting stock. The company went public via an initial public offering (IPO) in 2004. In 2015, Google was reorganized as a wholly owned subsidiary of Alphabet Inc. Google is Alphabet's largest subsidiary and is a holding company for Alphabet's internet properties and interests. Sundar Pichai was appointed CEO of Google on October 24, 2015, replacing Larry Page, who became the CEO of Alphabet. On December 3, 2019, Pichai also became the CEO of Alphabet."

spacy_doc = nlp(google_text)

for word in spacy_doc.ents:
    # label -> numeric id of the entity
    # label_ -> a human-readable label
    print(word.text, ": ", word.label_)

html = displacy.render(spacy_doc, style="ent", jupyter=False)

google_text_clean = re.sub(r'[^\w\s]', '', google_text).lower()

print(google_text_clean)

spacy_doc_clean = nlp(google_text_clean)

# When we remove capitalization, the model lost some vital clues 
# that helps recognize names of people, places or organizations
# By removing punctuation we dirupted the sentence boundaries that depend on it to be recognized correctly

for word in spacy_doc_clean.ents:
    print(word.text,  ": ", word.label_)

# displacy.serve(spacy_doc, style="ent", port=8008)