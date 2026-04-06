import re 

# my_folder = "C:\desktop\notes"
my_folder_fixed = r"C:\desktop\notes"

# print(my_folder)
print(my_folder_fixed)

result_search = re.search("pattern", r"string to contain the pattern")

print(result_search)

result_search_two = re.search("pattern", r"the phrase to fint isn't in this string")
# The result is none
print(result_search_two)

# re.sub("old text", "new text", "phrase") searches for a specific text within a string and replaces it with a new content

string = r"sara was able to help me find the items I needed quickly"
new_strng = re.sub("sara", "sarah", string)


print(new_strng)


customer_reviews = ['sam was a great help to me in the store', 
 'the cashier was very rude to me, I think her name was eleanor', 
 'amazing work from sadeen!', 
 'sarah was able to help me find the items i needed quickly', 
 'lucy is such a great addition to the team', 
 'great service from sara she found me what i wanted']

sarahs_reviews = []

# By adding the '?' after the h in the pattern, it tells us that letter is optional
# So by this pattern it'll return either sara or sarah

pattern_to_find = r"sarah?"

for string in customer_reviews:
    if (re.search(pattern_to_find, string)):
        sarahs_reviews.append(string)


print(sarahs_reviews)

# Reviews that starts with the letter 'a'
a_reviews = []
pattern_to_find = r"^a"

for string in customer_reviews:
    if (re.search(pattern_to_find, string)):
        a_reviews.append(string)

print(a_reviews)

# Reviews taht ends with the letter 'y'
y_reviews = []
pattern_to_find = r"y$"

for string in customer_reviews:
    if (re.search(pattern_to_find, string)):
        y_reviews.append(string)

print(y_reviews)

# | works like an or

needwant_review = []
pattern_to_find = r"(need|want)ed"

for string in customer_reviews:
    if(re.search(pattern_to_find, string)):
        needwant_review.append(string)

print(needwant_review)

# !!! IMPORTANT
# Removing punctuations from text

no_punct_reviews = []
pattern_to_find = r"[^\w\s]"

# [] => Set of characters we want to match
# ^ at the beggining means "not"
# \w = words characters
# \s = whitespaces
# Find anything that is not a word character or not a space -> punctuation

for string in customer_reviews:
    no_punct_review = re.sub(pattern_to_find, "", string)
    # Here we are replacing everything that is not a word, number or whitespace with an empty string
    no_punct_reviews.append(no_punct_review)

print(no_punct_reviews)