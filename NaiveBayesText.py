from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

training_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

#Printing out the training set - 10 lines, and the target.
#print("\n".join(training_data.data[0].split("\n")[:10]))
#print("Target is:", training_data.target_names[training_data.target[10]])

# we just count the word occurrences
count_vector = CountVectorizer()
x_train_counts = count_vector.fit_transform(training_data.data)
#print(count_vector.vocabulary_)

# we transform the word occurrences into tf-idf
# TfidfVectorizer = CountVectorizer + TfidfTransformer
tfid_transformer = TfidfTransformer()
x_train_tfidf = tfid_transformer.fit_transform(x_train_counts)

#print(x_train_tfidf) #Print TFIDF

model = MultinomialNB().fit(x_train_tfidf, training_data.target)


#Try for new group of sentences - predict
new = ['My favourite topic has something to do with quantum physics and quantum mechanics',
       'This has nothing to do with church or religion',
       'Software engineering is getting hotter and hotter nowadays',
       'Nigeria is an awesome country',
       'testing is a niche skill in technology']

x_new_counts = count_vector.transform(new)
x_new_tfidf = tfid_transformer.transform(x_new_counts)

predicted = model.predict(x_new_tfidf)
print(predicted)

for doc, category in zip(new, predicted):
    print('%r --------> %s' % (doc, training_data.target_names[category]))