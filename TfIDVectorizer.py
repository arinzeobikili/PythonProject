from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer()

tfidf = vec.fit_transform(['I like machine learning and clustering algorithms',
    'Apples, oranges, and any other fruit are healthy',
    'Is it feasible with machine learning algorithms?',
    'My family is healthy because of the health fruits'])


#print(tfidf.A) #--- Print out the document term matrix
print((tfidf*tfidf.T).A) #----- Print out the similarity matrix

