import pandas as pd

df = pd.read_csv("sms.tsv", sep="\t", header=None) #loading data set using tabs, no header row
df.columns = ["label", "message"] #added column name 

#print(df.head()) #testing

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#converting 'ham' and 'spam' to 0 and 1
le = LabelEncoder()
df["label_num"] = le.fit_transform(df["label"])

#split into training sets
X = df["message"] #input message
y = df["label_num"] #output 0 or 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train) #learns and turns message into bag of words count vector
X_test_vec = vectorizer.transform(X_test) #uses the same learned word list to convert test data

#print(X_train_vec.shape) #testing

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train_vec, y_train) #teaching the model

#accuracy = model.score(X_test_vec, y_test) #testing
#print(f"Model accuracy: {accuracy:.2f}")

def predict_spam(message): #takes a message and predits if it is spam or not
    message_vec = vectorizer.transform([message])
    prediction = model.predict(message_vec) [0]
    label = le.inverse_transform([prediction]) [0]
    print(f"Prediction: {label}")

while True: #get message as an input, type "exit" to end the loop
    message = input("Type your SMS here: ")
    if message.strip().lower() == "exit":
        break
    else:
        predict_spam(message)

#predict_spam("You've won a free prize!") #test
#predict_spam("How are you?") #test
