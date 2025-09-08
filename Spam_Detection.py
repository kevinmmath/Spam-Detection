import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('SMSSpamCollection.csv', sep ='\t')
df.columns = ['label', 'message']

df['label'] = df['label'].map({'ham':0, 'spam':1})

proportion = df['label'].value_counts()
print(proportion)

X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Classification Report: ",classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix: Spam Detection")
plt.xlabel("Predicted")
plt.ylabel("Actual")

my_messages = [
    "Hi mom, I'll be home late for dinner tonight.",
    "URGENT! You have won a 1,000,000 prize. Go to www.example.com to claim now!",
    "Can you please send me the report by 5pm?",
    "Congratulations! You've been selected to receive a free cruise. Text YES to 12345"
]

my_messages_vec = vectorizer.transform(my_messages)
bin_predictions = model.predict(my_messages_vec)

inverse_label_map = {0: 'ham', 1: 'spam'}
predictions = [inverse_label_map[pred] for pred in bin_predictions]

print(predictions)