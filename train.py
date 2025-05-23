# train_model.py
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

digits = datasets.load_digits()
X = digits.images.reshape((len(digits.images), -1))
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = SVC(gamma=0.001)
model.fit(X_train, y_train)

joblib.dump(model, "digit_model.h5")
print(" Model saved as sklearn_digit_model.pkl")
