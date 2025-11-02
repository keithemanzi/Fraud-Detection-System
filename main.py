# Step one, load the data set

import pandas as pd

# Using the right file path, r at the begining of file path because its windows not linux
df = pd.read_csv(r'c:\Users\Keith Emanzi\Documents\Fraud Detection Szystem\archive\creditcard.csv')

# Preview the data
print(df.head())
print(df.shape)

# Checking for missing values, this should return 0 throught, if not,
# then we fill them up with averages or like we drop them coy ML model do not handle missing values
print(df.isnull().sum())


#  Normalising the AMT column because its an outlier in the data and could messup the model prediction
from sklearn.preprocessing import StandardScaler

df['normAmount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))

# Drop the original Amount Column because now we got the scaled version and Time because its not a time stamp of the transaction just seconds after theprevios one
df = df.drop(['Amount', 'Time'], axis=1)
# Now they were dropped so lets go baby boi
print(df.head())
print(df.shape)


# We split the data such that the model learns from one part and get tested on the other
from sklearn.model_selection import train_test_split

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Split into train and test sets
# x is the features except class and z is the target fraud or not
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # testsize = 0.3, means we have held out 30 percent of the data for testing 



# Logistic regression testing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Fraud is rare be attentive!!! thats what thistells the model
model = LogisticRegression(class_weight='balanced')

# Alles gute class 0 (legit transactions) are 85307 and class 1 (the fakes) are 136. so the model is a success. 




# Now the visuals to spice it up. 

# CONFUSION MATRIX; Shows how many fake vs legit transactions the model got correct
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot it
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Fraud"])

disp.plot()


# ROC CURVE; Shows how well the model diffrentiates the legit transactions from the fakes
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Get predicted probabilities
model = LogisticRegression()
model.fit(X_train, y_train)  # This must come first

y_probs = model.predict_proba(X_test)[:, 1]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Plot it
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

