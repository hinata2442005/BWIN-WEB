# %%
# Number 1
# import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, ConfusionMatrixDisplay, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# %%
# Number 2
# Load dataset
df = pd.read_csv("Breast_Cancer.csv")
df.head()

# %%
# Number 3
# general summary of the dataframe
df.info()

# %%
# Number 4
# Convert categorical feature into binary variables
# M = Malignant = 1 = Positive, B = Benign = 0 = Negative
df["diagnosis"] = np.where (df["diagnosis"]=="M", 1, 0)

# %%
# Number 5
# Show 5 first columns
df.head()

# %%
# Number 6
# Count number of observations in each class
print(df["diagnosis"].value_counts())
benign, malignant = df['diagnosis'].value_counts()
print('Number of cells labeled Benign: ', benign)
print('Number of cells labeled Malignant : ', malignant)
print('')
print('% of cells labeled Benign', round(benign / len(df) * 100, 2), '%')
print('% of cells labeled Malignant', round(malignant / len(df) * 100, 2), '%')
import seaborn as sns

# Visualize distribution of classes
colors = ["pink","Red"]
sns.countplot(x="diagnosis", data=df, palette= colors)
plt.title('Count of Diagnosis (M = 1: Malignant, B = 0: Benign)')
plt.show()

# %%
# Number 7
# Drop the column "id" and "diagnosis"
Dataframe = df.drop(["id","diagnosis"], axis=1)

# %%
# Number 8
# Show first 5 columns
Dataframe.head()

# %%
# Number 9
import numpy as np
np.bool = bool

# Generate and visualize the correlation matrix
corr = Dataframe.corr().round(2)

# Mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set figure size
f, ax = plt.subplots(figsize=(20, 20))

# Define custom colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap
sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.tight_layout()

# %%
# Number 9
# Drop all "worst" columns
Dataframe = Dataframe.drop(["radius_worst", "texture_worst", "perimeter_worst", "area_worst","smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"], axis="columns")

# Drop all columns related to the "perimeter" and "area" attributtes
Dataframe = Dataframe.drop(["perimeter_mean", "area_mean", "perimeter_se", "area_se"], axis="columns")

# Drop all columns related to the "concavity" and "concave point" attribute
Dataframe = Dataframe.drop(["concavity_mean", "concavity_se", "concave points_mean", "concave points_se"], axis="columns")

# %%
# Number 10
# Verify remaining columns
Dataframe.columns

# %%
# Number 11
# Draw the heatmap again, with the new correlation matrix
corr = Dataframe.corr().round(2)
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
plt.tight_layout()

# %%
# Number 12
X = Dataframe
Y = df["diagnosis"]

# %%
# Number 13
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size =0.3, random_state = 40)

# %%
# Number 14
Y_train.head()

# %%
# Number 15
X_train.head()

# %%
# Number 16
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %%
# Number 17
svm_model = SVC(kernel='rbf', C=1, gamma=0.1,probability=True)
svm_model.fit (X_train, Y_train)

# %%
# Number 18
Y_pred = svm_model.predict(X_test)

# %%
# Number 19
# Check if the model is overfitting or underfitting
print("Train Accuracy:", svm_model.score(X_train, Y_train))
print("Test Accuracy:", svm_model.score(X_test, Y_test))

# %%
# Number 20
# Train accuracy
train_acc = svm_model.score(X_train, Y_train) * 100
# Test accuracy
test_acc = svm_model.score(X_test, Y_test) * 100
print(f"Train Accuracy: {train_acc:.2f}%")
print(f"Test Accuracy: {test_acc:.2f}%")

# %%
# Number 21
# Print classification report
print(classification_report(Y_test, Y_pred))

# %%
# Number 22
# Calculate Metrics for testing
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# %%
# Number 23
cm = confusion_matrix(Y_test, Y_pred,labels=[0,1])
print(cm)

# %%
# Number 24
# cm = [[TN, FP],
#       [FN, TP]]
TN, FP, FN, TP = cm.ravel()  # tách từng giá trị

print("True Negative (TN):", TN)
print("False Positive (FP):", FP)
print("False Negative (FN):", FN)
print("True Positive (TP):", TP)

# %%
# Number 25
cm_df = pd.DataFrame(
    [[TN, FP],
     [FN, TP]],
    index=["True B", "True M"],
    columns=["Pred B", "Pred M"]
)

print(cm_df)

# %%
# Number 26
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Reds')

# %%
# Number 26
# TCalculate FPR, TPR
fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
auc = roc_auc_score(Y_test, Y_pred)

# Plot
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color="red", linewidth=3, label=f'ROC Curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='black', linewidth=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Support Vector Machine')
plt.legend()
plt.grid(True)
plt.show()

# %%
# Number 27
# Save model + scaler
import joblib
joblib.dump(svm_model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print(">>> model.pkl và scaler.pkl đã được tạo thành công!")
