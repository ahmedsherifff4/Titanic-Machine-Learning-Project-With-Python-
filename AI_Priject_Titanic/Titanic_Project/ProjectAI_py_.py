# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# %% [markdown]
# # Load data
# %%
train = pd.read_csv("Titanic_train.csv")
test  = pd.read_csv("Titanic_test.csv")
# %%
print("Train shape:", train.shape)
print("Test  shape:", test.shape)
# %%
train.head()
# %%
test.head()
# %%
train.info()
# %%
train.describe()
# %% [markdown]
# # 2) Data cleaning
# %%
print("Missing values – Train:")
print(train.isnull().sum())
print()
print("Missing values – Test:")
print(test.isnull().sum())
# %%
# Age & Fare distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(train['Age'].dropna(), kde=True, ax=axes[0], color='blue')
axes[0].set_title('Age Distribution')
sns.histplot(train['Fare'], kde=True, ax=axes[1], color='red')
axes[1].set_title('Fare Distribution')
plt.tight_layout()
plt.show()
# %%
train['Age']      = train['Age'].fillna(train['Age'].median())
test['Age']       = test['Age'].fillna(train['Age'].median())
test['Fare']      = test['Fare'].fillna(train['Fare'].median())
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
test['Embarked']  = test['Embarked'].fillna(train['Embarked'].mode()[0])
# %%
train.isnull().sum()
# %%
test.isnull().sum()
# %%
#data visualization
plt.figure(figsize=(6, 5))
#by data augumtantion
sns.countplot(data=train, x='Survived', hue='Survived', palette='Set2', legend=False)

plt.title('Overall Survival Count')
plt.show()
# %%

fig, axes = plt.subplots(1, 3, figsize=(15, 5))


sns.countplot(data=train, x='Pclass', hue='Survived', ax=axes[0], palette='Set2')
axes[0].set_title('Survival by Pclass')


sns.countplot(data=train, x='Sex', hue='Survived', ax=axes[1], palette='Set2')
axes[1].set_title('Survival by Sex')


sns.countplot(data=train, x='Embarked', hue='Survived', ax=axes[2], palette='Set2')
axes[2].set_title('Survival by Embarked')

plt.tight_layout()
plt.show()
# %%
# Age & Fare distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(train['Age'].dropna(), kde=True, ax=axes[0], color='blue')
axes[0].set_title('Age Distribution')
sns.histplot(train['Fare'], kde=True, ax=axes[1], color='red')
axes[1].set_title('Fare Distribution')
plt.tight_layout()
plt.show()
# %%
# Correlation heatmap (numeric only)
plt.figure(figsize=(10, 6))
numeric_train = train.select_dtypes(include=[np.number])
sns.heatmap(numeric_train.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
# %%
# Extract Title from Name
train['Title'] = train['Name'].str.extract(r',\s*([^\.]+)\.')
test['Title']  = test['Name'].str.extract(r',\s*([^\.]+)\.')
print(train['Title'].value_counts())
# %%
# Visualise survival by title
plt.figure(figsize=(10, 5))
sns.countplot(data=train, x='Title', hue='Survived', palette='Set2')
plt.title('Survival Count by Title')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# %%
# Survival rate by title
survival_rate = train.groupby('Title')['Survived'].mean().sort_values(ascending=False)
survival_rate.plot(kind='bar', color='steelblue', edgecolor='black', figsize=(10, 4))
plt.title('Survival Rate by Title')
plt.ylabel('Survival Rate')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# %%
# Consolidate rare titles
common_titles = ['Mr', 'Miss', 'Mrs', 'Master']
train['Title'] = train['Title'].apply(lambda x: x if x in common_titles else 'Rare')
test['Title']  = test['Title'].apply(lambda x: x if x in common_titles else 'Rare')
print(train['Title'].value_counts())
# %%
# Family size & IsAlone
for df in [train, test]:
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
# %%
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))


sns.kdeplot(data=train[train['Survived'] == 0]['Age'], label='Not Survived', fill=True, color='red')
sns.kdeplot(data=train[train['Survived'] == 1]['Age'], label='Survived', fill=True, color='green')

plt.title('Age Distribution by Survival')
plt.xlabel('Age')
plt.ylabel('Density')
plt.legend()
plt.show()
# %%
# Age bins
for df in [train, test]:
    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100],
                          labels=['Child', 'Teen', 'Young', 'Adult', 'Senior'])
# %%
train.head()
# %%
train.isnull().sum()
# %%
test.isnull().sum()
# %%
# Drop columns not needed for modelling
drop_cols = ['PassengerId', 'Ticket', 'Cabin', 'Name', 'SibSp', 'Parch']
train.drop(columns=drop_cols, inplace=True)
test.drop(columns=drop_cols, inplace=True)
# %%
train.head()
# %%

fig, axes = plt.subplots(1, 4, figsize=(18, 4))

for ax, col in zip(axes, ['Age', 'Fare', 'Pclass', 'FamilySize']):
    sns.boxplot(data=train, y=col, ax=ax)
    ax.set_title(f'{col} Boxplot')

plt.tight_layout()
plt.show()
# %%

train['Fare'] = train['Fare'].apply(np.log1p)
test['Fare'] = test['Fare'].apply(np.log1p)


plt.figure(figsize=(7, 5))
sns.histplot(train['Fare'], kde=True, color='green')
plt.title('Fare Distribution (After Log Transformation)')
plt.show()
# %%
#the boxplots after log transformation
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, col in zip(axes, ['Age', 'Fare', 'Pclass', 'FamilySize']):
    sns.boxplot(data=train, y=col, ax=ax)
    ax.set_title(f'{col} Boxplot')
    

plt.tight_layout()
plt.show()    
# %%
# Encode binary: Sex
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
test['Sex']  = test['Sex'].map({'male': 0, 'female': 1})
# %%
train = pd.get_dummies(train, columns=['Embarked', 'Title'], drop_first=True)
test  = pd.get_dummies(test,  columns=['Embarked', 'Title'], drop_first=True)
# %%
train.head()
# %%
# One-hot encode AgeBin
train = pd.get_dummies(train, columns=['AgeBin'], drop_first=False, prefix='AgeBin')
test = pd.get_dummies(test, columns=['AgeBin'], drop_first=False, prefix='AgeBin')

# Ensure both datasets have the same columns
train_cols = set(train.columns)
test_cols = set(test.columns)

# Add missing columns to test with 0 values
for col in train_cols - test_cols:
    test[col] = 0

# Reorder test columns to match train
test = test[train.columns]
# %%
train.head()
# %%
for df in [train, test]:
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

print("Train columns:", train.columns.tolist())
print("Test columns:", test.columns.tolist())
# %%
train.head()
# %%
# Standard scaling
# min-max scaling can be used for tree-based models the mae the accuracy deacrese
# Note: AgeBin is now one-hot encoded, so we only scale continuous numeric columns

scaler = StandardScaler()
cols_to_scale = ['Age', 'Fare', 'Pclass', 'FamilySize']
train[cols_to_scale] = scaler.fit_transform(train[cols_to_scale])
test[cols_to_scale]  = scaler.transform(test[cols_to_scale])
train.head()
# %%
X = train.drop('Survived', axis=1)
y = train['Survived']
# 75% 25%
# 60% 40%
# 90% 10%
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

test = test[X_train.columns]

print("X_train:", X_train.shape)
print("X_val:  ", X_val.shape)


# %%
# Drop Age first
train.drop('Age', axis=1, inplace=True)
test.drop('Age', axis=1, inplace=True)
# %%
# Correlation heatmap (numeric only)
plt.figure(figsize=(10, 6))
numeric_train = train.select_dtypes(include=[np.number])
sns.heatmap(numeric_train.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
# %%
train.head()
# %%
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)

lr_train_acc = accuracy_score(y_train, lr.predict(X_train))
lr_val_acc   = accuracy_score(y_val,   lr.predict(X_val))

print("=" * 50)
print("Model : Logistic Regression")
print(f"Train Accuracy : {lr_train_acc:.4f}")
print(f"Val   Accuracy : {lr_val_acc:.4f}")
print(classification_report(y_val, lr.predict(X_val)))
# %%
cm = confusion_matrix(y_val, lr.predict(X_val))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression – Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
# %%
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    lr, X_train, y_train, cv=5, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10))

plt.figure(figsize=(10, 5))
plt.plot(train_sizes, train_scores.mean(axis=1), label='Train Accuracy', color='blue')
plt.plot(train_sizes, val_scores.mean(axis=1),   label='Val Accuracy',   color='orange')
plt.fill_between(train_sizes,
                 train_scores.mean(axis=1) - train_scores.std(axis=1),
                 train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1, color='blue')
plt.fill_between(train_sizes,
                 val_scores.mean(axis=1) - val_scores.std(axis=1),
                 val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1, color='orange')
plt.title('Logistic Regression – Learning Curve')
plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
# %%
svm = SVC(random_state=42)
svm.fit(X_train, y_train)

svm_train_acc = accuracy_score(y_train, svm.predict(X_train))
svm_val_acc   = accuracy_score(y_val,   svm.predict(X_val))

print("=" * 50)
print("Model : SVM")
print(f"Train Accuracy : {svm_train_acc:.4f}")
print(f"Val   Accuracy : {svm_val_acc:.4f}")
print(classification_report(y_val, svm.predict(X_val)))
# %%
cm = confusion_matrix(y_val, svm.predict(X_val))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('SVM – Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
# %%
# ── Decision Tree ────────────────────────────────────────────────────────────
dt = DecisionTreeClassifier(random_state=42, max_depth=5)
dt.fit(X_train, y_train)

dt_train_acc = accuracy_score(y_train, dt.predict(X_train))
dt_val_acc   = accuracy_score(y_val,   dt.predict(X_val))

print("=" * 50)
print("Model : Decision Tree")
print(f"Train Accuracy : {dt_train_acc:.4f}")
print(f"Val   Accuracy : {dt_val_acc:.4f}")
print(classification_report(y_val, dt.predict(X_val)))


# %%
cm = confusion_matrix(y_val, dt.predict(X_val))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Decision Tree – Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
# %%
rf = RandomForestClassifier(random_state=42,max_depth=5)
rf.fit(X_train, y_train)

rf_train_acc = accuracy_score(y_train, rf.predict(X_train))
rf_val_acc   = accuracy_score(y_val,   rf.predict(X_val))

print("=" * 50)
print("Model : Random Forest")
print(f"Train Accuracy : {rf_train_acc:.4f}")
print(f"Val   Accuracy : {rf_val_acc:.4f}")
print(classification_report(y_val, rf.predict(X_val)))
# %%
cm = confusion_matrix(y_val, rf.predict(X_val))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest – Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
# %%
# Find best K
k_range = range(1, 21)
train_scores = []
val_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, knn.predict(X_train)))
    val_scores.append(accuracy_score(y_val, knn.predict(X_val)))

# Plot
plt.figure(figsize=(10, 5))
plt.plot(k_range, train_scores, label='Train Accuracy', color='blue')
plt.plot(k_range, val_scores,   label='Val Accuracy',   color='orange')
plt.title('KNN – Finding Best K')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

best_k = val_scores.index(max(val_scores)) + 1
print(f"Best K: {best_k} → Val Accuracy: {max(val_scores):.4f}")

# Train with best K
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

knn_train_acc = accuracy_score(y_train, knn.predict(X_train))
knn_val_acc   = accuracy_score(y_val,   knn.predict(X_val))

print("=" * 50)
print("Model : KNN")
print(f"Train Accuracy : {knn_train_acc:.4f}")
print(f"Val   Accuracy : {knn_val_acc:.4f}")
print(classification_report(y_val, knn.predict(X_val)))
# %%
cm = confusion_matrix(y_val, knn.predict(X_val))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('KNN – Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# %%
# ── Logistic Regression Tuning ───────────────────────────────────────────────
lr_params = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}
lr_grid = GridSearchCV(LogisticRegression(random_state=42, max_iter=10), param_grid=
                       lr_params, cv=5, scoring='accuracy', n_jobs=-1)
lr_grid.fit(X_train, y_train)
print("Best LR params:", lr_grid.best_params_)
print("Best CV score :", round(lr_grid.best_score_, 4))

lr_best = lr_grid.best_estimator_
lr_best_train_acc = accuracy_score(y_train, lr_best.predict(X_train))
lr_best_val_acc   = accuracy_score(y_val,   lr_best.predict(X_val))

print("=" * 50)
print("Model : Logistic Regression (Tuned)")
print(f"Train Accuracy : {lr_best_train_acc:.4f}")
print(f"Val   Accuracy : {lr_best_val_acc:.4f}")
print(f"Gap (Overfitting): {lr_best_train_acc - lr_best_val_acc:.4f}")
print(classification_report(y_val, lr_best.predict(X_val)))
# %%
cm = confusion_matrix(y_val, lr_best.predict(X_val))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression (Tuned) – Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
# %%
# ── SVM Tuning ───────────────────────────────────────────────────────────────
svm_params = {
    'C': [0.1, 1, 10],
    #guessing that gamma is more relevant for RBF kernel, but we can still include it for linear to see if it has any effect (it should be ignored by the linear kernel).
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto']
}
svm_grid = GridSearchCV(SVC(random_state=42),
                        svm_params, cv=5, scoring='accuracy', n_jobs=-1)
svm_grid.fit(X_train, y_train)
print("Best SVM params:", svm_grid.best_params_)
print("Best CV score  :", round(svm_grid.best_score_, 4))

svm_best = svm_grid.best_estimator_
svm_best_train_acc = accuracy_score(y_train, svm_best.predict(X_train))
svm_best_val_acc   = accuracy_score(y_val,   svm_best.predict(X_val))

print("=" * 50)
print("Model : SVM (Tuned)")
print(f"Train Accuracy : {svm_best_train_acc:.4f}")
print(f"Val   Accuracy : {svm_best_val_acc:.4f}")
print(f"Gap (Overfitting): {svm_best_train_acc - svm_best_val_acc:.4f}")
print(classification_report(y_val, svm_best.predict(X_val)))


# %%
cm = confusion_matrix(y_val, svm_best.predict(X_val))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('SVM (Tuned) – Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
# %%
# ── Decision Tree Tuning ──────────────────────────────────────────────────────
# Test different max_depth values to find the optimal balance
depth_range = range(2, 16)
#over 16 caouse overfitting, so i stopped at 16
dt_train_scores = []
dt_val_scores = []

for depth in depth_range:
    dt_temp = DecisionTreeClassifier(random_state=42, max_depth=depth)
    dt_temp.fit(X_train, y_train)
    dt_train_scores.append(accuracy_score(y_train, dt_temp.predict(X_train)))
    dt_val_scores.append(accuracy_score(y_val, dt_temp.predict(X_val)))

# Plot to find optimal depth

plt.figure(figsize=(10, 5))
plt.plot(depth_range, dt_train_scores, label='Train Accuracy', color='blue', marker='o')
plt.plot(depth_range, dt_val_scores,   label='Val Accuracy',   color='orange', marker='s')
plt.title('Decision Tree – Finding Optimal Max Depth')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Find the optimal depth
optimal_depth = depth_range[dt_val_scores.index(max(dt_val_scores))]
print(f"Optimal Max Depth: {optimal_depth}")

# Train the final model with the optimal depth
dt_best = DecisionTreeClassifier(random_state=42, max_depth=4)
dt_best.fit(X_train, y_train)

dt_best_train_acc = accuracy_score(y_train, dt_best.predict(X_train))
dt_best_val_acc   = accuracy_score(y_val,   dt_best.predict(X_val))

print("=" * 50)
print("Model : Decision Tree (Tuned)")
print(f"Train Accuracy : {dt_best_train_acc:.4f}")
print(f"Val   Accuracy : {dt_best_val_acc:.4f}")
print(f"Gap (Overfitting): {dt_best_train_acc - dt_best_val_acc:.4f}")
print(classification_report(y_val, dt_best.predict(X_val)))
# %%
cm = confusion_matrix(y_val, dt_best.predict(X_val))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Decision Tree (Tuned) – Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
# %%
# ── Model Comparison ──────────────────────────────────────────────────────────
results = pd.DataFrame({
    # xgboost can be added here as well if we have time to train it, but it might take a bit longer to run compared to the other models.
    # catboost can also be added, but it might require additional installation and setup, so we can consider it if we have time after covering the main models. 
    'Model': [
        'Logistic Regression',
        'Logistic Regression (Tuned)',
        'SVM',
        'SVM (Tuned)',
        'Decision Tree',
        'Decision Tree (Tuned)',
        'Random Forest',
        'KNN'
    ],
    'Train Accuracy': [
        lr_train_acc,
        lr_best_train_acc,
        svm_train_acc,
        svm_best_train_acc,
        dt_train_acc,
        dt_best_train_acc,
        rf_train_acc,
        knn_train_acc
    ],
    'Val Accuracy': [
        lr_val_acc,
        lr_best_val_acc,
        svm_val_acc,
        svm_best_val_acc,
        dt_val_acc,
        dt_best_val_acc,
        rf_val_acc,
        knn_val_acc
    ]
}).sort_values('Val Accuracy', ascending=False).reset_index(drop=True)

print(results.to_string(index=False))

# Graph
fig, ax = plt.subplots(figsize=(12, 6))
x = range(len(results))
width = 0.35

bars1 = ax.barh([i + width/2 for i in x], results['Train Accuracy'], width, label='Train Accuracy', color='steelblue')
bars2 = ax.barh([i - width/2 for i in x], results['Val Accuracy'],   width, label='Val Accuracy',   color='orange')

ax.axvline(x=0.80, color='red', linestyle='--', label='80% threshold')
ax.set_yticks(list(x))
ax.set_yticklabels(results['Model'])
ax.set_xlabel('Accuracy')
ax.set_title('Model Comparison – Train vs Val Accuracy')
ax.legend()
plt.tight_layout()
plt.show()
# %%
# ── Auto-select Best Model ─────────────────────────────────────────────────────
models_dict = {
    'Logistic Regression':        lr,
    'Logistic Regression (Tuned)': lr_best,
    'SVM':                        svm,
    'SVM (Tuned)':                svm_best,
    'Decision Tree':              dt,
    'Decision Tree (Tuned)':      dt_best,
    'Random Forest':              rf,
    'KNN':                        knn
}

val_accs = {
    'Logistic Regression':        lr_val_acc,
    'Logistic Regression (Tuned)': lr_best_val_acc,
    'SVM':                        svm_val_acc,
    'SVM (Tuned)':                svm_best_val_acc,
    'Decision Tree':              dt_val_acc,
    'Decision Tree (Tuned)':      dt_best_val_acc,
    'Random Forest':              rf_val_acc,
    'KNN':                        knn_val_acc
}

best_name  = max(val_accs, key=val_accs.get)
best_model = models_dict[best_name]
print(f"\n Best Model: {best_name} ({round(val_accs[best_name], 4)})")
# %%
print("X_train columns:", X_train.columns.tolist())
print("test columns:", test.columns.tolist())
# %%
print("X_train columns:", X_train.columns.tolist())
print("test columns:", test.columns.tolist())
print("Missing:", set(X_train.columns) - set(test.columns))
# %%
# Ensure test has the same columns as X_train
# خلي الـ test يتطابق مع X_train columns
missing_cols = set(X_train.columns) - set(test.columns)
for col in missing_cols:
    test[col] = 0  # أضف الأعمدة الناقصة بقيمة 0

extra_cols = set(test.columns) - set(X_train.columns)
test = test.drop(columns=extra_cols)
test = test[X_train.columns]

test_preds = best_model.predict(test)
print("Prediction distribution:", pd.Series(test_preds).value_counts().to_dict())

submission = pd.DataFrame({'Prediction': test_preds})
submission.to_csv('titanic_predictions.csv', index=False)
print("Saved titanic_predictions.csv ")
submission.head(10)
# %%
result_df = test.copy()
result_df['Prediction'] = test_preds
result_df['Prediction'] = result_df['Prediction'].map({0: 'Died', 1: 'Survived'})

result_df.to_csv('titanic_predictions.csv', index=False)
print("Saved titanic_predictions.csv ")
result_df.head(10)
# %%

#sheet = sheets.InteractiveSheet(df=result_df)
# %% [markdown]
# # "I hope this project meets your expectations Shosha."