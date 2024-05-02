import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline



dftest = pd.read_csv('Testing.csv')
dftrain = pd.read_csv('Training.csv')
dftrain.drop('Unnamed: 133', axis=1, inplace=True)
columns = list(dftrain.columns)

X_train = dftrain.iloc[:, :-1].values 
y_train = dftrain.iloc[:, 132].values 
X_test = dftest.iloc[:, :-1].values 
y_test = dftest.iloc[:, 132].values 
classifierDT = DecisionTreeClassifier(splitter='best', criterion='entropy', min_samples_leaf=2)
classifierDT.fit(X_train, y_train)
y_predDT = classifierDT.predict(X_test)
imp = classifierDT.feature_importances_
columns = columns[:132]
column_names = ['symptom', 'importance']
df3 = np.vstack((columns, imp)).T
df3 = pd.DataFrame(df3, columns = column_names)
coefficients = classifierDT.feature_importances_
importance_threshold = np.quantile(coefficients, q = 0.75)
low_importance_features = np.array(df3.symptom[np.abs(coefficients) <= importance_threshold])
columns = list(low_importance_features)

for i in columns :
    dftrain.drop(i, axis=1, inplace=True)
    dftest.drop(i, axis=1, inplace=True)

X_train = dftrain.iloc[:, :-1].values 
y_train = dftrain.iloc[:, 33].values 
X_test = dftest.iloc[:, :-1].values 
y_test = dftest.iloc[:, 33].values 

svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Mise à l'échelle des données
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True))  # Classificateur SVM
])
svm_pipeline.fit(X_train, y_train)
newdata = [[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

# Prédisez avec le modèle SVM
proba_svm = svm_pipeline.predict_proba(newdata)
pred_svm = svm_pipeline.predict(newdata)

# Affichez les résultats
print(pred_svm)
print(proba_svm.round(5))
