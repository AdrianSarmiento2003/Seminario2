import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import joblib


from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

import seaborn as sns


#df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/cicids2017_cleaned.csv")
df = pd.read_csv("cicids2017_cleaned.csv")
label_col = "Attack Type"
df[label_col] = df[label_col].astype(str)

# Binarizar: Normal -> 0, Ataque -> 1

df["y"] = df[label_col].apply(lambda x: 0 if x.upper().strip() in ["BENIGN", "NORMAL", "NORMAL TRAFFIC"] else 1)

# Lista de features

features = [

  "Destination Port","Flow Duration",
  "Total Fwd Packets","Total Length of Fwd Packets","Fwd Packet Length Max","Fwd Packet Length Min",
  "Fwd Packet Length Mean","Fwd Packet Length Std","Bwd Packet Length Max","Bwd Packet Length Min",
  "Bwd Packet Length Mean","Bwd Packet Length Std","Flow Bytes/s","Flow Packets/s",
  "Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min",
  "Fwd IAT Mean","Fwd IAT Std","Fwd IAT Max","Fwd IAT Min",
  "Bwd IAT Mean","Bwd IAT Std","Bwd IAT Max","Bwd IAT Min",
  "Fwd Packets/s","Bwd Packets/s",
  "Min Packet Length","Max Packet Length","Packet Length Mean","Packet Length Std","Packet Length Variance",
  "FIN Flag Count","PSH Flag Count","ACK Flag Count",
  "Average Packet Size","Active Mean","Active Max","Active Min","Idle Mean","Idle Max","Idle Min",

]


# Filtrar columnas que sí existen
features = [f for f in features if f in df.columns]
print("Features usadas:", features)


X = df[features].fillna(0).astype(float)
y = df["y"].astype(int)




X_train, X_temp, y_train, y_temp = train_test_split(
  X, y, test_size=0.3, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
  X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42

)


print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")


pipe = Pipeline([
  ("scaler", MinMaxScaler()),
  ("rf", RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"))
])

t0 = time.time()
pipe.fit(X_train, y_train)
t_train = time.time() - t0
print(f"\n Entrenamiento terminado en {t_train:.1f}s")

# Validación

y_pred_val = pipe.predict(X_val)
print("\n Métricas del Modelo Base (Validación):")
print(classification_report(y_val, y_pred_val, digits=4))


param_grid = {
  "rf__n_estimators": [100, 200],
  "rf__criterion": ["gini", "entropy"],
  "rf__max_depth": [50, 150, None],
  "rf__min_samples_split": [2, 5],
  "rf__min_samples_leaf": [1, 2]
}


grid_search = GridSearchCV(
  pipe,
  param_grid=param_grid,
  cv=3,
  scoring="f1",
  verbose=3,
  n_jobs=-1

)

grid_search.fit(X_train, y_train)
print("\n Mejores hiperparámetros encontrados:", grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred_test = best_model.predict(X_test)

print("\n Reporte Final (Test Set):")
print(classification_report(y_test, y_pred_test, digits=4))

# Matriz de confusión

cm = confusion_matrix(y_test, y_pred_test)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusión - Test Set")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

joblib.dump(best_model, "rf_pipeline_cicids.joblib")
print(" Modelo guardado como rf_pipeline_cicids.joblib")
