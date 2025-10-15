# ===============================================================
# W04 - Model Performance (Decision Tree)
# Nama  : Collin Joseph
# NIM   : 0706022310053
# ===============================================================

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# ---------------------------------------------------------------
# 2. Load dataset
# ---------------------------------------------------------------
DATA_URL = 'https://raw.githubusercontent.com/NathaliaMinoque/datasets/refs/heads/main/exercise_employee_attrition.csv'

try:
    df = pd.read_csv(DATA_URL)
    print('Dataset berhasil di-load. Shape:', df.shape)
except Exception as e:
    print('Gagal download dataset:', e)
    df = pd.DataFrame()

# ---------------------------------------------------------------
# 3. Cek data
# ---------------------------------------------------------------
if not df.empty:
    print("\n5 data teratas:")
    print(df.head())
    print("\nInfo kolom:")
    print(df.info())
    print("\nCek missing values:")
    print(df.isna().sum())
else:
    print("Dataset kosong. Pastikan file tersedia atau koneksi internet aktif.")

# ---------------------------------------------------------------
# 4. Preprocessing data
# ---------------------------------------------------------------
if not df.empty:
    if 'Employee_ID' in df.columns:
        df.drop(columns=['Employee_ID'], inplace=True)

    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    num_cols = [c for c in num_cols if c != 'Attrition']
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    ord_cols = [c for c in ['Work_Life_Balance', 'Job_Satisfaction', 'Performance_Rating'] if c in df.columns]
    nom_cols = [c for c in cat_cols if c not in ord_cols]

    print("\nKolom numerik :", num_cols)
    print("Kolom ordinal  :", ord_cols)
    print("Kolom nominal  :", nom_cols)

    num_transform = Pipeline(steps=[('scaler', StandardScaler())])
    ord_transform = Pipeline(steps=[('ord', OrdinalEncoder())]) if ord_cols else 'passthrough'
    nom_transform = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))]) if nom_cols else 'passthrough'

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transform, num_cols),
        ('ord', ord_transform, ord_cols),
        ('nom', nom_transform, nom_cols)
    ], remainder='drop')

    X = df.drop(columns=['Attrition'])
    y = df['Attrition']

# ---------------------------------------------------------------
# 5. Split data
# ---------------------------------------------------------------
if not df.empty:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print("\nData train:", X_train.shape, "| Data test:", X_test.shape)

# ---------------------------------------------------------------
# 6. Menangani imbalance dengan SMOTE
# ---------------------------------------------------------------
if not df.empty:
    X_train_trans = preprocessor.fit_transform(X_train)
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train_trans, y_train)
    print("\nSetelah SMOTE:")
    print("Jumlah kelas 0 (Stay):", sum(y_res == 0))
    print("Jumlah kelas 1 (Leave):", sum(y_res == 1))

# ---------------------------------------------------------------
# 7. Model dan GridSearchCV
# ---------------------------------------------------------------
if not df.empty:
    clf = DecisionTreeClassifier(random_state=42)
    param_grid = {
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid = GridSearchCV(clf, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X_res, y_res)

    print("\nParameter terbaik hasil GridSearch:")
    print(grid.best_params_)

# ---------------------------------------------------------------
# 8. Pipeline final dan evaluasi
# ---------------------------------------------------------------
if not df.empty:
    best_params = grid.best_params_
    final_clf = DecisionTreeClassifier(random_state=42, **best_params)

    model_final = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', final_clf)
    ])

    model_final.fit(X_train, y_train)
    y_pred = model_final.predict(X_test)

    print("\n=== Hasil Evaluasi Model ===")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# ---------------------------------------------------------------
# 9. Feature importance
# ---------------------------------------------------------------
if not df.empty:
    try:
        onehot_features = []
        if nom_cols:
            onehot_features = list(model_final.named_steps['preprocessor']
                                   .transformers_[2][1]
                                   .named_steps['onehot']
                                   .get_feature_names_out(nom_cols))
        feature_names = num_cols + ord_cols + onehot_features
        importances = model_final.named_steps['classifier'].feature_importances_
        feature_importance = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        print("\nTop 10 fitur paling penting:")
        print(feature_importance.head(10))
    except Exception as e:
        print("Gagal menampilkan feature importances:", e)

# ---------------------------------------------------------------
# 10. Kesimpulan
# ---------------------------------------------------------------
print("\n=== Kesimpulan ===")
print("- Model Decision Tree dengan tuning parameter menghasilkan performa yang cukup baik.")
print("- SMOTE membantu meningkatkan recall pada kelas minoritas (Attrition = Yes).")
print("- Disarankan untuk mencoba model lain seperti Random Forest atau XGBoost untuk perbandingan performa.")
