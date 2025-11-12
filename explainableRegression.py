import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Sarcina AI: Regresie Explicativă (Explainable Regression) sau Analiza Importanței Caracteristicilor Secvențiale (Sequential Feature Importance Analysis).
# Modelul AI: Un model bazat pe arbori, cum ar fi XGBoost sau Random Forest.
# Metodologia "Magică": Ingineria Caracteristicilor Temporale (Lagged Feature Engineering) + SHAP (SHapley Additive exPlanations).

# Unitatea noastră de analiză nu este "pacientul" sau "ciclul", ci "un moment-cadru" (a frame in time).
# Modelul AI va fi antrenat să răspundă la întrebarea: "Pentru a prezice MR area cm2 la Cadrul t (de ex., Frame 3),
# care este impactul LA area cm2 la Cadrul t (același timp), și care este impactul LA area cm2 de la Cadrul t-1 (Frame 2)?"
# Vom folosi aceeași metodologie XGBoost + SHAP

FILE_PATH = "pacienti.csv"

FEATURE_COLUMNS = [
    'LA area cm2', 'LA length cm', 'LA volume ml',
    # 'MV tenting height mm',
    # 'PML angle degrees',
    'MV annulus mm', 'LV area cm2', 'LV length cm',
    'LV volume ml', 'RR interval msec'
]
TARGET_COLUMN = 'MR area cm2'
ID_COLUMN = 'Numeric ID'
CYCLE_COLUMN = 'Cycle'
SYSTOLE_COLUMN = 'systole=1 diastole=0'
FRAME_COLUMN = 'Frame' # Avem nevoie de această coloană pentru sortare

# --- 2. Încărcare, CURĂȚARE și Filtrare ---

def clean_value(val):
    """ Funcție robustă pentru a curăța o singură valoare. """
    if isinstance(val, str):
        val = val.strip(" []")
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


try:
    df = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"EROARE: Fișierul nu a fost găsit la {FILE_PATH}.")
    exit()

print(f"S-au încărcat {df.shape[0]} rânduri.")

df = df[df[SYSTOLE_COLUMN] == 1].copy()
print(f"Au rămas {df.shape[0]} rânduri după filtrarea systole=1.")

# Curățăm toate coloanele numerice
cols_to_clean = FEATURE_COLUMNS + [TARGET_COLUMN]
for col in cols_to_clean:
    if col in df.columns:
        df[col] = df[col].apply(clean_value)

# Selectăm doar coloanele de care avem nevoie
all_cols = [ID_COLUMN, CYCLE_COLUMN, FRAME_COLUMN] + FEATURE_COLUMNS + [TARGET_COLUMN]
df_clean = df[all_cols].dropna()

print(f"Au rămas {df_clean.shape[0]} rânduri după curățare și eliminarea NaN-urilor inițiale.")

if df_clean.empty:
    print("EROARE: Nu au rămas date după filtrare și curățare.")
    exit()
# --- 3. Ingineria Caracteristicilor Temporale (Nivel 'Frame') ---

# Sortăm datele corect: Pacient -> Ciclu -> Cadru
# Aceasta este cea mai importantă linie!
df_clean.sort_values(by=[ID_COLUMN, CYCLE_COLUMN, FRAME_COLUMN], inplace=True)

print("Se creează caracteristicile decalate (lagged features) la nivel de cadru (Frame)...")

# Câte cadre (frames) în urmă ne uităm?
N_LAGS = 2 # Vom crea t-1 și t-2.

# Grupăm după ID-ul pacientului ȘI ciclu.
# .shift() se va opri și o va lua de la capăt pentru fiecare ciclu nou al fiecărui pacient.
# Acest lucru previne "scurgerea" datelor de la Ciclul 1 la Ciclul 2.
grouped = df_clean.groupby([ID_COLUMN, CYCLE_COLUMN])

for lag in range(1, N_LAGS + 1):
    for col in FEATURE_COLUMNS:
        # Aplicăm .shift() în interiorul fiecărui grup
        df_clean[f'{col}_lag{lag}'] = grouped[col].shift(lag)

# Decalajele vor crea valori NaN la începutul fiecărui ciclu
# (de ex., la Frame 1, nu există Frame t-1). Le eliminăm.
df_final = df_clean.dropna()

print(f"Datele finale au {df_final.shape[0]} rânduri gata de antrenament (după crearea decalajelor).")

# --- 4. Pregătirea pentru Modelul AI ---

# Separăm caracteristicile (X) de țintă (y)
# X conține acum caracteristicile curente (t) ȘI caracteristicile decalate (t-1, t-2)
y = df_final[TARGET_COLUMN]
X = df_final.drop(columns=[TARGET_COLUMN, ID_COLUMN, CYCLE_COLUMN, FRAME_COLUMN])

# Împărțim datele
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 5. Antrenarea Modelului AI (XGBoost) ---
print("Se antrenează modelul XGBoost...")
model = xgb.XGBRegressor(objective='reg:squarederror',
                         n_estimators=100,
                         early_stopping_rounds=10)

model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=False)

print("Model antrenat.")

# --- 6. Analiza Importanței (SHAP) ---
print("Se calculează valorile SHAP...")

# Folosim TreeExplainer explicit
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

print("Analiza SHAP a fost finalizată. Se generează graficele.")

# --- Graficul 1: Importanța Globală (Răspunsul Direct) ---
# Acesta vă va arăta ordinea caracteristicilor, inclusiv decalajele!
print("Se afișează Graficul 1: Importanța Globală a Caracteristicilor.")
print("Acesta arată CE caracteristică (inclusiv decalajul la nivel de cadru) are cel mai mare impact mediu.")

shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title("Importanța Globală (Nivel Cadru)")
plt.show()

# --- Graficul 2: Graficul de Dispersie SHAP (Mai Detaliat) ---
print("Se afișează Graficul 2: Analiza Detaliată a Impactului.")
print("Acesta arată CUM o caracteristică (de ex. 'LA area_lag1') influențează ținta.")

shap.summary_plot(shap_values, X_test, show=False)
plt.title("Analiza Detaliată a Impactului SHAP (Nivel Cadru)")
plt.show()