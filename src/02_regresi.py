# -*- coding: utf-8 -*-
"""
Modul 02 — Regresi: memprediksi konsumsi BBM harian per unit (liter).

Data: telemetri armada SIMULASI (lihat README). Target: konsumsi_bbm_liter
pada hari unit beroperasi (jam_operasi > 0). Model dibandingkan:
regresi linear dan Random Forest. Model terbaik disimpan ke
models/model_bbm.joblib untuk dipakai API (modul 07).
"""
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLAH = os.path.join(BASE, "data", "olahan")
OUT = os.path.join(BASE, "output")
MODELS = os.path.join(BASE, "models")
os.makedirs(MODELS, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 130})

FITUR = ["jam_operasi", "muatan_ton", "jarak_km", "umur_tahun", "getaran_mm_s",
         "suhu_mesin_c", "hari_sejak_servis", "hujan", "tipe_truck"]
tel = pd.read_csv(os.path.join(OLAH, "telemetri_armada_simulasi.csv"))
tel = tel[tel["jam_operasi"] > 0].copy()
tel["tipe_truck"] = (tel["tipe"] == "Haul Truck").astype(int)
X = tel[FITUR]; y = tel["konsumsi_bbm_liter"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

hasil = []
model_terbaik, skor_terbaik = None, -1e18
for nama, model in [("Regresi Linear", LinearRegression()),
                    ("Random Forest", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))]:
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    r2 = r2_score(yte, pred)
    hasil.append(dict(model=nama, r2=round(r2, 4),
                      rmse_liter=round(float(np.sqrt(mean_squared_error(yte, pred))), 1),
                      mae_liter=round(float(mean_absolute_error(yte, pred)), 1)))
    if r2 > skor_terbaik:
        skor_terbaik, model_terbaik = r2, model
    if nama == "Random Forest":
        imp = pd.Series(model.feature_importances_, index=FITUR).sort_values()
        fig, ax = plt.subplots(figsize=(8.5, 4.4))
        ax.barh(imp.index, imp.values, color="#1f6fb2")
        ax.set_title("Pentingnya fitur — Random Forest (prediksi konsumsi BBM)")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "regresi_fitur.png")); plt.close(fig)
        pred_rf = pred

pd.DataFrame(hasil).to_csv(os.path.join(OUT, "metrik_regresi.csv"), index=False)
joblib.dump({"model": model_terbaik, "fitur": FITUR}, os.path.join(MODELS, "model_bbm.joblib"))

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.scatter(yte, pred_rf, s=8, alpha=0.45, color="#1f6fb2")
lim = [0, max(yte.max(), pred_rf.max()) * 1.05]
ax.plot(lim, lim, color="#c0501f", lw=1.5)
ax.set_xlabel("Aktual (liter)"); ax.set_ylabel("Prediksi (liter)")
ax.set_title("Prediksi vs aktual konsumsi BBM — Random Forest")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "regresi_prediksi_vs_aktual.png")); plt.close(fig)

print(pd.DataFrame(hasil).to_string(index=False))
print("Model tersimpan: models/model_bbm.joblib")
