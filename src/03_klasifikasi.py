# -*- coding: utf-8 -*-
"""
Modul 03 — Klasifikasi: memprediksi unit yang akan gagal dalam 7 hari.

Data: telemetri armada SIMULASI. Ini pola masalah predictive maintenance:
beri skor risiko per unit-hari dari getaran, suhu, hari sejak servis,
umur, jam operasi, dan hujan. Model dibandingkan: regresi logistik
(diskalakan) dan Random Forest. Model terbaik disimpan ke
models/model_risiko.joblib untuk API (modul 07).
"""
import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLAH = os.path.join(BASE, "data", "olahan")
OUT = os.path.join(BASE, "output")
MODELS = os.path.join(BASE, "models")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 130})

FITUR = ["getaran_mm_s", "suhu_mesin_c", "hari_sejak_servis", "umur_tahun",
         "jam_operasi", "hujan"]
tel = pd.read_csv(os.path.join(OLAH, "telemetri_armada_simulasi.csv"))
tel = tel[tel["jam_operasi"] > 0].copy()
X = tel[FITUR]; y = tel["gagal_dalam_7_hari"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler().fit(Xtr)
hasil, terbaik = [], None
for nama, model, pakai_skala in [
        ("Regresi Logistik", LogisticRegression(max_iter=1000), True),
        ("Random Forest", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1), False)]:
    a, b = (scaler.transform(Xtr), scaler.transform(Xte)) if pakai_skala else (Xtr, Xte)
    model.fit(a, ytr)
    prob = model.predict_proba(b)[:, 1]
    pred = (prob >= 0.5).astype(int)
    auc = roc_auc_score(yte, prob)
    hasil.append(dict(model=nama, auc=round(auc, 4),
                      akurasi=round(accuracy_score(yte, pred), 4),
                      f1=round(f1_score(yte, pred), 4)))
    if terbaik is None or auc > terbaik[0]:
        terbaik = (auc, model, nama, prob, pred)
pd.DataFrame(hasil).to_csv(os.path.join(OUT, "metrik_klasifikasi.csv"), index=False)
auc, model, nama_model, prob, pred = terbaik
skaler_simpan = scaler if nama_model == "Regresi Logistik" else None
joblib.dump({"model": model, "fitur": FITUR, "scaler": skaler_simpan},
            os.path.join(MODELS, "model_risiko.joblib"))

cm = confusion_matrix(yte, pred)
fig, ax = plt.subplots(figsize=(5.6, 4.6))
ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=14,
                color="white" if cm[i, j] > cm.max() / 2 else "#1c2733")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Prediksi aman", "Prediksi gagal"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Aktual aman", "Aktual gagal"])
ax.set_title(f"Confusion matrix — {nama_model} (data uji)")
ax.grid(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "klasifikasi_confusion_matrix.png")); plt.close(fig)

print(pd.DataFrame(hasil).to_string(index=False))
print("Model tersimpan: models/model_risiko.joblib | model terbaik:", nama_model)
