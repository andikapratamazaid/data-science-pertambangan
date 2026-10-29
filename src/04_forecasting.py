# -*- coding: utf-8 -*-
"""
Modul 04 — Forecasting: meramalkan harga nikel dan batubara (data asli
World Bank Pink Sheet, bulanan 2015-2024).

Metode dibandingkan secara jujur dengan backtesting: latih 2015-2022,
uji 2023-2024, lawan Naive (harga bulan lalu), SES, Holt, dan
Holt-Winters (statsmodels). Model terbaik per komoditas dilatih ulang
dengan seluruh data lalu meramalkan 12 bulan ke depan (2025) beserta
intervalnya. Hasil ramalan disimpan ke output/ramalan_harga.csv dan
dipakai dashboard serta API.
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLAH = os.path.join(BASE, "data", "olahan")
OUT = os.path.join(BASE, "output")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 130})

kom = pd.read_csv(os.path.join(OLAH, "harga_komoditas.csv"), parse_dates=["tanggal"])
SERIES = {"nikel": "nikel_usd_mt", "batubara": "batubara_usd_mt"}

def metrik(aktual, ramal):
    err = aktual - ramal
    return dict(rmse=float(np.sqrt((err ** 2).mean())),
                mae=float(np.abs(err).mean()),
                mape=float(np.abs(err / aktual).mean() * 100))

def ramal_model(train, h, metode):
    if metode == "Naive":
        sigma = float(np.nanstd(np.diff(train.values)))
        return np.repeat(train.iloc[-1], h), sigma
    if metode == "SES":
        fit = ExponentialSmoothing(train, trend=None, seasonal=None).fit()
    elif metode == "Holt":
        fit = ExponentialSmoothing(train, trend="add", seasonal=None).fit()
    else:
        fit = ExponentialSmoothing(train, trend="add", seasonal="add",
                                   seasonal_periods=12).fit()
    fc = fit.forecast(h)
    resid = train - fit.fittedvalues
    sigma = float(np.nanstd(resid))
    return np.asarray(fc), sigma

baris_metrik, baris_ramalan = [], []
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, (nama, kolom) in zip(axes, SERIES.items()):
    s = kom.set_index("tanggal")[kolom].asfreq("MS")
    latih, uji = s[s.index <= "2022-12-01"], s[s.index > "2022-12-01"]
    skor = {}
    for metode in ["Naive", "SES", "Holt", "Holt-Winters"]:
        fc, _ = ramal_model(latih, len(uji), metode)
        m = metrik(uji.values, fc)
        skor[metode] = m["rmse"]
        baris_metrik.append(dict(komoditas=nama, model=metode,
                                 rmse=round(m["rmse"], 1), mae=round(m["mae"], 1),
                                 mape_persen=round(m["mape"], 1)))
    terbaik = min(skor, key=skor.get)
    fc, sigma = ramal_model(s, 12, terbaik)
    tgl_ramal = pd.date_range("2025-01-01", periods=12, freq="MS")
    for t, v in zip(tgl_ramal, fc):
        baris_ramalan.append(dict(komoditas=nama, tanggal=t.date(), model=terbaik,
                                  ramalan=round(float(v), 1),
                                  bawah_80=round(float(v - 1.28 * (sigma or 0)), 1),
                                  atas_80=round(float(v + 1.28 * (sigma or 0)), 1)))
    ax.plot(s.index, s.values, color="#1c2733", lw=1.8, label="Aktual 2015–2024")
    ax.plot(tgl_ramal, fc, color="#c0501f", lw=2, ls="--", label=f"Ramalan 2025 ({terbaik})")
    if sigma:
        ax.fill_between(tgl_ramal, fc - 1.28 * sigma, fc + 1.28 * sigma,
                        color="#c0501f", alpha=0.15, label="Interval ±80%")
    ax.legend(fontsize=9); ax.set_title(f"Harga {nama} (US$/mt)")
fig.suptitle("Forecasting harga komoditas — backtest 2023–2024, ramalan 2025", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "forecasting_harga.png")); plt.close(fig)

pd.DataFrame(baris_metrik).to_csv(os.path.join(OUT, "metrik_forecasting.csv"), index=False)
pd.DataFrame(baris_ramalan).to_csv(os.path.join(OUT, "ramalan_harga.csv"), index=False)
print(pd.DataFrame(baris_metrik).to_string(index=False))
