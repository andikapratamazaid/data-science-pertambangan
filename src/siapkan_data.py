# -*- coding: utf-8 -*-
"""
Siapkan Data — mengubah data mentah menjadi CSV olahan di data/olahan/:
- harga_komoditas.csv : batubara (US$/mt), nikel (US$/mt), emas (US$/troy oz),
  tembaga (US$/mt), bulanan 2015-2024, dari World Bank Pink Sheet.
- harga_saham_tambang.csv : harga penutupan bulanan ANTM, INCO, PTBA, MDKA
  (adjusted close, Yahoo Finance), 2019-2026.
"""
import json
import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MENTAH = os.path.join(BASE, "data", "mentah")
OLAH = os.path.join(BASE, "data", "olahan")
os.makedirs(OLAH, exist_ok=True)

# ---- Pink Sheet ----
raw = pd.read_excel(os.path.join(MENTAH, "pink_sheet_worldbank.xlsx"),
                    sheet_name="Monthly Prices", header=None)
header = raw.iloc[4]
data = raw.iloc[6:].copy()
data.columns = ["periode"] + list(header[1:])
data = data.replace("…", pd.NA)
kolom = {"Coal, Australian": "batubara_usd_mt", "Nickel": "nikel_usd_mt",
         "Gold": "emas_usd_oz", "Copper": "tembaga_usd_mt"}
sub = data[["periode"] + list(kolom.keys())].rename(columns=kolom)
sub["tanggal"] = pd.to_datetime(sub["periode"], format="%YM%m")
sub = sub.drop(columns="periode")
for c in kolom.values():
    sub[c] = pd.to_numeric(sub[c], errors="coerce")
sub = sub[(sub["tanggal"] >= "2015-01-01") & (sub["tanggal"] <= "2024-12-01")]
sub = sub.sort_values("tanggal").reset_index(drop=True)
sub.to_csv(os.path.join(OLAH, "harga_komoditas.csv"), index=False)
print("harga_komoditas:", sub.shape, "| rentang:", sub["tanggal"].min(), "->", sub["tanggal"].max())

# ---- Saham tambang IDX ----
frames = []
for tk in ["ANTM.JK", "INCO.JK", "PTBA.JK", "MDKA.JK"]:
    d = json.load(open(os.path.join(MENTAH, f"yahoo_{tk}.json")))
    r = d["chart"]["result"][0]
    ts = r["timestamp"]
    close = r["indicators"]["adjclose"][0]["adjclose"]
    f = pd.DataFrame({"tanggal": pd.to_datetime(ts, unit="s").normalize(),
                      "emiten": tk.replace(".JK", ""), "harga_tutup": close})
    frames.append(f)
saham = pd.concat(frames).pivot_table(index="tanggal", columns="emiten",
                                      values="harga_tutup").reset_index()
saham.to_csv(os.path.join(OLAH, "harga_saham_tambang.csv"), index=False)
print("harga_saham_tambang:", saham.shape, "| rentang:", saham["tanggal"].min(), "->", saham["tanggal"].max())
