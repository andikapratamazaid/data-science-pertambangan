# -*- coding: utf-8 -*-
"""
Pengambilan Data Asli — Project 10 Data Science Pertambangan

1. World Bank Commodity Markets "Pink Sheet": harga komoditas bulanan
   resmi (batubara Australia, nikel, emas, tembaga, dll).
2. Yahoo Finance: harga saham bulanan emiten tambang IDX
   (ANTM, INCO, PTBA, MDKA), 2019-2026.

Hasil mentah disimpan ke data/mentah/. Jalankan siapkan_data.py sesudahnya
untuk mengubahnya menjadi CSV olahan.
"""
import json
import os
import time
import urllib.request
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MENTAH = os.path.join(BASE, "data", "mentah")
os.makedirs(MENTAH, exist_ok=True)


def unduh(url, tujuan):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    with open(tujuan, "wb") as f:
        f.write(data)
    print("tersimpan:", os.path.basename(tujuan), len(data), "byte")


unduh("https://thedocs.worldbank.org/en/doc/5d903e848db1d1b83e0ec8f744e55570-0350012021"
      "/related/CMO-Historical-Data-Monthly.xlsx",
      os.path.join(MENTAH, "pink_sheet_worldbank.xlsx"))

p1 = int(datetime(2019, 1, 1).timestamp())
p2 = int(datetime(2026, 10, 29).timestamp())
for tk in ["ANTM.JK", "INCO.JK", "PTBA.JK", "MDKA.JK"]:
    unduh(f"https://query1.finance.yahoo.com/v8/finance/chart/{tk}"
          f"?period1={p1}&period2={p2}&interval=1mo",
          os.path.join(MENTAH, f"yahoo_{tk}.json"))
    time.sleep(1)
print("Selesai.")
