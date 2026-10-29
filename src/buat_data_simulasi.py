# -*- coding: utf-8 -*-
"""
Buat Data Simulasi — DATA SIMULASI, BUKAN DATA PERUSAHAAN ASLI.

Data operasional tambang (telemetri alat berat dan produksi harian) tidak
dipublikasikan oleh perusahaan tambang mana pun, jadi untuk modul regresi
dan klasifikasi project ini datanya DISIMULASIKAN dengan hubungan antar-
variabel yang realistis dan dinyatakan terbuka:

- konsumsi BBM naik dengan jam operasi, muatan, dan umur unit;
- risiko kerusakan (label gagal_dalam_7_hari) naik dengan getaran, suhu
  mesin, hari sejak servis, dan umur unit; setelah rusak, unit berhenti
  satu hari dan hari sejak servis kembali ke nol;
- produksi harian mengikuti jumlah unit yang tersedia, jam operasi, dan
  hujan.

File hasil:
- data/olahan/telemetri_armada_simulasi.csv (12 unit x 365 hari, 2025)
- data/olahan/produksi_harian_simulasi.csv (365 hari, 2025)

Ganti kedua file ini dengan data asli bila suatu saat tersedia; seluruh
script modul membaca dari file yang sama.
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLAH = os.path.join(BASE, "data", "olahan")
os.makedirs(OLAH, exist_ok=True)
rng = np.random.default_rng(42)

UNIT = ([{"unit": f"DT-{i:02d}", "tipe": "Haul Truck", "umur": int(rng.integers(1, 10))}
         for i in range(1, 9)] +
        [{"unit": f"EX-{i:02d}", "tipe": "Excavator", "umur": int(rng.integers(1, 10))}
         for i in range(1, 5)])
tanggal = pd.date_range("2025-01-01", "2025-12-31", freq="D")

baris_tel, baris_prod = [], []
servis = {u["unit"]: int(rng.integers(0, 30)) for u in UNIT}
down = {u["unit"]: False for u in UNIT}

def peluang_gagal(getaran, suhu, sejak_servis, umur):
    z = -6.2 + 0.55 * getaran + 0.045 * (suhu - 80) + 0.035 * sejak_servis + 0.16 * umur
    return 1 / (1 + np.exp(-z))

for tgl in tanggal:
    hujan = rng.random() < (0.42 if tgl.month in (11, 12, 1, 2, 3) else 0.15)
    akhir_pekan = tgl.dayofweek >= 5
    unit_jalan, jam_total = 0, 0.0
    for u in UNIT:
        nama = u["unit"]
        if down[nama]:
            jam = 0.0
            down[nama] = False
            servis[nama] = 0
        else:
            dasar = 13.5 if not akhir_pekan else 9.0
            jam = float(np.clip(rng.normal(dasar, 2.2), 0, 18))
            if hujan:
                jam *= 0.78
        servis[nama] += 1
        getaran = float(np.clip(rng.normal(3.6 + 0.28 * u["umur"] + 0.03 * servis[nama], 1.1), 1.2, 10.5))
        suhu = float(np.clip(rng.normal(79 + 0.85 * u["umur"] + 2.1 * getaran + 0.05 * servis[nama], 4.5), 68, 112))
        muatan = float(rng.normal(215, 28)) if u["tipe"] == "Haul Truck" else float(rng.normal(38, 6))
        jarak = float(jam * rng.normal(11.5, 1.6)) if u["tipe"] == "Haul Truck" else 0.0
        laju = 26.5 if u["tipe"] == "Haul Truck" else 19.0
        bbm = jam * laju * (0.82 + 0.20 * (muatan / 215 if u["tipe"] == "Haul Truck" else 1.0)) \
            * (1 + 0.012 * u["umur"]) * float(rng.normal(1.0, 0.06)) if jam > 0 else 0.0
        p = peluang_gagal(getaran, suhu, servis[nama], u["umur"]) if jam > 0 else 0.0
        gagal = bool(rng.random() < p)
        if gagal:
            down[nama] = True
        baris_tel.append(dict(tanggal=tgl, unit=nama, tipe=u["tipe"], umur_tahun=u["umur"],
                              jam_operasi=round(jam, 2), muatan_ton=round(muatan, 1),
                              jarak_km=round(jarak, 1), konsumsi_bbm_liter=round(max(bbm, 0), 1),
                              suhu_mesin_c=round(suhu, 1), getaran_mm_s=round(getaran, 2),
                              hari_sejak_servis=servis[nama], hujan=int(hujan),
                              gagal_dalam_7_hari=int(gagal)))
        unit_jalan += int(jam > 0)
        jam_total += jam
    produksi = max(0.0, unit_jalan * jam_total / max(unit_jalan, 1) * 305 * float(rng.normal(1.0, 0.07))
                   * (0.82 if hujan else 1.0))
    baris_prod.append(dict(tanggal=tgl, unit_beroperasi=unit_jalan,
                           total_jam_operasi=round(jam_total, 1), hujan=int(hujan),
                           produksi_batubara_ton=round(produksi, 0)))

tel = pd.DataFrame(baris_tel)
prod = pd.DataFrame(baris_prod)
tel.to_csv(os.path.join(OLAH, "telemetri_armada_simulasi.csv"), index=False)
prod.to_csv(os.path.join(OLAH, "produksi_harian_simulasi.csv"), index=False)
print("telemetri:", tel.shape, "| tingkat label gagal: %.1f%%" % (100 * tel["gagal_dalam_7_hari"].mean()))
print("produksi:", prod.shape, "| rata-rata ton/hari: %.0f" % prod["produksi_batubara_ton"].mean())
