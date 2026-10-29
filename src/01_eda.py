# -*- coding: utf-8 -*-
"""
Modul 01 — Exploratory Data Analysis (EDA)

Menjawab pertanyaan eksplorasi awal seorang data scientist di perusahaan
tambang:
1. Bagaimana perjalanan harga komoditas utama (data asli World Bank)?
2. Bagaimana kinerja saham emiten tambang IDX (data asli Yahoo Finance)?
3. Faktor apa yang berhubungan dengan konsumsi BBM dan kerusakan unit
   pada data operasional (data simulasi, berlabel di README)?

Output: grafik di output/eda_*.png dan output/ringkasan_eda.md.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLAH = os.path.join(BASE, "data", "olahan")
OUT = os.path.join(BASE, "output")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 130})

kom = pd.read_csv(os.path.join(OLAH, "harga_komoditas.csv"), parse_dates=["tanggal"])
saham = pd.read_csv(os.path.join(OLAH, "harga_saham_tambang.csv"), parse_dates=["tanggal"])
tel = pd.read_csv(os.path.join(OLAH, "telemetri_armada_simulasi.csv"), parse_dates=["tanggal"])
prod = pd.read_csv(os.path.join(OLAH, "produksi_harian_simulasi.csv"), parse_dates=["tanggal"])

# Grafik 1: harga komoditas
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
axes[0].plot(kom["tanggal"], kom["nikel_usd_mt"], color="#2e8b57", lw=2, label="Nikel (US$/mt)")
axes[0].plot(kom["tanggal"], kom["tembaga_usd_mt"], color="#7a7a7a", lw=1.6, label="Tembaga (US$/mt)")
axes[0].legend(); axes[0].set_title("Nikel & tembaga")
axes[1].plot(kom["tanggal"], kom["batubara_usd_mt"], color="#c0501f", lw=2)
axes[1].set_title("Batubara Australia (US$/mt)")
fig.suptitle("Harga komoditas tambang 2015–2024 (World Bank Pink Sheet)", fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "eda_1_komoditas.png")); plt.close(fig)

# Grafik 2: saham terindeks
fig, ax = plt.subplots(figsize=(9.5, 4.4))
idx = saham.set_index("tanggal")[["ANTM", "INCO", "PTBA", "MDKA"]]
idx = idx / idx.iloc[0] * 100
for c in idx.columns:
    ax.plot(idx.index, idx[c], lw=1.8, label=c)
ax.legend(ncol=4); ax.set_ylabel("Indeks (awal 2019 = 100)")
ax.set_title("Kinerja saham emiten tambang IDX (adjusted close, Yahoo Finance)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "eda_2_saham.png")); plt.close(fig)

# Grafik 3: korelasi telemetri
num = tel[["jam_operasi", "muatan_ton", "jarak_km", "konsumsi_bbm_liter", "suhu_mesin_c",
           "getaran_mm_s", "hari_sejak_servis", "umur_tahun", "gagal_dalam_7_hari"]]
kor = num.corr(numeric_only=True)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
kb = kor["konsumsi_bbm_liter"].drop("konsumsi_bbm_liter").sort_values()
axes[0].barh(kb.index, kb.values, color="#1f6fb2")
axes[0].set_title("Korelasi dengan konsumsi BBM")
kg = kor["gagal_dalam_7_hari"].drop("gagal_dalam_7_hari").sort_values()
axes[1].barh(kg.index, kg.values, color="#c0501f")
axes[1].set_title("Korelasi dengan label gagal 7 hari")
fig.suptitle("Hubungan antar-variabel telemetri armada (data simulasi)", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "eda_3_korelasi.png")); plt.close(fig)

# Grafik 4: produksi bulanan dan efek hujan
prod["bulan"] = prod["tanggal"].dt.to_period("M")
bul = prod.groupby("bulan")["produksi_batubara_ton"].sum() / 1e6
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
axes[0].bar(range(len(bul)), bul.values, color="#1f6fb2")
axes[0].set_xticks(range(len(bul))); axes[0].set_xticklabels([str(b) for b in bul.index], rotation=45, fontsize=8)
axes[0].set_ylabel("Juta ton"); axes[0].set_title("Produksi per bulan (simulasi)")
efek = prod.groupby("hujan")["produksi_batubara_ton"].mean()
axes[1].bar(["Tidak hujan", "Hujan"], efek.values, color=["#2e8b57", "#c0501f"])
axes[1].set_ylabel("Rata-rata ton/hari"); axes[1].set_title("Produksi hari hujan vs tidak")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "eda_4_produksi.png")); plt.close(fig)

nik_awal, nik_akhir = kom["nikel_usd_mt"].iloc[0], kom["nikel_usd_mt"].iloc[-1]
nik_puncak = kom.loc[kom["nikel_usd_mt"].idxmax()]
bb_awal, bb_akhir = kom["batubara_usd_mt"].iloc[0], kom["batubara_usd_mt"].iloc[-1]
bb_puncak = kom.loc[kom["batubara_usd_mt"].idxmax()]
kinerja = ((idx.iloc[-1] / 100 - 1) * 100).round(0)
turun_hujan = (1 - efek[1] / efek[0]) * 100

md = f"""# Ringkasan EDA — Project 10 Data Science Pertambangan

## Data asli: pasar
- **Nikel** bergerak dari US${nik_awal:,.0f}/mt (Jan 2015) ke US${nik_akhir:,.0f}/mt
  (Des 2024), dengan puncak US${nik_puncak['nikel_usd_mt']:,.0f}/mt pada
  {nik_puncak['tanggal']:%B %Y}, masa lonjakan harga logam baterai.
- **Batubara** dari US${bb_awal:,.0f}/mt ke US${bb_akhir:,.0f}/mt, puncak
  US${bb_puncak['batubara_usd_mt']:,.0f}/mt pada {bb_puncak['tanggal']:%B %Y},
  krisis energi global 2022.
- Kinerja saham sejak awal 2019 (indeks 100): ANTM {kinerja['ANTM']:+.0f}%,
  INCO {kinerja['INCO']:+.0f}%, PTBA {kinerja['PTBA']:+.0f}%, MDKA {kinerja['MDKA']:+.0f}%.

## Data simulasi: operasional (bukan data perusahaan asli)
- Korelasi terkuat dengan konsumsi BBM adalah jam operasi dan jarak tempuh,
  sesuai rancangan simulasi; model regresi di modul 02 mengukurnya secara
  formal.
- Label gagal berkorelasi positif dengan getaran, suhu mesin, dan hari
  sejak servis — tiga sinyal yang dipakai model klasifikasi modul 03.
- Produksi hari hujan rata-rata **{turun_hujan:.0f}% lebih rendah** dari
  hari tidak hujan pada data simulasi.
"""
with open(os.path.join(OUT, "ringkasan_eda.md"), "w", encoding="utf-8") as f:
    f.write(md)
print("EDA selesai | nikel 2015->2024: %.0f -> %.0f | produksi hujan turun %.0f%%" % (nik_awal, nik_akhir, turun_hujan))
