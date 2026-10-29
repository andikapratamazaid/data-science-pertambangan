# -*- coding: utf-8 -*-
"""
Modul 05 — NLP: analisis sentimen headline berita tambang Indonesia.

Data: 20 headline ASLI media Indonesia (Oktober 2026, dikumpulkan dari
hasil pencarian berita; judul, media, tanggal, dan URL tersimpan di
data/olahan/headline_tambang.csv).

Metode: analisis sentimen berbasis leksikon Bahasa Indonesia yang
transparan (daftar kata positif/negatif dinyatakan terbuka di bawah),
plus penandaan topik per kata kunci dan frekuensi kata. Ini pendekatan
baseline NLP yang jujur: sederhana, dapat diaudit kata per kata, dan
keterbatasannya dinyatakan di ringkasan (sampel kecil, hanya judul).
"""
import os
import re
from collections import Counter
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLAH = os.path.join(BASE, "data", "olahan")
OUT = os.path.join(BASE, "output")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 130})

POSITIF = {"naik", "menguat", "berkilau", "berkah", "melampaui", "rekomendasikan",
           "rekomendasi", "buy", "tumbuh", "meningkat", "positif", "prospek",
           "tembus", "relaksasi", "laba", "berhasil", "stabil", "membaik",
           "melonjak", "terbaik", "keuntungan"}
NEGATIF = {"melemah", "turun", "dipangkas", "menekan", "koreksi", "menyempit",
           "menahan", "batasi", "diturunkan", "membatasi", "berhenti",
           "berhentinya", "guncangan", "surplus", "lemah", "tekanan", "risiko",
           "anjlok", "defisit", "pangkas", "dampak"}
STOP = {"dan", "di", "ke", "dari", "yang", "untuk", "pada", "rp", "us", "ton",
        "jadi", "ini", "itu", "akan", "atau", "dengan", "dalam", "per", "triliun",
        "jt", "rb", "the", "a", "an", "hingga", "menurut", "saat", "tapi", "masih"}
TOPIK = {"Nikel": ["nikel", "smelter"], "Batubara": ["batubara", "batu"],
         "Saham & Emiten": ["saham", "antm", "ptba", "inco", "emiten", "mind", "adaro", "vale", "antam"],
         "Kebijakan & RKAB": ["rkab", "pemerintah", "esdm", "bahlil", "kuota", "pnbp", "target"]}

df = pd.read_csv(os.path.join(OLAH, "headline_tambang.csv"))

def kata(judul):
    return re.findall(r"[a-z]+", judul.lower())

def skor_sentimen(tokens):
    pos = sum(t in POSITIF for t in tokens)
    neg = sum(t in NEGATIF for t in tokens)
    return pos - neg, pos, neg

hasil = []
semua_kata = Counter()
for _, r in df.iterrows():
    tokens = kata(r["judul"])
    semua_kata.update(t for t in tokens if t not in STOP and len(t) > 2)
    skor, pos, neg = skor_sentimen(tokens)
    label = "Positif" if skor > 0 else ("Negatif" if skor < 0 else "Netral")
    topik = [nama for nama, kunci in TOPIK.items()
             if any(k in r["judul"].lower() for k in kunci)]
    hasil.append(dict(judul=r["judul"], media=r["media"], tanggal=r["tanggal"],
                      skor_sentimen=skor, kata_positif=pos, kata_negatif=neg,
                      label_sentimen=label, topik="; ".join(topik)))
res = pd.DataFrame(hasil)
res.to_csv(os.path.join(OUT, "nlp_hasil_sentimen.csv"), index=False)

dist = res["label_sentimen"].value_counts()
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
warna = {"Positif": "#2e8b57", "Negatif": "#c0501f", "Netral": "#7a7a7a"}
axes[0].bar(dist.index, dist.values, color=[warna.get(x, "#1f6fb2") for x in dist.index])
axes[0].set_title("Distribusi sentimen 20 headline (leksikon)")
top = pd.DataFrame(semua_kata.most_common(10), columns=["kata", "frekuensi"]).sort_values("frekuensi")
axes[1].barh(top["kata"], top["frekuensi"], color="#1f6fb2")
axes[1].set_title("Kata tersering di headline (selain kata henti)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "nlp_sentimen.png")); plt.close(fig)

md = f"""# Ringkasan NLP — Sentimen Headline Berita Tambang (Okt 2026)

- Sampel: **{len(res)} headline asli** media Indonesia, dikumpulkan
  29 Oktober 2026.
- Distribusi sentimen leksikon: {", ".join(f"{k} {v}" for k, v in dist.items())}.
- Kata tersering: {", ".join(f"{k} ({v}x)" for k, v in semua_kata.most_common(5))}.
- Tema dominan sampel ini adalah kebijakan produksi (RKAB dan kuota)
  serta gerak harga: headline positif umumnya berisi kenaikan harga dan
  rekomendasi saham, headline negatif berisi pelemahan harga dan
  pemangkasan produksi.

## Keterbatasan
1. Sampel kecil (20 judul) dan hanya dari judul, bukan isi artikel;
   hasilnya potret sesaat, bukan indeks sentimen yang stabil.
2. Leksikon manual tidak menangkap ironi, konteks, dan kata bermakna
   ganda (mis. "turun" pada "produksi turun, PNBP naik" bisa berdampak
   positif bagi negara). Untuk produksi, metode ini adalah baseline
   sebelum model terlatih seperti IndoBERT.
"""
with open(os.path.join(OUT, "ringkasan_nlp.md"), "w", encoding="utf-8") as f:
    f.write(md)
print("Distribusi sentimen:", dict(dist))
print("Kata tersering:", semua_kata.most_common(5))
