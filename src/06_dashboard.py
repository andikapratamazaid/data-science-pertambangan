# -*- coding: utf-8 -*-
"""
Modul 06 — Dashboard eksekutif: merangkum seluruh modul ke satu file
HTML mandiri (output/dashboard_eksekutif.html) yang bisa dibuka di
peramban mana pun dan dibagikan ke manajemen.

Berisi: harga komoditas terakhir + ramalan (data asli, modul 04), skor
model (modul 02-04), unit dengan risiko gagal tertinggi menurut model
klasifikasi pada data telemetri terakhir (simulasi, modul 03), dan
sentimen berita (data asli, modul 05). Setiap bagian berlabel sumber
datanya: Asli atau Simulasi.
"""
import base64
import os
import joblib
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLAH = os.path.join(BASE, "data", "olahan")
OUT = os.path.join(BASE, "output")
MODELS = os.path.join(BASE, "models")


def img64(nama):
    with open(os.path.join(OUT, nama), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


kom = pd.read_csv(os.path.join(OLAH, "harga_komoditas.csv"), parse_dates=["tanggal"])
terakhir = kom.iloc[-1]
ram = pd.read_csv(os.path.join(OUT, "ramalan_harga.csv"))
reg = pd.read_csv(os.path.join(OUT, "metrik_regresi.csv"))
kla = pd.read_csv(os.path.join(OUT, "metrik_klasifikasi.csv"))
fcs = pd.read_csv(os.path.join(OUT, "metrik_forecasting.csv"))
nlp = pd.read_csv(os.path.join(OUT, "nlp_hasil_sentimen.csv"))

tel = pd.read_csv(os.path.join(OLAH, "telemetri_armada_simulasi.csv"), parse_dates=["tanggal"])
risiko = joblib.load(os.path.join(MODELS, "model_risiko.joblib"))
hari_terakhir = tel[tel["tanggal"] == tel["tanggal"].max()].copy()
X = hari_terakhir[risiko["fitur"]]
if risiko.get("scaler") is not None:
    X = risiko["scaler"].transform(X)
hari_terakhir["probabilitas"] = risiko["model"].predict_proba(X)[:, 1]
hari_terakhir = hari_terakhir.sort_values("probabilitas", ascending=False)

def tabel(df, kolom=None):
    d = df if kolom is None else df[kolom]
    head = "".join(f"<th>{c}</th>" for c in d.columns)
    body = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>"
                   for row in d.itertuples(index=False))
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

baris_risiko = "".join(
    f"<tr><td>{r.unit}</td><td>{r.tipe}</td><td>{r.probabilitas:.0%}</td>"
    f"<td>{r.getaran_mm_s}</td><td>{r.suhu_mesin_c}</td><td>{int(r.hari_sejak_servis)}</td></tr>"
    for r in hari_terakhir.head(5).itertuples())
nik_2025 = ram[ram["komoditas"] == "nikel"]["ramalan"].mean()
bb_2025 = ram[ram["komoditas"] == "batubara"]["ramalan"].mean()
sentimen = nlp["label_sentimen"].value_counts().to_dict()

html = f"""<!DOCTYPE html>
<html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dashboard Eksekutif — Data Science Pertambangan</title>
<style>
body{{font-family:system-ui,'Segoe UI',Arial,sans-serif;background:#f2f4f7;color:#1c2733;margin:0}}
.bungkus{{max-width:1060px;margin:0 auto;padding:24px 16px 48px}}
h1{{font-size:24px;margin:4px 0}} h2{{font-size:17px;margin:26px 0 8px}}
.sub{{color:#5a6a7a;font-size:14px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-top:16px}}
.kartu{{background:#fff;border:1px solid #dbe2ea;border-radius:12px;padding:14px}}
.kartu .angka{{font-size:22px;font-weight:700;margin:4px 0}}
.kartu .label{{font-size:12px;color:#5a6a7a;text-transform:uppercase;letter-spacing:.4px}}
.kartu .sumber{{font-size:11px;margin-top:6px}}
.asli{{color:#2e8b57;font-weight:700}} .sim{{color:#c0501f;font-weight:700}}
img.grafik{{width:100%;border:1px solid #dbe2ea;border-radius:12px;background:#fff}}
.dua{{display:grid;grid-template-columns:1fr 1fr;gap:14px}} @media(max-width:760px){{.dua{{grid-template-columns:1fr}}}}
table{{width:100%;border-collapse:collapse;background:#fff;font-size:13.5px}}
th,td{{border:1px solid #dbe2ea;padding:7px 9px;text-align:left}}
th{{background:#eef2f6}} .catatan{{font-size:13px;color:#5a6a7a}}
</style></head><body><div class="bungkus">
<h1>Dashboard Eksekutif — Data Science Pertambangan</h1>
<p class="sub">Ringkasan satu halaman dari project data science perusahaan tambang:
pasar (data asli), operasional (data simulasi berlabel), dan sentimen berita (data asli).</p>

<div class="grid">
<div class="kartu"><div class="label">Harga nikel Des 2024</div>
<div class="angka">US${terakhir['nikel_usd_mt']:,.0f}/mt</div>
<div class="sumber"><span class="asli">DATA ASLI</span> · World Bank</div></div>
<div class="kartu"><div class="label">Harga batubara Des 2024</div>
<div class="angka">US${terakhir['batubara_usd_mt']:,.0f}/mt</div>
<div class="sumber"><span class="asli">DATA ASLI</span> · World Bank</div></div>
<div class="kartu"><div class="label">Ramalan rata-rata nikel 2025</div>
<div class="angka">US${nik_2025:,.0f}/mt</div>
<div class="sumber"><span class="asli">DATA ASLI</span> · backtest modul 04</div></div>
<div class="kartu"><div class="label">Ramalan rata-rata batubara 2025</div>
<div class="angka">US${bb_2025:,.0f}/mt</div>
<div class="sumber"><span class="asli">DATA ASLI</span> · backtest modul 04</div></div>
<div class="kartu"><div class="label">Akurasi arah risiko (AUC)</div>
<div class="angka">{kla['auc'].max():.3f}</div>
<div class="sumber"><span class="sim">SIMULASI</span> · model klasifikasi</div></div>
<div class="kartu"><div class="label">Sentimen berita Okt 2026</div>
<div class="angka">{sentimen.get('Positif', 0)}+ / {sentimen.get('Negatif', 0)}- / {sentimen.get('Netral', 0)} netral</div>
<div class="sumber"><span class="asli">DATA ASLI</span> · 20 headline media</div></div>
</div>

<h2>Pasar: harga komoditas dan ramalan <span class="asli">(Data Asli)</span></h2>
<img class="grafik" src="{img64('forecasting_harga.png')}">
<p class="catatan">Catatan jujur: pada backtest 2023–2024 model Naive (harga bulan lalu)
mengalahkan SES, Holt, dan Holt-Winters pada kedua komoditas. Harga komoditas tambang
sangat sulit diramal; dashboard ini menampilkan model terbaik menurut backtest,
bukan model yang terlihat paling meyakinkan.</p>
{tabel(fcs)}

<h2>Operasional: skor model <span class="sim">(Data Simulasi)</span></h2>
<div class="dua"><div><p class="catatan"><b>Regresi konsumsi BBM</b> (liter per unit-hari)</p>{tabel(reg)}</div>
<div><p class="catatan"><b>Klasifikasi risiko gagal 7 hari</b></p>{tabel(kla)}</div></div>
<div class="dua" style="margin-top:14px"><img class="grafik" src="{img64('regresi_fitur.png')}">
<img class="grafik" src="{img64('klasifikasi_confusion_matrix.png')}"></div>

<h2>Unit risiko tertinggi — hari terakhir data <span class="sim">(Simulasi)</span></h2>
<table><thead><tr><th>Unit</th><th>Tipe</th><th>Probabilitas gagal 7 hari</th>
<th>Getaran (mm/s)</th><th>Suhu (C)</th><th>Hari sejak servis</th></tr></thead>
<tbody>{baris_risiko}</tbody></table>

<h2>Sentimen berita tambang <span class="asli">(Data Asli)</span></h2>
<img class="grafik" src="{img64('nlp_sentimen.png')}">
<p class="catatan">Baseline leksikon pada 20 headline media Indonesia, Oktober 2026.
Sampel kecil dan hanya judul: potret sesaat, bukan indeks sentimen produksi.</p>

<h2>Harga komoditas 2015–2024 <span class="asli">(Data Asli)</span></h2>
<img class="grafik" src="{img64('eda_1_komoditas.png')}">

<p class="catatan" style="margin-top:28px">Dibangun sebagai project portofolio data science:
EDA, regresi, klasifikasi, forecasting, NLP, dashboard, API, dan Docker dalam satu repositori.
Data pasar dan berita asli dari sumber publik; data operasional adalah simulasi berlabel
karena perusahaan tambang tidak mempublikasikan telemetri armadanya.</p>
</div></body></html>"""

tujuan = os.path.join(OUT, "dashboard_eksekutif.html")
with open(tujuan, "w", encoding="utf-8") as f:
    f.write(html)
print("Dashboard tersimpan:", tujuan, "|", len(html), "karakter")
