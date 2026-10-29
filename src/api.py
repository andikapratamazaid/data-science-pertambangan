# -*- coding: utf-8 -*-
"""
Modul 07 — ML API (FastAPI): menyajikan model regresi dan klasifikasi
sebagai layanan HTTP, plus ramalan harga komoditas.

Jalankan lokal:
    uvicorn src.api:app --reload        (dari folder project)
atau:
    docker compose up --build           (port 8000)

Endpoint:
    GET  /                      info layanan
    POST /prediksi/konsumsi-bbm prediksi konsumsi BBM harian per unit
    POST /prediksi/risiko-gagal probabilitas unit gagal dalam 7 hari
    GET  /ramalan/harga?komoditas=nikel|batubara   ramalan 2025 (modul 04)

Catatan kejujuran: model dilatih dari data operasional SIMULASI
(lihat README). API ini mendemonstrasikan pola penyajian model
(skor risiko perawatan, estimasi BBM), bukan model produksi.
"""
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(BASE, "models")
OUT = os.path.join(BASE, "output")

bbm = joblib.load(os.path.join(MODELS, "model_bbm.joblib"))
risiko = joblib.load(os.path.join(MODELS, "model_risiko.joblib"))
ramalan = pd.read_csv(os.path.join(OUT, "ramalan_harga.csv"))

app = FastAPI(title="API Data Science Pertambangan", version="1.0",
              description="Demo penyajian model ML untuk perusahaan tambang.")


class InputBBM(BaseModel):
    jam_operasi: float = Field(..., ge=0, le=24)
    muatan_ton: float = Field(..., ge=0)
    jarak_km: float = Field(..., ge=0)
    umur_tahun: int = Field(..., ge=0, le=40)
    getaran_mm_s: float = Field(..., ge=0)
    suhu_mesin_c: float = Field(..., ge=0)
    hari_sejak_servis: int = Field(..., ge=0)
    hujan: int = Field(..., ge=0, le=1)
    tipe_truck: int = Field(..., ge=0, le=1, description="1 = Haul Truck, 0 = Excavator")


class InputRisiko(BaseModel):
    getaran_mm_s: float = Field(..., ge=0)
    suhu_mesin_c: float = Field(..., ge=0)
    hari_sejak_servis: int = Field(..., ge=0)
    umur_tahun: int = Field(..., ge=0, le=40)
    jam_operasi: float = Field(..., ge=0, le=24)
    hujan: int = Field(..., ge=0, le=1)


@app.get("/")
def info():
    return {"layanan": "API Data Science Pertambangan",
            "endpoint": ["/prediksi/konsumsi-bbm", "/prediksi/risiko-gagal",
                         "/ramalan/harga"],
            "catatan": "Model dilatih dari data operasional simulasi (demo)."}


@app.post("/prediksi/konsumsi-bbm")
def prediksi_bbm(inp: InputBBM):
    X = pd.DataFrame([[getattr(inp, f) for f in bbm["fitur"]]], columns=bbm["fitur"])
    nilai = float(bbm["model"].predict(X)[0])
    return {"konsumsi_bbm_liter": round(max(nilai, 0.0), 1)}


@app.post("/prediksi/risiko-gagal")
def prediksi_risiko(inp: InputRisiko):
    X = pd.DataFrame([[getattr(inp, f) for f in risiko["fitur"]]], columns=risiko["fitur"])
    if risiko.get("scaler") is not None:
        X = risiko["scaler"].transform(X)
    prob = float(risiko["model"].predict_proba(X)[0][1])
    kategori = "TINGGI" if prob >= 0.5 else ("SEDANG" if prob >= 0.25 else "RENDAH")
    saran = {"TINGGI": "Jadwalkan inspeksi sebelum unit dioperasikan lagi.",
             "SEDANG": "Pantau getaran dan suhu; rencanakan servis dalam minggu ini.",
             "RENDAH": "Operasi normal; servis mengikuti jadwal rutin."}[kategori]
    return {"probabilitas_gagal_7_hari": round(prob, 3),
            "kategori_risiko": kategori, "saran": saran}


@app.get("/ramalan/harga")
def ramalan_harga(komoditas: str = Query(..., pattern="^(nikel|batubara)$")):
    sub = ramalan[ramalan["komoditas"] == komoditas]
    if sub.empty:
        raise HTTPException(404, "Ramalan tidak ditemukan")
    return {"komoditas": komoditas, "model": sub["model"].iloc[0],
            "ramalan_2025": sub[["tanggal", "ramalan", "bawah_80", "atas_80"]]
            .to_dict(orient="records")}
