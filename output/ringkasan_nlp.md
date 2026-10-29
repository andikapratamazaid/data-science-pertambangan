# Ringkasan NLP — Sentimen Headline Berita Tambang (Okt 2026)

- Sampel: **20 headline asli** media Indonesia, dikumpulkan
  29 Oktober 2026.
- Distribusi sentimen leksikon: Negatif 7, Positif 7, Netral 6.
- Kata tersering: nikel (10x), produksi (8x), harga (6x), batu (6x), bara (6x).
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
