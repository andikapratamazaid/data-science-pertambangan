# Ringkasan EDA — Project 10 Data Science Pertambangan

## Data asli: pasar
- **Nikel** bergerak dari US$14,849/mt (Jan 2015) ke US$15,445/mt
  (Des 2024), dengan puncak US$33,924/mt pada
  March 2022, masa lonjakan harga logam baterai.
- **Batubara** dari US$61/mt ke US$130/mt, puncak
  US$431/mt pada September 2022,
  krisis energi global 2022.
- Kinerja saham sejak awal 2019 (indeks 100): ANTM +285%,
  INCO -1%, PTBA +124%, MDKA +207%.

## Data simulasi: operasional (bukan data perusahaan asli)
- Korelasi terkuat dengan konsumsi BBM adalah jam operasi dan jarak tempuh,
  sesuai rancangan simulasi; model regresi di modul 02 mengukurnya secara
  formal.
- Label gagal berkorelasi positif dengan getaran, suhu mesin, dan hari
  sejak servis — tiga sinyal yang dipakai model klasifikasi modul 03.
- Produksi hari hujan rata-rata **38% lebih rendah** dari
  hari tidak hujan pada data simulasi.
