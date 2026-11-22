# Sistem Prediksi Kelulusan Mahasiswa (Semester 1–8)

Proyek machine learning berbasis algoritma **Decision Tree** untuk memprediksi status kelulusan mahasiswa (`Lulus Tepat Waktu` vs `Berisiko Terlambat`) berdasarkan data akademik hingga semester 8.

## Fitur Akademik (`X`)
- `ipk_sem1_8`: IPK mahasiswa semester 1–8
- `sks_lulus`: Total SKS yang telah lulus
- `kehadiran_persen`: Persentase kehadiran perkuliahan
- `jumlah_mengulang`: Jumlah mata kuliah yang diulang

## Target (`y`)
- `status_lulus`: `1` = Lulus Tepat Waktu, `0` = Berisiko Terlambat

## Cara Menjalankan
1. Instal dependensi:
   ```bash
   py -m pip install -r requirements.txts