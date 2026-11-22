# Sistem Prediksi Kelulusan Mahasiswa (Semester 1–8) Menggunakan Algoritma Decision Tree

Aplikasi prediksi kelulusan mahasiswa berbasis **Machine Learning** menggunakan algoritma **Decision Tree** (`DecisionTreeClassifier` dengan `class_weight="balanced"`) untuk data akademik Semester 1–8, dilengkapi antarmuka web interaktif berbasis **Streamlit**.

## Fitur Utama
- **Prediksi Satuan (Single Prediction):** Memprediksi status kelulusan satu mahasiswa berdasarkan input data akademik Semester 1–8.
- **Prediksi Massal / Batch CSV:** Mengunggah file CSV (seperti data kohort mahasiswa lengkap dengan `nim`) untuk memprediksi banyak mahasiswa sekaligus, menampilkan persentase kelulusan, dan mengunduh/menampilkan hasilnya.
- **Interpretasi Model & Aturan IF-THEN:** Menampilkan struktur pohon keputusan (`pohon_keputusan_kelulusan.png`) serta ekstraksi aturan *IF-THEN* agar alur keputusan model mudah dipahami secara akademis.
- **Evaluasi Model:** Menyajikan metrik evaluasi beserta visualisasi **Confusion Matrix** (`confusion_matrix.png`) dan **Feature Importance** (`feature_importance.png`).
- **Validasi Input & Penanganan Ketidakseimbangan Kelas:** Menggunakan `class_weight="balanced"` saat pelatihan dan validasi kolom/data saat unggah file CSV.

## Teknologi yang Digunakan
- **Bahasa Pemrograman:** Python
- **Pengolahan Data & Komputasi:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (`DecisionTreeClassifier`)
- **Penyimpanan Model:** Joblib
- **Visualisasi:** Matplotlib
- **Antarmuka Web:** Streamlit

## Struktur Folder Proyek
```text
proyek_kelulusan_ai/
├── app_kelulusan.py                      # Aplikasi web interaktif Streamlit
├── train_model.py                        # Skrip pelatihan model Decision Tree & pembuatan grafik evaluasi
├── dataset_kelulusan_mahasiswa_sem1_8.csv# Dataset utama pelatihan model (Semester 1–8)
├── model_decision_tree_kelulusan.joblib  # Model Decision Tree terlatih
├── confusion_matrix.png                  # Visualisasi Confusion Matrix
├── feature_importance.png                # Visualisasi tingkat kepentingan fitur
├── pohon_keputusan_kelulusan.png         # Visualisasi struktur pohon keputusan
├── requirements.txt                      # Daftar dependensi library Python
└── README.md                             # Dokumentasi proyek