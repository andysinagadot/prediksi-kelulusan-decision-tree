import os
import joblib
import pandas as pd
from sklearn.tree import export_text
import streamlit as st

st.set_page_config(
    page_title="Prediksi Kelulusan Mahasiswa (Sem 1–8)",
    page_icon="🎓",
    layout="wide",
)

FEATURE_COLS = [
    "ipk_sem1_8",
    "sks_lulus",
    "kehadiran_persen",
    "jumlah_mengulang",
]


@st.cache_resource
def load_model():
  return joblib.load("model_decision_tree_kelulusan.joblib")


try:
  model = load_model()
except FileNotFoundError:
  st.error(
      "File 'model_decision_tree_kelulusan.joblib' belum ditemukan. "
      "Jalankan 'py train_model.py' terlebih dahulu."
  )
  st.stop()

st.title("🎓 Prediksi Kelulusan Mahasiswa (Semester 1–8)")
st.write(
    "Aplikasi prediksi kelulusan berbasis algoritma **Decision Tree** dengan"
    " data akademik hingga **Semester 8**."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📝 Prediksi Satuan",
    "📂 Prediksi Massal (CSV)",
    "🌳 Aturan IF-THEN",
    "📊 Evaluasi & Grafik",
])

# --- TAB 1: PREDIKSI SATUAN ---
with tab1:
  with st.form("form_prediksi_satuan"):
    col1, col2 = st.columns(2)
    with col1:
      ipk = st.number_input(
          "IPK (Semester 1–8)",
          min_value=0.00,
          max_value=4.00,
          value=3.25,
          step=0.01,
          format="%.2f",
      )
      sks = st.number_input(
          "Total SKS Lulus",
          min_value=0,
          max_value=160,
          value=124,
          step=1,
      )
    with col2:
      kehadiran = st.slider(
          "Persentase Kehadiran (%)",
          min_value=0,
          max_value=100,
          value=85,
      )
      mengulang = st.number_input(
          "Jumlah Mata Kuliah Mengulang",
          min_value=0,
          max_value=15,
          value=0,
          step=1,
      )

    submit = st.form_submit_button("Prediksi Kelulusan", use_container_width=True)

  if submit:
    data_input = pd.DataFrame(
        [[ipk, sks, kehadiran, mengulang]], columns=FEATURE_COLS
    )
    pred = model.predict(data_input)[0]
    proba = model.predict_proba(data_input)[0]

    st.subheader("Hasil Prediksi:")
    if pred == 1:
      st.success(
          f"✅ **Lulus Tepat Waktu** (Keyakinan Model: {proba[1] * 100:.1f}%)"
      )
    else:
      st.warning(
          "⚠️ **Berisiko Terlambat Lulus** (Keyakinan Model:"
          f" {proba[0] * 100:.1f}%)"
      )

# --- TAB 2: PREDIKSI MASSAL VIA CSV (DENGAN PERSENTASE & NIM) ---
with tab2:
  st.subheader("Unggah File CSV Data Mahasiswa")
  st.caption(
      "Kolom wajib: `ipk_sem1_8`, `sks_lulus`, `kehadiran_persen`,"
      " `jumlah_mengulang`. Kolom `nim` sangat disarankan (jika tidak ada,"
      " sistem akan membuat nomor urut otomatis)."
  )

  uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

  if uploaded_file is not None:
    try:
      df_upload = pd.read_csv(uploaded_file)
    except Exception as e:
      st.error(f"Gagal membaca file CSV: {e}")
      st.stop()

    missing_cols = [c for c in FEATURE_COLS if c not in df_upload.columns]
    if missing_cols:
      st.error(f"Kolom berikut belum ada di file CSV: {', '.join(missing_cols)}")
      st.stop()

    df_clean = df_upload.copy()
    if "nim" not in df_clean.columns:
      df_clean.insert(
          0, "nim", [f"MHS-{i + 1:03d}" for i in range(len(df_clean))]
      )
    else:
      df_clean["nim"] = df_clean["nim"].astype(str)

    for col in FEATURE_COLS:
      df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    nan_count = int(df_clean[FEATURE_COLS].isna().sum().sum())
    if nan_count > 0:
      st.warning(
          f"Ditemukan {nan_count} nilai kosong atau format angka tidak valid"
          " pada kolom fitur."
      )
      hapus_nan = st.checkbox(
          "Hapus otomatis baris yang memiliki nilai kosong/tidak valid",
          value=True,
      )
      if hapus_nan:
        df_clean = df_clean.dropna(subset=FEATURE_COLS).copy()
      else:
        st.info("Silakan perbaiki data kosong pada file CSV sebelum memprediksi.")
        st.stop()

    if len(df_clean) == 0:
      st.error("Tidak ada baris data valid yang tersisa untuk diprediksi.")
      st.stop()

    preds = model.predict(df_clean[FEATURE_COLS])
    total_mhs = len(df_clean)
    jumlah_lulus = int((preds == 1).sum())
    jumlah_terlambat = int((preds == 0).sum())
    persen_lulus = (jumlah_lulus / total_mhs) * 100
    persen_terlambat = (jumlah_terlambat / total_mhs) * 100

    df_hasil = df_clean.copy()
    df_hasil["prediksi_kode"] = preds
    df_hasil["status_prediksi"] = [
        "Lulus Tepat Waktu" if p == 1 else "Berisiko Terlambat" for p in preds
    ]

    # Ringkasan Persentase & Jumlah
    st.markdown("### Ringkasan Hasil Prediksi")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Mahasiswa Diproses", f"{total_mhs} orang")
    m2.metric(
        "Persentase Lulus Tepat Waktu",
        f"{persen_lulus:.1f}%",
        f"{jumlah_lulus} mahasiswa",
    )
    m3.metric(
        "Persentase Berisiko Terlambat",
        f"{persen_terlambat:.1f}%",
        f"{jumlah_terlambat} mahasiswa",
    )

    # Susun urutan kolom agar 'nim' dan 'status_prediksi' terlihat jelas di depan
    other_cols = [
        c for c in df_hasil.columns if c not in ["nim", "status_prediksi"]
    ]
    kolom_tampil = ["nim", "status_prediksi"] + other_cols

    filter_status = st.selectbox(
        "Filter tampilan tabel:",
        ["Semua Mahasiswa", "Hanya Lulus Tepat Waktu", "Hanya Berisiko Terlambat"],
    )

    if filter_status == "Hanya Lulus Tepat Waktu":
      df_tampil = df_hasil[df_hasil["prediksi_kode"] == 1][kolom_tampil]
    elif filter_status == "Hanya Berisiko Terlambat":
      df_tampil = df_hasil[df_hasil["prediksi_kode"] == 0][kolom_tampil]
    else:
      df_tampil = df_hasil[kolom_tampil]

    st.dataframe(df_tampil, use_container_width=True, hide_index=True)

    csv_bytes = df_hasil[kolom_tampil].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Unduh Hasil Prediksi (.csv)",
        data=csv_bytes,
        file_name="hasil_prediksi_kelulusan_sem1_8.csv",
        mime="text/csv",
    )
# --- TAB 3: ATURAN IF-THEN ---
with tab3:
  st.subheader("Aturan Keputusan Pohon (IF-THEN)")
  st.write(
      "Berikut aturan logika percabangan yang dipelajari model Decision Tree:"
  )
  rules_text = export_text(model, feature_names=FEATURE_COLS)
  st.code(rules_text, language="text")

# --- TAB 4: EVALUASI & GRAFIK ---
with tab4:
  st.subheader("Visualisasi Evaluasi & Struktur Pohon")
  col_a, col_b = st.columns(2)
  with col_a:
    if os.path.exists("confusion_matrix.png"):
      st.image(
          "confusion_matrix.png",
          caption="Confusion Matrix",
          use_container_width=True,
      )
    else:
      st.info("Jalankan `py train_model.py` untuk membuat confusion_matrix.png")
  with col_b:
    if os.path.exists("feature_importance.png"):
      st.image(
          "feature_importance.png",
          caption="Feature Importance",
          use_container_width=True,
      )
    else:
      st.info(
          "Jalankan `py train_model.py` untuk membuat feature_importance.png"
      )

  st.divider()
  if os.path.exists("pohon_keputusan_kelulusan.png"):
    st.image(
        "pohon_keputusan_kelulusan.png",
        caption="Diagram Pohon Keputusan (Semester 1–8)",
        use_container_width=True,
    )