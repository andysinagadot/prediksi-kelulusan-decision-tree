import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

# 1. Baca dataset semester 1-8
df = pd.read_csv("dataset_kelulusan_mahasiswa_sem1_8.csv")

FEATURE_COLS = [
    "ipk_sem1_8",
    "sks_lulus",
    "kehadiran_persen",
    "jumlah_mengulang",
]
X = df[FEATURE_COLS]
y = df["status_lulus"]

# 2. Bagi data latih (80%) dan data uji (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Latih model Decision Tree dengan class_weight="balanced"
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    class_weight="balanced",
    random_state=42,
)
model.fit(X_train, y_train)

# 4. Evaluasi model
y_pred = model.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)
print(f"Akurasi Model (Balanced): {akurasi * 100:.2f}%\n")
print(
    classification_report(
        y_test, y_pred, target_names=["Terlambat (0)", "Tepat Waktu (1)"]
    )
)

# 5. Cetak aturan IF-THEN
print("Aturan Pohon Keputusan:")
print(export_text(model, feature_names=FEATURE_COLS))

# 6. Simpan diagram pohon keputusan
plt.figure(figsize=(16, 8))
plot_tree(
    model,
    feature_names=FEATURE_COLS,
    class_names=["Terlambat", "Tepat Waktu"],
    filled=True,
    rounded=True,
    fontsize=9,
)
plt.title("Decision Tree Prediksi Kelulusan Mahasiswa (Semester 1–8)")
plt.tight_layout()
plt.savefig("pohon_keputusan_kelulusan.png", dpi=300)
plt.close()

# 7. Simpan gambar Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["Terlambat (0)", "Tepat Waktu (1)"]
)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix Model Decision Tree")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()

# 8. Simpan gambar Feature Importance
importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(
    ascending=True
)
plt.figure(figsize=(7, 4.5))
importances.plot(kind="barh", color="#2563EB")
plt.xlabel("Tingkat Kepentingan Fitur (Importance)")
plt.title("Feature Importance Prediksi Kelulusan")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.close()

# 9. Simpan model terlatih
joblib.dump(model, "model_decision_tree_kelulusan.joblib")
print(
    "Selesai! Model, pohon_keputusan_kelulusan.png, confusion_matrix.png, dan"
    " feature_importance.png berhasil disimpan."
)