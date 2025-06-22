import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import mne
from mne.preprocessing import ICA
from mne_icalabel.gui import label_ica_components
from mne_icalabel import label_components

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# Carica i dati
sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(
    sample_data_folder, "MEG", "sample", "sample_audvis_raw.fif"
)
raw = mne.io.read_raw_fif(sample_data_raw_file)
raw.crop(tmax=60.0).pick_types(eeg=True, stim=True, eog=True)
raw.load_data()

# ICA
filt_raw = raw.copy().filter(l_freq=1.0, h_freq=100.0)
filt_raw = filt_raw.set_eeg_reference("average")
ica = ICA(n_components=15, max_iter="auto", method="infomax", random_state=97,
          fit_params=dict(extended=True))
ica.fit(filt_raw)

# Etichettatura manuale via GUI
gui = label_ica_components(raw, ica)

manual_labels = ica.labels_  # labels_ è un dizionario: {label_type: [component indices]}
input("Premi INVIO dopo aver chiuso tutte le finestre per continuare...")

print(manual_labels)


ic_labels = label_components(filt_raw, ica, method="iclabel")  # automatico
auto_labels = ic_labels["labels"]       # es. ['brain', 'muscle artifact', ...]
print(auto_labels)

n_components = len(auto_labels)
manual_labels_full = ['unlabeled'] * n_components
for label_type, indices in ica.labels_.items():
    for idx in indices:
        manual_labels_full[idx] = label_type

# Mappa le etichette automatiche su uno schema coerente
label_mapping = {
    'brain': 'brain',
    'muscle artifact': 'muscle',
    'eye blink': 'eog',
    'heart beat': 'ecg',
    'line noise': 'line_noise',
    'channel noise': 'ch_noise',
    'other': 'other',
}
mapped_auto_labels = [label_mapping.get(lbl, 'other') for lbl in auto_labels]

# Prepara dataset supervisionato (solo componenti etichettati manualmente)
X = []
y = []
for i in range(ica.n_components_):
    if manual_labels_full[i] != 'unlabeled':
        X.append(ica.mixing_matrix_[:, i])  # feature: spatial pattern
        y.append(manual_labels_full[i])

X = np.array(X)
y = np.array(y)

# Codifica etichette
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Addestra classificatore SVM
clf = SVC(kernel='rbf', gamma='scale')
clf.fit(X, y_encoded)

# Predici su tutte le componenti
X_all = np.array([ica.mixing_matrix_[:, i] for i in range(ica.n_components_)])
y_pred_encoded = clf.predict(X_all)
y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)

# Trova le etichette presenti in almeno uno dei due insiemi
present_labels = sorted(list(set(mapped_auto_labels) | set(y_pred_labels)))

# Matrice di confusione
cm = confusion_matrix(mapped_auto_labels, y_pred_labels, labels=present_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=present_labels)
disp.plot()
plt.title("SVM (supervised) vs ICLabel (pretrained)")
plt.show()

# Report
print("Accuracy (SVM vs ICLabel):", accuracy_score(mapped_auto_labels, y_pred_labels))
print("Classification Report:\n", classification_report(mapped_auto_labels, y_pred_labels, labels=present_labels, target_names=present_labels))


'''

TUTTO IL CODICE DA QUI IN POI È PER LA STAMPA DEL GRAFICO CON LA 'classification_report'

'''

# Supponiamo che queste siano le tue etichette vere e predette
y_true = mapped_auto_labels
y_pred = y_pred_labels

# Ottieni il report come dizionario
report_dict = classification_report(y_true, y_pred, output_dict=True)

# Estrai nomi delle classi (escludi "accuracy", "macro avg", "weighted avg")
class_labels = [label for label in report_dict.keys()
                if label not in ("accuracy", "macro avg", "weighted avg")]

# Estrai metriche per ogni classe
precisions = [report_dict[label]["precision"] for label in class_labels]
recalls = [report_dict[label]["recall"] for label in class_labels]
f1_scores = [report_dict[label]["f1-score"] for label in class_labels]

# Imposta posizione delle barre
x = np.arange(len(class_labels))
width = 0.25

# Crea il grafico
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width, precisions, width, label='Precision')
ax.bar(x, recalls, width, label='Recall')
ax.bar(x + width, f1_scores, width, label='F1-score')

# Labeling
ax.set_ylabel('Score')
ax.set_title('Classification Report by Label')
ax.set_xticks(x)
ax.set_xticklabels(class_labels)
ax.set_ylim(0, 1.1)
ax.legend(loc='lower right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()