r"""
.. _tuto-iclabel:

Repairing artifacts with ICA automatically using ICLabel Model
==============================================================

This tutorial covers automatically repairing signals using ICA with
the ICLabel model\ :footcite:`PionTonachini2019`, which originates in EEGLab.
For conceptual background on ICA, see :ref:`this scikit-learn tutorial
<sphx_glr_auto_examples_decomposition_plot_ica_blind_source_separation.py>`.
For a basic understanding of how to use ICA to remove artifacts, see `the
tutorial
<https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html>`_
in MNE-Python.

ICLabel is designed to classify ICs fitted with an extended infomax ICA
decomposition algorithm on EEG datasets referenced to a common average and
filtered between [1., 100.] Hz. It is possible to run ICLabel on datasets that
do not meet those specification, but the classification performance
might be negatively impacted. Moreover, the ICLabel paper did not study the
effects of these preprocessing steps.

.. note::
    This example involves running the ICA Infomax algorithm, which requires
    `scikit-learn`_ to be installed. Please install this optional dependency before
    running the example.
"""

# %%
# We begin as always by importing the necessary Python modules and loading some
# :ref:`example data <sample-dataset>`. Because ICA can be computationally
# intense, we'll also crop the data to 60 seconds; and to save ourselves from
# repeatedly typing ``mne.preprocessing`` we'll directly import a few functions
# and classes from that submodule.

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"


import mne
from mne.preprocessing import ICA

from mne_icalabel import label_components

from mne_icalabel.iclabel.features import get_iclabel_features 
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.colors import ListedColormap

from collections import defaultdict
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from matplotlib.cm import get_cmap



def plot_tsne_with_kmeans(tsne_data, features_scaled, reliable_mask, unreliable_mask, iclabel_names, title, n_clusters=7):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)

    # Mappa cluster -> etichetta ICLabel prevalente
    cluster_to_label = {}
    for cluster in range(n_clusters):
        idx = np.where(cluster_labels == cluster)[0]
        if len(idx) == 0:
            cluster_to_label[cluster] = "N/A"
            continue
        labels_in_cluster = iclabel_names[idx]
        # Trova etichetta più comune
        unique_labels, counts = np.unique(labels_in_cluster, return_counts=True)
        most_common_label = unique_labels[np.argmax(counts)]
        cluster_to_label[cluster] = most_common_label

    # Colori dei cluster (non legati a etichette ICLabel)
    cmap = get_cmap('tab10')
    cluster_colors = [cmap(i%10) for i in cluster_labels]

    plt.figure(figsize=(9, 7))
    plt.title(f"{title}: KMeans Clustering con etichette ICLabel")

    # Inaffidabili: marker 'X'
    plt.scatter(tsne_data[unreliable_mask, 0], tsne_data[unreliable_mask, 1],
                c=np.array(cluster_colors)[unreliable_mask], marker='X', label='Inaffidabile (<0.8)', linewidths=0.5)

    # Affidabili: marker 'o'
    plt.scatter(tsne_data[reliable_mask, 0], tsne_data[reliable_mask, 1],
                c=np.array(cluster_colors)[reliable_mask], marker='o', label='Affidabile (≥0.8)')

    # Legenda cluster (colori)
    cluster_handles = []
    for i in range(n_clusters):
        cluster_handles.append(
            plt.Line2D([0], [0], marker='o', color='w',
                                 label=f'Cluster {i}: {cluster_to_label[i]}',
                                 markerfacecolor=cmap(i % 10),
                                 markersize=10,
                                 markeredgecolor='black'))
    marker_handles = [
        plt.Line2D([0], [0], marker='o', linestyle='None', markerfacecolor='white', markeredgecolor='black', color='w', label='Affidabile (≥0.8)', markersize=10),
        plt.Line2D([0], [0], marker='X', linestyle='None', markerfacecolor='white', markeredgecolor='black', color='w', label='Inaffidabile (<0.8)', markersize=10)
    ]

    plt.legend(handles=cluster_handles + marker_handles, title="Legenda")
    plt.tight_layout()
    plt.show()



sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(
    sample_data_folder, "MEG", "sample", "sample_audvis_raw.fif"
)
raw = mne.io.read_raw_fif(sample_data_raw_file)

# Here we'll crop to 60 seconds and drop gradiometer channels for speed
raw.crop(tmax=60.0).pick(picks=["eeg", "stim", "eog"])
raw.load_data()

# %%
# .. note::
#     Before applying ICA (or any artifact repair strategy), be sure to observe
#     the artifacts in your data to make sure you choose the right repair tool.
#     Sometimes the right tool is no tool at all — if the artifacts are small
#     enough you may not even need to repair them to get good analysis results.
#     See :ref:`tut-artifact-overview` for guidance on detecting and
#     visualizing various types of artifact.
#
#
# Example: EOG and ECG artifact repair
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Visualizing the artifacts
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Let's begin by visualizing the artifacts that we want to repair. In this
# dataset they are big enough to see easily in the raw data:

# pick some channels that clearly show heartbeats and blinks
artifact_picks = mne.pick_channels_regexp(raw.ch_names, regexp=r"(MEG [12][45][123]1|EEG 00.)") #regexp=r"(EEG 00.)"
raw.plot(order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False, block=False)

# %%
# Filtering to remove slow drifts
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Before we run the ICA, an important step is filtering the data to remove
# low-frequency drifts, which can negatively affect the quality of the ICA fit.
# The slow drifts are problematic because they reduce the independence of the
# assumed-to-be-independent sources (e.g., during a slow upward drift, the
# neural, heartbeat, blink, and other muscular sources will all tend to have
# higher values), making it harder for the algorithm to find an accurate
# solution. A high-pass filter with 1 Hz cutoff frequency is recommended.
# However, because filtering is a linear operation, the ICA solution found from
# the filtered signal can be applied to the unfiltered signal (see
# :footcite:`WinklerEtAl2015` for more information), so we'll keep a copy of
# the unfiltered `~mne.io.Raw` object around so we can apply the ICA solution
# to it later.

filt_raw = raw.copy().filter(l_freq=1.0, h_freq=100.0)

# %%
# Fitting and plotting the ICA solution
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# .. sidebar:: Ignoring the time domain
#
#     The ICA algorithms implemented in MNE-Python find patterns across
#     channels, but ignore the time domain. This means you can compute ICA on
#     discontinuous `~mne.Epochs` or `~mne.Evoked` objects (not
#     just continuous `~mne.io.Raw` objects), or only use every Nth
#     sample by passing the ``decim`` parameter to ``ICA.fit()``.
#
#     .. note:: `~mne.Epochs` used for fitting ICA should not be
#               baseline-corrected. Because cleaning the data via ICA may
#               introduce DC offsets, we suggest to baseline correct your data
#               **after** cleaning (and not before), should you require
#               baseline correction.
#
# Now we're ready to set up and fit the ICA. Since we know (from observing our
# raw data) that the EOG and ECG artifacts are fairly strong, we would expect
# those artifacts to be captured in the first few dimensions of the PCA
# decomposition that happens before the ICA. Therefore, we probably don't need
# a huge number of components to do a good job of isolating our artifacts
# (though it is usually preferable to include more components for a more
# accurate solution). As a first guess, we'll run ICA with ``n_components=15``
# (use only the first 15 PCA components to compute the ICA decomposition) — a
# very small number given that our data has 59 good EEG channels, but with the
# advantage that it will run quickly and we will able to tell easily whether it
# worked or not (because we already know what the EOG / ECG artifacts should
# look like).
#
# ICA fitting is not deterministic (e.g., the components may get a sign
# flip on different runs, or may not always be returned in the same order), so
# we'll also specify a `random seed`_ so that we get identical results each
# time this tutorial is built by our web servers.

# %%
# Before fitting ICA, we will apply a common average referencing, to comply
# with the ICLabel requirements.

filt_raw = filt_raw.set_eeg_reference("average")

# %%
# We will use the 'extended infomax' method for fitting the ICA, to comply with
# the ICLabel requirements. ICLabel was not tested with other ICA decomposition
# algorithm, but its performance and accuracy should not be impacted by the
# algorithm.

ica = ICA(
    n_components=10,
    max_iter="auto",
    method="fastica",           #Sempre su richiesta del prof, ho cambiato da "infomax" a "fastica" togliendo anche l'ultimo parametro 
    #random_state=97,           #QUESTO È IL RANDOM SEED CHE HO TOLTO SOTTO CONSIGLIO DEL PROF CIARAMELLA 
    #fit_params=dict(extended=True),    # SE method="infomax" SI, SE method="fastica" NO
)
ica.fit(filt_raw)
ica

# %%
# Some optional parameters that we could have passed to the
# `~mne.preprocessing.ICA.fit` method include ``decim`` (to use only
# every Nth sample in computing the ICs, which can yield a considerable
# speed-up) and ``reject`` (for providing a rejection dictionary for maximum
# acceptable peak-to-peak amplitudes for each channel type, just like we used
# when creating epoched data in the :ref:`tut-overview` tutorial).
#
# Now we can examine the ICs to see what they captured.
# `~mne.preprocessing.ICA.plot_sources` will show the time series of the
# ICs. Note that in our call to `~mne.preprocessing.ICA.plot_sources` we
# can use the original, unfiltered `~mne.io.Raw` object:

raw.load_data()

# %%
# Here we can pretty clearly see that the first component (``ICA000``) captures
# the EOG signal quite well (for more info on visually identifying Independent
# Components, `this EEGLAB tutorial`_ is a good resource). We can also
# visualize the scalp field distribution of each component using
# `~mne.preprocessing.ICA.plot_components`. These are interpolated based
# on the values in the ICA mixing matrix:
#
# .. LINKS
#
# .. _`blind source separation`:
#    https://en.wikipedia.org/wiki/Signal_separation
# .. _`statistically independent`:
#    https://en.wikipedia.org/wiki/Independence_(probability_theory)
# .. _`scikit-learn`: https://scikit-learn.org
# .. _`random seed`: https://en.wikipedia.org/wiki/Random_seed
# .. _`regular expression`: https://www.regular-expressions.info/
# .. _`qrs`: https://en.wikipedia.org/wiki/QRS_complex
# .. _`this EEGLAB tutorial`: https://labeling.ucsd.edu/tutorial/labels




# blinks & muscle
#ica.plot_overlay(raw, exclude=[0], picks="eeg") 

# %%
# Selecting ICA components automatically
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Now that we've explored what components need to be removed, we can
# apply the automatic ICA component labeling algorithm, which will
# assign a probability value for each component being one of:
#
# - brain
# - muscle artifact
# - eye blink
# - heart beat
# - line noise
# - channel noise
# - other
#
# The output of the ICLabel ``label_components`` function produces
# predicted probability values for each of these classes in that order.
# See :footcite:`PionTonachini2019` for full details.

ic_labels = label_components(filt_raw, ica, method="iclabel")

# ICA0 was correctly identified as an eye blink, whereas ICA6 was
# classified as a muscle artifact.
print("\n\n")
for idx, (label, prob) in enumerate(zip(ic_labels["labels"], ic_labels["y_pred_proba"])):
    print(f"Component {idx}: label = {label},\t probability = {prob:.2f}")

ica.plot_sources(raw, show_scrollbars=False, show=True, block=False)

# sphinx_gallery_thumbnail_number = 1
#ica.plot_components()


# Stampa di tutte le label con prob <= 80%
count = 0
print("\n\n")
for idx, (label, prob) in enumerate(zip(ic_labels["labels"], ic_labels["y_pred_proba"])):
    if prob < 0.8:
        print(f"Component {idx}: label = {label}")
        count += 1

print(f"\nNUMBER OF ICA COMP WITH PROB < 0.8: {count}")

# Calcolo della percentuale di componenti affidabili (prob >= 0.8)
total_components = len(ic_labels["labels"])
reliable_components = sum(1 for prob in ic_labels["y_pred_proba"] if prob >= 0.8)
unreliable_components = total_components - reliable_components

model_reliability_percent = 100 * reliable_components / total_components

print(f"\n\nMODEL RELIABILITY: {model_reliability_percent:.2f}%")
print(f"RELIABLE COMPONENTS: {reliable_components}/{total_components}")
print(f"UNRELIABLE COMPONENTS: {unreliable_components}/{total_components}")


# Estrazione dati
labels = ic_labels["labels"]
probabilities = ic_labels["y_pred_proba"]
components = [f"Comp {i}" for i in range(len(labels))]
unique_labels = sorted(set(labels))

# Creazione matrice: righe = componenti, colonne = label, valori = probabilità se label matcha
data_matrix = []
for label, prob in zip(labels, probabilities):
    row = []
    for ul in unique_labels:
        row.append(prob if ul == label else 0.0)
    data_matrix.append(row)

df_heatmap = pd.DataFrame(data_matrix, index=components, columns=unique_labels)

# Plot della heatmap
plt.figure(figsize=(10, len(components) * 0.5))
sns.heatmap(df_heatmap, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Probability'})
plt.title("Assigned Label Probabilities per Component")
plt.xlabel("Labels")
plt.ylabel("Components")
plt.tight_layout()
plt.show()

# Filtra componenti con probabilità < 0.8
low_confidence = [(i, label, prob) for i, (label, prob) in enumerate(zip(labels, probabilities)) if prob < 0.8]
# Filtra e ordina per label
low_confidence = sorted(
    [(i, label, prob) for i, (label, prob) in enumerate(zip(labels, probabilities)) if prob < 0.8],
    key=lambda x: (x[1], x[2])
)
if low_confidence:
    comps, lbls, probs = zip(*low_confidence)

    # Colori per label
    label_colors = {l: c for l, c in zip(sorted(set(lbls)), sns.color_palette("husl", len(set(lbls))))}
    colors = [label_colors[l] for l in lbls]

    # Plot
    plt.figure(figsize=(10, len(comps) * 0.4))
    bars = plt.barh([f"Comp {c}" for c in comps], [p * 100 for p in probs], color=colors)
    plt.xlabel("Probability (%)")
    plt.title("ICA Components with Probability < 80%")

    # Legenda
    handles = [plt.Line2D([0], [0], color=clr, lw=4) for lbl, clr in label_colors.items()]
    plt.legend(handles, label_colors.keys(), title="Labels", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()
else:
    print("Nessuna componente con probabilità < 80%")


# Preparazione strutture dati
label_stats = defaultdict(lambda: {"total": 0, "reliable": 0, "uncertainty_sum": 0.0})

# Popolamento statistiche
for label, prob in zip(ic_labels["labels"], ic_labels["y_pred_proba"]):
    label_stats[label]["total"] += 1
    label_stats[label]["uncertainty_sum"] += (1 - prob)
    if prob >= 0.8:
        label_stats[label]["reliable"] += 1

# Creazione DataFrame per visualizzazione
stats_df = pd.DataFrame([
    {
        "Label": label,
        "Total": values["total"],
        "Reliable": values["reliable"],
        "Unreliable": values["total"] - values["reliable"],
        "Reliability (%)": 100 * values["reliable"] / values["total"],
        "Avg Uncertainty (%)": 100 * values["uncertainty_sum"] / values["total"]
    }
    for label, values in label_stats.items() if values["total"] > 0 and values["reliable"] <= values["total"]
])

stats_df.sort_values("Reliability (%)", ascending=True, inplace=True)
print(stats_df)

# Plot percentuali di affidabilità
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=stats_df, x="Reliability (%)", y="Label", palette="coolwarm", orient="h")
plt.title("Affidabilità del modello per ciascuna label (<80% filter)")
plt.xlabel("Reliability (%)")
plt.ylabel("Label")
for p in ax.patches:
    width = p.get_width()
    label_y = p.get_y() + p.get_height() / 2
    ax.text(width / 2, label_y, f"{width:.1f}%", va='center', ha='left', color='black', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.show()

#plot percentuali di incertezza media
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=stats_df, x="Avg Uncertainty (%)", y="Label", palette="magma", orient="h")
plt.title("Incertezza media del modello per ciascuna label")
plt.xlabel("Incertezza media (%)")
plt.ylabel("Label")
for p in ax.patches:
    width = p.get_width()
    label_y = p.get_y() + p.get_height() / 2
    ax.text(width / 2, label_y, f"{width:.1f}%", va='center', ha='left', color='white', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.show()


# Grafico a torta dell'affidabilità generale
plt.figure(figsize=(6, 6))
plt.pie(
    [reliable_components, unreliable_components],
    labels=["Affidabile", "Inaffidabile"],
    colors=["green", "red"],
    autopct="%1.1f%%",
    startangle=140,
    labeldistance=0.5,      # sposta le etichette all'interno
    textprops={'color': "white", 'fontsize': 12, 'weight': 'bold'}
)
plt.title("Affidabilità globale del modello sulle componenti ICA")
plt.axis("equal")
plt.tight_layout()
plt.show()



# Stampa di tutte le IC classificate come 'brain' che però hanno prob < 80%
'''for idx, (label, prob) in enumerate(zip(ic_labels["labels"], ic_labels["y_pred_proba"])):
    if label == "brain" and prob < 0.8:
        print(f"Component {idx} è brain ma con bassa confidenza: {prob:.2f}")'''

'''
                        ALGORITMO T-SNE

class sklearn.manifold.TSNE(
            n_components=2, *, 
            perplexity=30.0, 
            early_exaggeration=12.0, 
            learning_rate='auto', 
            max_iter=None, 
            n_iter_without_progress=300, 
            min_grad_norm=1e-07, 
            metric='euclidean', 
            metric_params=None, 
            init='pca', 
            verbose=0, 
            random_state=None, 
            method='barnes_hut', 
            angle=0.5, 
            n_jobs=None, 
            n_iter='deprecated'
)
                        

The perplexity is related to the number of nearest neighbors that is 
used in other manifold learning algorithms. Larger datasets usually 
require a larger perplexity. Consider selecting a value between 5 and 
50. Different values can result in significantly different results. 
The perplexity must be less than the number of samples.

'''
features = get_iclabel_features(filt_raw, ica) 
topo, psd, autocorr = features  # unpack

#mappatura delle label a num corrispondenti per utilizzare colori consistenti tra i grafici
'''label_names = np.array(ic_labels["labels"])
unique_labels = np.unique(label_names)
label_to_int = {name: idx for idx, name in enumerate(unique_labels)}
label_ints = np.array([label_to_int[l] for l in label_names])'''

label_names = np.array(ic_labels["labels"])
known_labels = ['brain', 'muscle artifact', 'eye blink', 'heart beat', 'line noise', 'channel noise', 'other']
colors = ['tab:blue', 'tab:pink', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:gray']
label_to_color = {label: color for label, color in zip(known_labels, colors)}
label_colors = np.array([label_to_color.get(label, 'black') for label in label_names])

label_enc = LabelEncoder()
numeric_labels = label_enc.fit_transform(label_names)

scaler = StandardScaler()

reliable_mask = ic_labels["y_pred_proba"] >= 0.8
unreliable_mask = ~reliable_mask

# 1. TOPOGRAPHY
topo = topo.squeeze(axis=2)  # (32, 32, 50)
topo_flat = topo.reshape(32 * 32, 50).T  # (50, 1024)

X_topo_scaled = scaler.fit_transform(topo_flat)

tsne = TSNE(n_components=2, random_state=42, perplexity=3, learning_rate=50, early_exaggeration=16, init='pca')
topo_tsne = tsne.fit_transform(X_topo_scaled)

plt.figure(figsize=(8, 6))
plt.title("t-SNE Topography Features")

# Componenti con label inaffidabile: marker 'X'
plt.scatter(topo_tsne[unreliable_mask, 0], topo_tsne[unreliable_mask, 1],
            c=label_colors[unreliable_mask], marker='X', label='Inaffidabile (<0.8)', linewidths=0.5)

# Componenti con label affidabile: marker 'o'
plt.scatter(topo_tsne[reliable_mask, 0], topo_tsne[reliable_mask, 1],
            c=label_colors[reliable_mask], marker='o', label='Affidabile (≥0.8)')

# Legenda con le classi ICLabel (colori)
class_handles = []
for label in known_labels:
    class_handles.append(
        plt.Line2D([0], [0],
                   marker='o',
                   color='w',
                   label=label,
                   markerfacecolor=label_to_color[label],
                   markersize=10,
                   markeredgecolor='black')
    )

# Marker legend (forma)
marker_handles = [
    plt.Line2D([0], [0], marker='o', linestyle='None', markerfacecolor='white', markeredgecolor='black', color='w', label='Affidabile (≥0.8)', markersize=10),
    plt.Line2D([0], [0], marker='X', linestyle='None', markerfacecolor='white', markeredgecolor='black', color='w', label='Inaffidabile (<0.8)', markersize=10)
]

# Aggiungi le legende
plt.legend(handles=class_handles + marker_handles, title="Legenda")
plt.tight_layout()
plt.show()

plot_tsne_with_kmeans(topo_tsne, X_topo_scaled, reliable_mask, unreliable_mask, label_names, "Topography")


# 2. PSD
psd = psd.squeeze(axis=2)  # (1, 100, 50)
psd_flat = psd.reshape(100, 50).T  # (50, 100)

X_psd_scaled = scaler.fit_transform(psd_flat)

psd_tsne = TSNE(n_components=2, random_state=42, perplexity=3, learning_rate=50, early_exaggeration=16, init='pca').fit_transform(X_psd_scaled)

plt.figure(figsize=(8, 6))
plt.title("t-SNE PSD Features")

plt.scatter(psd_tsne[unreliable_mask, 0], psd_tsne[unreliable_mask, 1],
            c=label_colors[unreliable_mask], marker='X', label='Inaffidabile (<0.8)', linewidths=0.5)
plt.scatter(psd_tsne[reliable_mask, 0], psd_tsne[reliable_mask, 1],
            c=label_colors[reliable_mask], marker='o', label='Affidabile (≥0.8)')

class_handles = [
    plt.Line2D([0], [0], marker='o', color='w', label=label,
               markerfacecolor=label_to_color[label], markersize=10,
               markeredgecolor='black') for label in known_labels
]
marker_handles = [
    plt.Line2D([0], [0], marker='o', linestyle='None', markerfacecolor='white', markeredgecolor='black', color='w', label='Affidabile (≥0.8)', markersize=10),
    plt.Line2D([0], [0], marker='X', linestyle='None', markerfacecolor='white', markeredgecolor='black', color='w', label='Inaffidabile (<0.8)', markersize=10)
]

plt.legend(handles=class_handles + marker_handles, title="Legenda")
plt.tight_layout()
plt.show()

plot_tsne_with_kmeans(psd_tsne, X_psd_scaled, reliable_mask, unreliable_mask, label_names, "PSD")


# 3. AUTOCORRELATION
autocorr = autocorr.squeeze(axis=2)  # (1, 100, 50)
autocorr_flat = autocorr.reshape(100, 50).T  # (50, 100)

X_auto_scaled = scaler.fit_transform(autocorr_flat)

autocorr_tsne = TSNE(n_components=2, random_state=42, perplexity=3, learning_rate=50, early_exaggeration=16, init='pca').fit_transform(X_auto_scaled)

plt.figure(figsize=(8, 6))
plt.title("t-SNE Autocorrelation Features")

plt.scatter(autocorr_tsne[unreliable_mask, 0], autocorr_tsne[unreliable_mask, 1],
            c=label_colors[unreliable_mask], marker='X', label='Inaffidabile (<0.8)', linewidths=0.5)
plt.scatter(autocorr_tsne[reliable_mask, 0], autocorr_tsne[reliable_mask, 1],
            c=label_colors[reliable_mask], marker='o', label='Affidabile (≥0.8)')

class_handles = [
    plt.Line2D([0], [0], marker='o', color='w', label=label,
               markerfacecolor=label_to_color[label], markersize=10,
               markeredgecolor='black') for label in known_labels
]
marker_handles = [
    plt.Line2D([0], [0], marker='o', linestyle='None', markerfacecolor='white', markeredgecolor='black', color='w', label='Affidabile (≥0.8)', markersize=10),
    plt.Line2D([0], [0], marker='X', linestyle='None', markerfacecolor='white', markeredgecolor='black', color='w', label='Inaffidabile (<0.8)', markersize=10)
]

plt.legend(handles=class_handles + marker_handles, title="Legenda")
plt.tight_layout()
plt.show()

plot_tsne_with_kmeans(autocorr_tsne, X_auto_scaled, reliable_mask, unreliable_mask, label_names, "Autocorrelation")


# CONCATENAZIONE DELLE 3 FEATURES (TOPO, PSD, AUTOCORR)
#all_features = np.concatenate([topo_flat, psd_flat, autocorr_flat], axis=1)  # (50, 1224)
all_features = np.concatenate([X_topo_scaled, X_psd_scaled, X_auto_scaled], axis=1)

tsne_combined = TSNE(n_components=2, random_state=42, perplexity=3, learning_rate=50, early_exaggeration=16, init='pca').fit_transform(all_features)

plt.figure(figsize=(8, 6))
plt.title("t-SNE Combined Features (Topography + PSD + Autocorr)")

plt.scatter(tsne_combined[unreliable_mask, 0], tsne_combined[unreliable_mask, 1],
            c=label_colors[unreliable_mask], marker='X', label='Inaffidabile (<0.8)', linewidths=0.5)
plt.scatter(tsne_combined[reliable_mask, 0], tsne_combined[reliable_mask, 1],
            c=label_colors[reliable_mask], marker='o', label='Affidabile (≥0.8)')

class_handles = [
    plt.Line2D([0], [0], marker='o', color='w', label=label,
               markerfacecolor=label_to_color[label], markersize=10,
               markeredgecolor='black') for label in known_labels
]
marker_handles = [
    plt.Line2D([0], [0], marker='o', linestyle='None', markerfacecolor='white', markeredgecolor='black', color='w', label='Affidabile (≥0.8)', markersize=10),
    plt.Line2D([0], [0], marker='X', linestyle='None', markerfacecolor='white', markeredgecolor='black', color='w', label='Inaffidabile (<0.8)', markersize=10)
]

plt.legend(handles=class_handles + marker_handles, title="Legenda")
plt.tight_layout()
plt.show()

plot_tsne_with_kmeans(tsne_combined, all_features, reliable_mask, unreliable_mask, label_names, "Combined Features")



print("\n--- METRICHE DI QUALITÀ t-SNE ---")

# 1. Trustworthiness
#
#Valuta quanto bene sono preservati i vicini nel passaggio dallo spazio 
#originale allo spazio ridotto (cioè: i vicini in 1224-D restano vicini 
#anche in 2D?).
#
#Scala: [0, 1] — più vicino a 1 = migliore.
'''valori bassi (<0.7), potrebbero voler dire che t-SNE sta 
"appiattendo" troppa struttura locale — e quindi può spiegare un plot 
visivamente "strano".
ATTENZIONE all'ultimo paramentro n_neighbors=5 forse SBAGLIATO... '''
tw_topo = trustworthiness(X_topo_scaled, topo_tsne, n_neighbors=3)
tw_psd = trustworthiness(X_psd_scaled, psd_tsne, n_neighbors=3)
tw_autocorr = trustworthiness(X_auto_scaled, autocorr_tsne, n_neighbors=3)
tw_combined = trustworthiness(all_features, tsne_combined, n_neighbors=3)

print(f"Trustworthiness Topo:      {tw_topo:.3f}")
print(f"Trustworthiness PSD:       {tw_psd:.3f}")
print(f"Trustworthiness Autocorr:  {tw_autocorr:.3f}")
print(f"Trustworthiness Combined:  {tw_combined:.3f}")
print("\n\n")

# 2. Separabilità dei cluster - usando le etichette ICLabel
sil_score1 = silhouette_score(topo_tsne, numeric_labels)
sil_score2 = silhouette_score(psd_tsne, numeric_labels)
sil_score3 = silhouette_score(autocorr_tsne, numeric_labels)
sil_score4 = silhouette_score(tsne_combined, numeric_labels)
'''
- Valori > 0.5 = cluster ben definiti
- Valori < 0.2 = scarsa separabilità'''

print(f"Silhouette Score topo:          {sil_score1:.3f}")
print(f"Silhouette Score psd:          {sil_score2:.3f}")
print(f"Silhouette Score autocorr:          {sil_score3:.3f}")
print(f"Silhouette Score combined:          {sil_score4:.3f}")
print("\n\n")

'''Mentre con Homogeneity, Completeness, Adjusted Rand Index (se compari ICA/ICLabel)
Si può vedere quanto i cluster t-SNE corrispondono alle classi ICLabel:'''
kmeans_labels1 = KMeans(n_clusters=len(known_labels), random_state=42).fit(topo_tsne).labels_
homog1 = homogeneity_score(numeric_labels, kmeans_labels1)
compl1 = completeness_score(numeric_labels, kmeans_labels1)
ari1 = adjusted_rand_score(numeric_labels, kmeans_labels1)

kmeans_labels2 = KMeans(n_clusters=len(known_labels), random_state=42).fit(psd_tsne).labels_
homog2 = homogeneity_score(numeric_labels, kmeans_labels2)
compl2 = completeness_score(numeric_labels, kmeans_labels2)
ari2 = adjusted_rand_score(numeric_labels, kmeans_labels2)

kmeans_labels3 = KMeans(n_clusters=len(known_labels), random_state=42).fit(autocorr_tsne).labels_
homog3 = homogeneity_score(numeric_labels, kmeans_labels3)
compl3 = completeness_score(numeric_labels, kmeans_labels3)
ari3 = adjusted_rand_score(numeric_labels, kmeans_labels3)

kmeans_labels4 = KMeans(n_clusters=len(known_labels), random_state=42).fit(tsne_combined).labels_
homog4 = homogeneity_score(numeric_labels, kmeans_labels4)
compl4 = completeness_score(numeric_labels, kmeans_labels4)
ari4 = adjusted_rand_score(numeric_labels, kmeans_labels4)

print(f"Homogeneity Score topo:         {homog1:.3f}")
print(f"Homogeneity Score psd:         {homog2:.3f}")
print(f"Homogeneity Score autocorr:         {homog3:.3f}")
print(f"Homogeneity Score combined:         {homog4:.3f}")
print("\n\n")
print(f"Completeness Score topo:        {compl1:.3f}")
print(f"Completeness Score psd:        {compl2:.3f}")
print(f"Completeness Score autocorr:        {compl3:.3f}")
print(f"Completeness Score combined:        {compl4:.3f}")
print("\n\n")
print(f"Adjusted Rand Index (ARI) topo: {ari1:.3f}")
print(f"Adjusted Rand Index (ARI) psd: {ari2:.3f}")
print(f"Adjusted Rand Index (ARI) autocorr: {ari3:.3f}")
print(f"Adjusted Rand Index (ARI) combined: {ari4:.3f}")
print("\n\n")

# 3. Correlazione tra distanza nel t-SNE e confidenza ICLabel
'''Se il coefficiente è negativo, suggerisce che componenti molto 
"affidabili" sono vicini nel t-SNE space → buon segno.'''
distances = squareform(pdist(topo_tsne))
mean_dist = distances.mean(axis=1)
reliability = ic_labels["y_pred_proba"]
corr, _ = spearmanr(mean_dist, reliability)
print(f"Correlazione distanza / affidabilità TOPO: {corr:.3f}")

distances1 = squareform(pdist(psd_tsne))
mean_dist1 = distances1.mean(axis=1)
reliability1 = ic_labels["y_pred_proba"]
corr1, _ = spearmanr(mean_dist1, reliability1)
print(f"Correlazione distanza / affidabilità PSD: {corr1:1.3f}")

distances2 = squareform(pdist(autocorr_tsne))
mean_dist2 = distances2.mean(axis=1)
reliability2 = ic_labels["y_pred_proba"]
corr2, _ = spearmanr(mean_dist2, reliability2)
print(f"Correlazione distanza / affidabilità AUTOCORR: {corr2:.3f}")

distances3 = squareform(pdist(tsne_combined))
mean_dist3 = distances3.mean(axis=1)
reliability3 = ic_labels["y_pred_proba"]
corr3, _ = spearmanr(mean_dist3, reliability3)
print(f"Correlazione distanza / affidabilità COMBINED: {corr3:.3f}")


# Dizionario delle metriche
trust_percent = ["Trustworthiness topo","Trustworthiness psd","Trustworthiness autocorr",
                 "Trustworthiness combined"]
percent_values = [tw_topo * 100,tw_psd * 100,tw_autocorr  * 100,tw_combined * 100]

# Plot a barre
plt.figure(figsize=(10, 7))
bars = plt.bar(trust_percent, percent_values, color='skyblue')
plt.ylim(0, 100)
plt.ylabel("Percentuale (%)")
plt.title("Trustworthiness t-SNE")
plt.xticks(rotation=45, ha='right')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f"{yval:.1f}%", ha='center', va='bottom')
plt.tight_layout()
plt.show()

# Dizionario delle metriche
homog_percent = ["Homogeneity topo","Homogeneity psd","Homogeneity autocorr",
                 "Homogeneity combined"]
percent_values1 = [homog1 * 100,homog2 * 100,homog3  * 100,homog4 * 100]

# Plot a barre
plt.figure(figsize=(10, 7))
bars = plt.bar(homog_percent, percent_values1, color='skyblue')
plt.ylim(0, 100)
plt.ylabel("Percentuale (%)")
plt.title("Homogeneity t-SNE")
plt.xticks(rotation=45, ha='right')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f"{yval:.1f}%", ha='center', va='bottom')
plt.tight_layout()
plt.show()

# Dizionario delle metriche
compl_percent = ["Completeness topo","Completeness psd","Completeness autocorr",
                 "Completeness combined"]
percent_values2 = [compl1 * 100,compl2 * 100,compl3  * 100,compl4 * 100]

# Plot a barre
plt.figure(figsize=(10, 7))
bars = plt.bar(compl_percent, percent_values2, color='skyblue')
plt.ylim(0, 100)
plt.ylabel("Percentuale (%)")
plt.title("Completeness t-SNE")
plt.xticks(rotation=45, ha='right')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f"{yval:.1f}%", ha='center', va='bottom')
plt.tight_layout()
plt.show()


#Dizionario metriche
sil_score_metrics = ["Sil score topo", "Sil score psd", "Sil score autocorr",
                     "Sil score combined"]
sil_score_values = [sil_score1, sil_score2, sil_score3, sil_score4]

plt.figure(figsize=(7, 5))
bars = plt.bar(sil_score_metrics, sil_score_values, color='lightcoral')
plt.ylim(-1, 1)
plt.ylabel("Valore")
plt.title("Silhouette score t-SNE")
plt.xticks(rotation=45, ha='right')
plt.axhline(0, color='gray', linewidth=0.8)

# Etichette sopra le barre
for bar in bars:
    yval = bar.get_height()
    offset = 0.05 if yval >= 0 else -0.1
    plt.text(bar.get_x() + bar.get_width() / 2, yval + offset, f"{yval:.2f}", ha='center', va='bottom')

plt.tight_layout()
plt.show()

#Dizionario metriche
ari_metrics = ["ARI topo", "ARI psd", "ARI autocorr ",
               "ARI combined"]
ari_values = [ari1, ari2, ari3, ari4]

plt.figure(figsize=(7, 5))
bars = plt.bar(ari_metrics, ari_values, color='lightcoral')
plt.ylim(-1, 1)
plt.ylabel("Valore")
plt.title("Adjusted Rand Index t-SNE")
plt.xticks(rotation=45, ha='right')
plt.axhline(0, color='gray', linewidth=0.8)

# Etichette sopra le barre
for bar in bars:
    yval = bar.get_height()
    offset = 0.05 if yval >= 0 else -0.1
    plt.text(bar.get_x() + bar.get_width() / 2, yval + offset, f"{yval:.2f}", ha='center', va='bottom')

plt.tight_layout()
plt.show()

#Dizionario metriche
corr_metrics = ["Corr topo", "Corr psd",
                "Corr autocorr","Corr combined"]
corr_values = [corr, corr1, corr2, corr3]

plt.figure(figsize=(7, 5))
bars = plt.bar(corr_metrics, corr_values, color='lightcoral')
plt.ylim(-1, 1)
plt.ylabel("Valore")
plt.title("Correlazione distanza/affidabilità t-SNE")
plt.xticks(rotation=45, ha='right')
plt.axhline(0, color='gray', linewidth=0.8)

# Etichette sopra le barre
for bar in bars:
    yval = bar.get_height()
    offset = 0.05 if yval >= 0 else -0.1
    plt.text(bar.get_x() + bar.get_width() / 2, yval + offset, f"{yval:.2f}", ha='center', va='bottom')

plt.tight_layout()
plt.show()


# %%
# We can also plot some diagnostics of IC using
# `~mne.preprocessing.ICA.plot_properties`:
#ica.plot_properties(raw, picks=[0], verbose=False) #prova a cambiare picks=[0,1,2,3, ....]
#plt.show()


# %%
# Extract Labels and Reconstruct Raw Data
# ---------------------------------------
#
# We can extract the labels of each component and exclude
# non-brain classified components, keeping 'brain' and 'other'.
# "Other" is a catch-all that for non-classifiable components.
# We will stay on the side of caution and assume we cannot blindly remove these.

labels = ic_labels["labels"]
exclude_idx = [
    idx for idx, label in enumerate(labels) if label not in ["brain", "other"]
]
print("\n\n")
print(f"Excluding these ICA components: {exclude_idx}")

# %%
# Now that the exclusions have been set, we can reconstruct the sensor signals
# with artifacts removed using the `~mne.preprocessing.ICA.apply` method
# (remember, we're applying the ICA solution from the *filtered* data to the
# original *unfiltered* signal). Plotting the original raw data alongside the
# reconstructed data shows that the heartbeat and blink artifacts are repaired.

# ica.apply() changes the Raw object in-place, so let's make a copy first:
reconst_raw = raw.copy()
ica.apply(reconst_raw, exclude=exclude_idx)

raw.plot(order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False, title="raw_data")
reconst_raw.plot(
    order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False, title="recons_data"
)

input("Premi INVIO dopo aver chiuso tutte le finestre per continuare...")


#plt.show()

del reconst_raw

# %%
# References
# ^^^^^^^^^^
# .. footbibliography::

