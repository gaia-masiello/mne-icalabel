"""
.. _tuto-gui-label-components:

Labeling ICA components with a GUI
==================================

This tutorial covers how to label ICA components with a GUI.

.. note:: Similar to ``mne-qt-browser``, we require the users
          to install a specific version of ``Qt``. Our installation
          ``pip install mne-icalabel[gui]`` will not install any
          specific ``Qt`` version. Therefore, one can install ``Qt5``
          through either ``PyQt5`` or ``PySide2`` or a more modern
          ``Qt6`` through either ``PyQt6`` or ``PySide6`` depending
          on their system. The users should install this separately
          to use the GUI functionality. See:
          https://www.riverbankcomputing.com/software/pyqt/ for more info
          on installing.

.. warning:: The GUI is still in active development, and may contain
             bugs, or changes without deprecation in future versions.
"""

# %%

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"  # <-- Forza l'uso di X11 anche sotto Wayland


import mne
from mne.preprocessing import ICA

from mne_icalabel.gui import label_ica_components
from mne_icalabel import label_components

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt


# %%
# Load in some sample data

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(
    sample_data_folder, "MEG", "sample", "sample_audvis_raw.fif"
)
raw = mne.io.read_raw_fif(sample_data_raw_file)

# Here we'll crop to 60 seconds and drop gradiometer channels for speed
raw.crop(tmax=60.0).pick_types(eeg=True, stim=True, eog=True)
raw.load_data()

# %%
# Preprocess and run ICA on the data
# ----------------------------------
# Before labeling components with the GUI, one needs to filter the data
# and then fit the ICA instance. Afterwards, one can run the GUI using the
# ``Raw`` data object and the fitted ``ICA`` instance.

# high-pass filter the data and then perform ICA
filt_raw = raw.copy().filter(l_freq=1.0, h_freq=100.0)
filt_raw = filt_raw.set_eeg_reference("average")


ica = ICA(n_components=15, max_iter="auto",method="infomax", random_state=97, fit_params=dict(extended=True),)
ica.fit(filt_raw)
ica

# %%
# Annotate ICA components with the GUI
# ------------------------------------
# The GUI will modify the ICA instance in place, and add the
# labels of each component to the ``labels_`` attribute. The
# GUI will show features of the ICA components similar to the
# :func:`mne.viz.plot_ica_properties` function. It will also provide an
# interface to label each ICA component into one of seven categories:
#
# - Brain
# - Muscle
# - Eye
# - Heart
# - Line Noise
# - Channel Noise
# - Other
#
# For more information on annotating ICA components, we suggest
# reading through the tutorial from ``ICLabel``
# (https://labeling.ucsd.edu/tutorial/about).

gui = label_ica_components(raw, ica)

# Dopo che chiudi la GUI:
manual_labels = ica.labels_  # labels_ è un dizionario: {label_type: [component indices]}
input("Premi INVIO dopo aver chiuso tutte le finestre per continuare...")

# The `ica` object is modified to contain the component labels
# after closing the GUI and can now be saved
# gui.close()  # typically you close when done

# Now, we can take a look at the components, which were modified in-place
# for the ICA instance.
print(manual_labels)


ic_labels = label_components(filt_raw, ica, method="iclabel")  # automatico
auto_labels = ic_labels["labels"]       # es. ['brain', 'muscle artifact', ...]
print(auto_labels)

n_components = len(auto_labels)
manual_labels = ['unlabeled'] * n_components

for label_type, indices in ica.labels_.items():
    for idx in indices:
        manual_labels[idx] = label_type

# %%
# Save the labeled components
# ---------------------------
# After the GUI labels, save the components using the ``write_components_tsv``
# function. This will save the ICA annotations to disc in BIDS-Derivative for
# EEG data format.
#
# Note: BIDS-EEG-Derivatives is not fully specified, so this functionality
# may change in the future without notice.

# fname = '<some path to save the components>'
# write_components_tsv(ica, fname)

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



print("Accuracy:", accuracy_score(manual_labels, mapped_auto_labels))
print("Report:\n", classification_report(manual_labels, mapped_auto_labels))


'''
Hai bisogno di due liste:
    - y_true, etichette umane
    - y_pred, etichette automatiche
'''
cm = confusion_matrix(manual_labels, mapped_auto_labels, labels=["brain", "eye blink", "heart beat", "muscle artifact", "line noise", "channel noise", "other"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["brain", "eye blink", "heart beat", "muscle artifact", "line noise", "channel noise", "other"])
disp.plot()
plt.show()
