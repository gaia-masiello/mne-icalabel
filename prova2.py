'''

This tutorial covers the basics of independent components analysis (ICA) and shows how ICA can be used for artifact repair; an extended example illustrates 
repair of ocular and heartbeat artifacts. For conceptual background on ICA, see this scikit-learn tutorial.

We begin as always by importing the necessary Python modules and loading some example data. Because ICA can be computationally intense, we’ll also crop the data 
to 60 seconds; and to save ourselves from repeatedly typing mne.preprocessing we’ll directly import a few functions and classes from that submodule:

'''

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os

import mne
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs

from mne_icalabel import label_components


sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(
    sample_data_folder, "MEG", "sample", "sample_audvis_filt-0-40_raw.fif"
)
raw = mne.io.read_raw_fif(sample_data_raw_file)

# Here we'll crop to 60 seconds and drop gradiometer channels for speed
raw.crop(tmax=60.0).pick(picks=["mag", "eeg", "stim", "eog"])
raw.load_data()

'''
Example: EOG and ECG artifact repair

Visualizing the artifacts
Let’s begin by visualizing the artifacts that we want to repair. 
In this dataset they are big enough to see easily in the raw data:

'''

# pick some channels that clearly show heartbeats and blinks
regexp = r"(MEG [12][45][123]1|EEG 00.)"
artifact_picks = mne.pick_channels_regexp(raw.ch_names, regexp=regexp)
raw.plot(order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False)

# we can get a summary of how the ocular artifact manifests across each channel type using create_eog_epochs like we did in the Overview of artifact detection tutorial:
eog_evoked = create_eog_epochs(raw).average()
eog_evoked.apply_baseline(baseline=(None, -0.2))
eog_evoked.plot_joint()

# now we’ll do the same for the heartbeat artifacts, using create_ecg_epochs:
ecg_evoked = create_ecg_epochs(raw).average()
ecg_evoked.apply_baseline(baseline=(None, -0.2))
ecg_evoked.plot_joint()

'''

Filtering to remove slow drifts

Before we run the ICA, an important step is filtering the data to remove low-frequency drifts, which can negatively affect the quality of the ICA fit. 
The slow drifts are problematic because they reduce the independence of the assumed-to-be-independent sources (e.g., during a slow upward drift, the neural, 
heartbeat, blink, and other muscular sources will all tend to have higher values), making it harder for the algorithm to find an accurate solution. 
A high-pass filter with 1 Hz cutoff frequency is recommended. However, because filtering is a linear operation, the ICA solution found from the filtered 
signal can be applied to the unfiltered signal (see [2] for more information), so we’ll keep a copy of the unfiltered Raw object around so we can apply 
the ICA solution to it later.

'''

filt_raw = raw.copy().filter(l_freq=1.0, h_freq=100)
filt_raw = filt_raw.set_eeg_reference("average")


'''

Fitting ICA

    Ignoring the time domain
    The ICA algorithms implemented in MNE-Python find patterns across channels, 
    but ignore the time domain. This means you can compute ICA on discontinuous 
    Epochs or Evoked objects (not just continuous Raw objects), or only use every 
    Nth sample by passing the decim parameter to ICA.fit().

Now we’re ready to set up and fit the ICA. Since we know (from observing our raw data) 
that the EOG and ECG artifacts are fairly strong, we would expect those artifacts to be 
captured in the first few dimensions of the PCA decomposition that happens before the ICA. 
Therefore, we probably don’t need a huge number of components to do a good job of isolating 
our artifacts (though it is usually preferable to include more components for a more 
accurate solution). As a first guess, we’ll run ICA with n_components=15 (use only the first 
15 PCA components to compute the ICA decomposition) — a very small number given that our 
data has over 300 channels, but with the advantage that it will run quickly and we will able 
to tell easily whether it worked or not (because we already know what the EOG / ECG artifacts 
should look like).

ICA fitting is not deterministic (e.g., the components may get a sign flip on different runs, 
or may not always be returned in the same order), so we’ll also specify a random seed so that 
we get identical results each time this tutorial is built by our web servers.


Warning: Epochs used for fitting ICA should not be baseline-corrected. 
         Because cleaning the data via ICA may introduce DC offsets, 
         we suggest to baseline correct your data after cleaning (and 
         not before), should you require baseline correction.

'''


ica = ICA(n_components=15, max_iter="auto", method="infomax", random_state=97, fit_params=dict(extended=True),)
ica.fit(filt_raw)
ica


# Looking at the ICA solution
# Now we can examine the ICs to see what they captured.
# Using get_explained_variance_ratio(), we can retrieve the fraction 
# of variance in the original data that is explained by our ICA components 
# in the form of a dictionary:
explained_var_ratio = ica.get_explained_variance_ratio(filt_raw)
for channel_type, ratio in explained_var_ratio.items():
    print(f"Fraction of {channel_type} variance explained by all components: {ratio}")

'''
The values were calculated for all ICA components jointly, but separately 
for each channel type (here: magnetometers and EEG).

We can also explicitly request for which component(s) and channel type(s) 
to perform the computation:
explained_var_ratio = ica.get_explained_variance_ratio(
    filt_raw, components=[0], ch_type="eeg"
)
# This time, print as percentage.
ratio_percent = round(100 * explained_var_ratio["eeg"])
print(
    f"Fraction of variance in EEG signal explained by first component: "
    f"{ratio_percent}%"
)

PROVA!!!

'''

raw.load_data()
ica.plot_sources(raw, show_scrollbars=False)

'''
plot_sources will show the time series of the ICs. Note that in our call 
to plot_sources we can use the original, unfiltered Raw object. 
A helpful tip is that right clicking (or control + click with a trackpad) 
on the name of the component will bring up a plot of its properties. 
In this plot, you can also toggle the channel type in the topoplot 
(if you have multiple channel types) with ‘t’ and whether the spectrum 
is log-scaled or not with ‘l’.


Here we can pretty clearly see that the first component (ICA000) 
captures the EOG signal quite well, and the second component (ICA001) 
looks a lot like a heartbeat (for more info on visually identifying 
Independent Components, this EEGLAB tutorial is a good resource). 
We can also visualize the scalp field distribution of each component 
using plot_components. These are interpolated based on the values in 
the ICA mixing matrix:
'''
ica.plot_components()

'''
Note: plot_components (which plots the scalp field topographies for 
      each component) has an optional inst parameter that takes an 
      instance of Raw or Epochs. Passing inst makes the scalp topographies 
      interactive: clicking one will bring up a diagnostic plot_properties 
      window (see below) for that component.
'''

'''
In the plots above it’s fairly obvious which ICs are capturing our 
EOG and ECG artifacts, but there are additional ways visualize them 
anyway just to be sure. First, we can plot an overlay of the original 
signal against the reconstructed signal with the artifactual ICs 
excluded, using plot_overlay: '''
# blinks
ica.plot_overlay(raw, exclude=[0], picks="eeg")
# heartbeats
ica.plot_overlay(raw, exclude=[1], picks="mag")

#VEDI SE FUNZIONA
ic_labels = label_components(filt_raw, ica, method="iclabel")
print(ic_labels["labels"])


#We can also plot some diagnostics of each IC using plot_properties:
ica.plot_properties(raw, picks=[0, 1])


'''
Selecting ICA components manually

Once we’re certain which components we want to exclude, we can specify 
that manually by setting the ica.exclude attribute. Similar to marking 
bad channels, merely setting ica.exclude doesn’t do anything immediately 
(it just adds the excluded ICs to a list that will get used later when 
it’s needed). Once the exclusions have been set, ICA methods like 
plot_overlay will exclude those component(s) even if no exclude 
parameter is passed, and the list of excluded components will be 
preserved when using mne.preprocessing.ICA.save and 
mne.preprocessing.read_ica.
'''
ica.exclude = [0, 1]  # indices chosen based on various plots above

'''
Now that the exclusions have been set, we can reconstruct the sensor 
signals with artifacts removed using the apply method (remember, we’re 
applying the ICA solution from the filtered data to the original unfiltered 
signal). Plotting the original raw data alongside the reconstructed data 
shows that the heartbeat and blink artifacts are repaired.
'''
# ica.apply() changes the Raw object in-place, so let's make a copy first:
reconst_raw = raw.copy()
ica.apply(reconst_raw)

raw.plot(order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False)
reconst_raw.plot(
    order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False
)
del reconst_raw