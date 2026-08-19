# Event Detection Models

TokTagger ships with four built-in models for detecting events in time series signals. Each model learns from your `TimeRegion` annotations and then scans new samples for similar patterns, producing `TimeRegion` predictions that you can review and validate.

!!! tip
    These models require the optional ML dependencies. Install them with:
    ```sh
    pip install toktagger[models]
    ```

## Overview

The four models fall into two families:

- **Template matching** — [DTW Motif](#dtw-motif) and [STUMPY Motif](#stumpy-motif) store example segments ("templates") taken directly from your annotations, then slide a window across new signals and flag windows that are close to a template under a distance measure. No classifier is trained; the "model" is just the set of templates plus a distance threshold.
- **Windowed classification** — [MiniRocket](#minirocket) and [Shapelet Transform](#shapelet-transform) train a binary classifier (event vs. background) on windows sampled from your annotations, then slide a window across new signals and classify each one.

All four models:

- Support single-channel or multi-channel (multivariate) signals — pass one or more entries in `signal_names`.
  The training form shows a dropdown of the signals in the project's samples, so you select the channels instead of typing them.
- z-normalise signal windows before comparison, so detection is based on shape rather than absolute amplitude.
- Merge adjacent positive detections into a single `TimeRegion`, then run greedy non-maximum suppression (NMS) to remove heavily overlapping regions of the same label.
- Only train on `TimeRegion` (start/end time) annotations — point annotations are ignored.
- Are registered for the `"time-series"` task, so they appear alongside any [custom models](custom_models.md) you add.

## DTW Motif

Registered as **`dtw_motif`**.

Template-matching event detector using z-normalised [Dynamic Time Warping (DTW)](https://dtaidistance.readthedocs.io/) distance. DTW allows templates to match signals that are stretched or compressed in time, which makes it more tolerant of speed variation between events than a fixed Euclidean comparison.

**How it works:** during training, a fixed-length segment is extracted and z-normalised around every matching annotation to form a set of templates (one per annotated event, optionally filtered to a single label). During prediction, a sliding window scans each new sample; at every step the window is compared against every template using DTW distance (or `dtw_ndim` for multivariate signals), and the closest template's label is assigned if its distance is below `threshold`.

### Training Parameters

- **class_label**: Annotation label to build templates from. Leave blank to build templates from every label present in the training annotations.
- **signal_names**: Signal channels to use. Provide one for single-channel mode, or multiple for multivariate DTW (e.g. `["Ip", "dalpha"]`).
- **threshold**: Maximum z-normalised DTW distance for a detection. For z-normalised windows, typical values are 2–20 — lower values require closer shape matches.
- **window_size**: Window size in samples. Unlike MiniRocket/Shapelet Transform/STUMPY, this is set explicitly rather than inferred from annotation durations.

### Prediction Parameters

- **step_size**: Sliding window stride in samples. Increase for faster inference at the cost of position precision.

### Output

One or more `TimeRegion` annotations per sample, one per contiguous cluster of matching window positions, labelled with the closest template's label.

!!! note
    DTW distance computation is relatively expensive, so prediction time scales with the number of templates and the number of sliding-window positions (`(signal length) / step_size`). Increase `step_size` or reduce the number of templates if predictions are too slow.

## STUMPY Motif

Registered as **`stumpy_motif`**.

Template-matching event detector using [STUMPY](https://stumpy.readthedocs.io/)'s FFT-based MASS (Mueen's Algorithm for Similarity Search) distance profile. This computes the same z-normalised Euclidean distance as a brute-force sliding comparison, but does so for every window position in a signal at once using an FFT, making it much faster than DTW Motif for long signals.

**How it works:** training extracts and z-normalises a fixed-length segment around every matching annotation to form templates, with `window_size` inferred automatically as the median annotation duration (in samples) across the training set. During prediction, `stumpy.mass` computes a full distance profile between each template and each sample signal (averaged across channels for multivariate signals); window positions whose distance falls below `threshold` are flagged and assigned the label of their closest template.

### Training Parameters

- **class_label**: Annotation label to include as templates. Leave blank to build templates from every label present in the training annotations.
- **signal_names**: Signal channels to use. Provide one for single-channel mode, or multiple for multivariate STUMPY (e.g. `["Ip", "dalpha"]`).
- **threshold**: Maximum z-normalised Euclidean distance (MASS) for a detection. Typical values are 1–5; lower values require closer matches.

### Prediction Parameters

- **threshold**: Optional override for the maximum MASS distance. Defaults to the threshold saved during training, so you can tune detection sensitivity at prediction time without retraining.

### Output

One or more `TimeRegion` annotations per sample, one per contiguous cluster of matching window positions, labelled with the closest template's label.

!!! tip
    Because prediction threshold can be overridden without retraining, STUMPY Motif is a good choice for iterating on detection sensitivity — train once, then try a few `threshold` values at predict time to see which best matches your annotations.

## MiniRocket

Registered as **`minirocket`**.

Sliding-window binary event classifier using [MiniRocket](https://github.com/angus924/minirocket) convolutional features and a Ridge classifier. MiniRocket applies a large, fixed set of random convolutional kernels to each window and pools the results into a feature vector, which a `RidgeClassifierCV` then classifies as event or background. It is fast to train and typically strong on shape-based classification tasks.

**How it works:** `window_size` is inferred as the median annotation duration (in samples). For each training sample, positive windows are extracted centered on annotations matching `class_label`, and negative ("background") windows are randomly sampled from the remainder of the signal, avoiding overlap with any annotation. A `MiniRocket` (or `MiniRocketMultivariate` for multi-channel signals) transformer is fit on these windows and used to generate features, which train a `RidgeClassifierCV`. During prediction, a sliding window scans each new sample, and every window is transformed and classified; positive windows are merged into detections.

### Training Parameters

- **class_label**: Annotation label to train the binary classifier on (event vs. background). Must match one of the project's configured time-region annotation labels.
- **signal_names**: Signal channels to classify. Provide one for single-channel mode, or multiple for multivariate (e.g. `["Ip", "dalpha"]`).
- **n_background_per_shot**: Number of background (negative) windows sampled per training shot.
- **num_kernels**: Number of MiniRocket convolutional kernels. Higher values can improve accuracy at the cost of training/prediction time.

### Prediction Parameters

- **step_size**: Sliding window stride in samples (larger = faster but coarser).

### Output

One or more `TimeRegion` annotations per sample, one per contiguous cluster of windows classified as positive, labelled with `class_label`.

!!! note
    Unlike DTW Motif and STUMPY Motif, MiniRocket trains a single binary classifier for one `class_label` at a time — it cannot produce multiple distinct event labels from a single trained model. Train a separate model per label if you need to detect several event types.

## Shapelet Transform

Registered as **`shapelet_transform`**.

Sliding-window binary event classifier using [sktime's `ShapeletTransformClassifier`](https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.classification.shapelet_based.ShapeletTransformClassifier.html). Shapelets are short, discriminative subsequences automatically discovered from the training windows; the classifier represents each window by its distance to the best matching shapelets, then classifies using those distances. Shapelets are more interpretable than MiniRocket's convolutional features, at the cost of slower training.

**How it works:** training follows the same window extraction approach as MiniRocket — `window_size` inferred from median annotation duration, positive windows centered on `class_label` annotations, and randomly sampled negative windows. A `ShapeletTransformClassifier` is then fit directly on the raw (z-normalised) windows. During prediction, a sliding window scans each sample and each window is classified directly by the fitted classifier; positive windows are merged into detections.

### Training Parameters

- **class_label**: Annotation label to train the binary classifier on (event vs. background). Must match one of the project's configured time-region annotation labels.
- **signal_names**: Signal channels to use. Provide one for univariate, or multiple for multivariate shapelet learning (e.g. `["Ip", "dalpha"]`).
- **n_background_per_shot**: Number of background (negative) windows sampled per training shot.
- **max_shapelets**: Maximum number of shapelets to extract per class.
- **n_shapelet_samples**: Number of candidate shapelet samples to evaluate.
- **batch_size**: Batch size for shapelet fitting.

### Prediction Parameters

- **step_size**: Sliding window stride in samples (larger = faster but coarser).

### Output

One or more `TimeRegion` annotations per sample, one per contiguous cluster of windows classified as positive, labelled with `class_label`.

!!! note
    Like MiniRocket, Shapelet Transform trains one binary classifier per `class_label`. Shapelet candidate search is more computationally expensive than MiniRocket's fixed kernel transform, so training time grows more quickly with `n_shapelet_samples` and the number/length of training windows.

## Choosing a Model

- **Few labelled examples, want to match a specific shape exactly?** Start with **STUMPY Motif** — it needs no negative examples, is fast to evaluate via FFT, and lets you retune the detection threshold at predict time without retraining.
- **Events vary in duration or speed but keep a similar shape?** Use **DTW Motif** — its warping-tolerant distance handles stretched/compressed events that Euclidean-based methods (STUMPY, MiniRocket, Shapelet Transform) may miss.
- **Enough labelled examples for a proper classifier, and want speed?** Use **MiniRocket** — it is typically the fastest to train of the two classification-based models and performs well on most shape-classification tasks.
- **Want more interpretable features, and training time is not a concern?** Use **Shapelet Transform** — the learned shapelets can be inspected as representative sub-patterns of the event class.
