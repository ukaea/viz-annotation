---
title: 'TokTagger: An open-source interactive annotation platform for tokamak diagnostic data'
tags:
  - Python
  - fusion energy
  - tokamak
  - plasma physics
  - data annotation
  - active learning
  - time series
  - machine learning
authors:
  - name: Samuel Jackson
    orcid: 0000-0001-5301-5095
    corresponding: true
    affiliation: 1
  - name: Matthew Field
    orcid: 0009-0004-1390-0697
    affiliation: 1
  - name: Joshua Blake
    affiliation: 1
  - name: Abdullah Saleem
    affiliation: 1
  - name: Nitesh Bhatia
    orcid: 0000-0003-1367-3477
    affiliation: 1
  - name: Rui Costa
    orcid: 0000-0001-6144-3356
    affiliation: 1
  - name: Nathan Cummings
    orcid: 0000-0003-4359-6337
    affiliation: 1
  - name: Saiful Khan
    orcid: 0000-0002-6796-5670
    affiliation: 1
  - name: Prakhar Sharma
    orcid: 0000-0002-7635-1857
    affiliation: 1
  - name: Stanislas Pamela
    orcid: 0000-0001-8854-1749
    affiliation: 1
  - name: Alejandra Gonzalez-Beltran
    orcid: 0000-0003-3499-8262
    affiliation: 1
affiliations:
 - name: UK Atomic Energy Authority, Culham Science Centre, United Kingdom
   index: 1
date: 17 July 2026
bibliography: paper.bib
---

# Summary

Tokamak fusion experiments such as MAST, MAST-U and JET produce large volumes of multi-modal diagnostic data with every experimental shot, including scalar time-series signals (e.g., plasma current, density, D-alpha emission), camera video, and signal derived spectrograms. Understanding and controlling plasma behaviour depends on identifying physical events within this data, such as edge-localised modes (ELMs), transitions between low- and high-confinement regimes (L-mode/H-mode), disruptions, and other magnetohydrodynamic phenomena. Historically, these events have been identified manually by physicists on a shot-by-shot basis. As experiments scale in number and diagnostic bandwidth, manual labelling no longer scales, motivating the use of machine learning models trained on human-labelled examples to automate detection of these events. Building such models first requires a way to efficiently create, manage, and iterate on labelled datasets drawn directly from tokamak data systems. `TokTagger` is an open-source, web-based annotation platform that addresses this need, combining an interactive labelling interface for time-series, image, and video data with an extensible Python backend that connects directly to existing fusion data-access systems and machine learning workflows.

![The time-series labelling interface, showing multiple synchronised diagnostic signals with time-region annotations.\label{fig:timeseries}](time-series.png)


# Statement of need

Labelling tokamak diagnostic data for machine learning presents challenges that are not well served by general-purpose annotation software. Diagnostic signals are multi-variate and must be viewed and labelled in relation to one another (e.g., correlating a plasma current, density, and soft X-ray emission to identify an IRE), rather than as independent, single-channel time series. Data must also be sourced directly from experiment-specific data systems, such as UDA [@Muir:2015] for MAST and MAST-U, SAL [@sal] for JET, to support inter-shot analysis, rather than from static exported files. Without dedicated tooling, researchers typically resort to ad hoc scripts, spreadsheets, or one-off notebooks to track labels for a given study. These approaches do not scale across users or experiments, are not easily reproducible, and provide no direct route from a labelled dataset to a trained model.

`TokTagger` was designed to close this gap between labelling and modelling. It provides a project-based workflow in which a physicist defines a labelling task (time-series or video annotation), attaches one of several tokamak-aware data loaders, and then browses and labels samples through a web UI. Semi-automated "annotators" (e.g., peak, change-point, jump, and outlier detection) suggest candidate labels to speed up manual work, and machine learning models can be trained directly on the growing set of human annotations and used to automate the labelling process. `TokTagger` is intended to be used by both diagnostic physicists building one-off labelled datasets and by machine learning researchers building larger, actively-maintained training sets for automated event detection across MAST-U and JET.

# State of the field

Several mature, general-purpose annotation tools already exist, including Label Studio [@labelstudio], CVAT [@cvat], and the VGG Image Annotator (VIA) [@Dutta:2019]. These tools offer flexible, well-tested interfaces for labelling images, video, audio, and text, and some support generic time-series annotation. However, none of them provide native support for visualising and jointly labelling multiple, synchronised diagnostic signals per sample, nor do they ship with connectors to fusion-specific data systems such as UDA, SAL, or FAIR-MAST. Using these tools for tokamak data therefore requires a substantial pre-processing and data-export step before labelling can begin, and an equally substantial post-processing step to get labels back into a form usable for training models on live experiment data.

Within the fusion community, event detection has more often relied on bespoke, single-purpose scripts written by individual researchers or groups for a specific diagnostic or campaign, rather than on shared, reusable software with a common interface and central annotation store. This makes it difficult for labelled datasets and labelling effort to be shared or built upon across studies.

The closest related work is the Data Fusion Labeler (dFL) [@MICHOSKI2026115872], which provides tools for time-series and spectrogram labelling. Our apporach is distinguished from dFL in that 1) it is open source and free to the community, 2) it supports multi-variate time-series labelling, spectrogram, and video labelling, and 3) it designed as a web interface with multi-user support and a centralised database for storing annotations, facilitating collaboration and sharing of labelled datasets.

`TokTagger` differs from these approaches in several ways. First, it provides synchronised, multi-variate time-series visualisation and labelling designed specifically for tokamak diagnostics, alongside image and frame-by-frame video labelling, within a single project/database framework. Second, it ships with data loaders that connect directly to UDA, SAL, and FAIR-MAST, as well as generic tabular and image-based loaders, removing the need for a separate data-export step. Third, it couples the labelling interface directly to a human-in-the-loop active learning system: models are trained on the current set of annotations, predictions are surfaced back into the UI with associated uncertainty estimates, and an uncertainty-based query strategy selects which unlabelled samples should be reviewed next. This tight loop between labelling and modelling, combined with fusion-specific data access, distinguishes `TokTagger` from both general-purpose annotation tools and one-off analysis scripts.

# Software design

`TokTagger` consists of a Python backend, built with FastAPI and backed by a NoSQL database (either MongoDB or Mongita for local installation), and a React/TypeScript frontend that communicates with the backend over a REST API. This separation allows the same annotation and model functionality to be driven either through the web UI or scripted directly against the Python API.

![Overview of the `TokTagger` system architecture, showing the React UI, FastAPI backend, MongoDB database, and Ray-managed model workers.\label{fig:architecture}](toktagger-arch.png){ width=80% }


The backend is organised around three extensible registries that follow a common decorator-based registration pattern, allowing users to add domain-specific functionality without modifying `TokTagger`'s core code:

- **Data loaders** (`toktagger.api.core.data_loaders`) define how samples are retrieved from a data source and converted into one of `TokTagger`'s standard data schemas (time series, multi-variate time series, spectrogram, or image). Built-in loaders cover UDA, SAL, FAIR-MAST, local tabular files, and local image/Numpy array files. New loaders are registered with a single `@LoaderRegistry.register(...)` decorator.

- **Annotators** provide algorithmic pre-labelling for time series: peak detection, outlier detection (Mean Absolute Deviation or Isolation Forest [@scikit-learn]), change-point detection (PELT or a Hidden Markov Model, via `ruptures` [@Truong:2020] and `hmmlearn`), and jump detection. These surface candidate annotations that a human reviewer can accept, adjust, or reject.

- **Models** (`@ModelRegistry.register(...)`) wrap arbitrary machine learning estimators (scikit-learn, PyTorch, or otherwise) behind a common `train`/`predict`/`save`/`load` interface, together with helpers for train/validation/test splitting and progress reporting back to the UI. Model training and inference are scheduled with Ray [@Moritz:2018], which allows CPU- and GPU-bound tasks to be distributed across worker nodes and enables the same code to run unmodified on a laptop or a shared cluster.

Annotation data is represented with a small set of Pydantic schemas — time points, time regions, bounding boxes, polygons, and their per-frame video equivalents — that are shared between the UI, the database layer, and the model API, so that a prediction returned by a model is structurally identical to an annotation created by a human. Projects track a query strategy (sequential, random, or uncertainty-based) that determines which sample is served next, allowing model predictions to directly steer labelling effort towards the most informative samples.

\autoref{fig:timeseries} shows the time-series labelling interface, in which multiple synchronised diagnostic signals are displayed with shared time axes and can be labelled with time-point and time-region annotations, either manually or with the assistance of the automated annotators described above. An equivalent frame-by-frame interface, supporting bounding box, polygon, and point annotations with propagation between frames, is provided for camera and video data. \autoref{fig:architecture} summarises the overall system architecture connecting the UI, API, database, and model workers.

The project is tested with `pytest`, covering the core annotation and data loader logic, the REST API, and the MongoDB integration, together with end-to-end browser tests written with Playwright that exercise the UI directly. Continuous integration runs the test suite and `ruff` linting on every change.

# Research impact statement

`TokTagger` was initially developed by the UK Atomic Energy Authority (UKAEA) and the Science and Technology Facilities Council (STFC) to support the creation of labelled datasets from MAST, MAST-U, and JET diagnostic data for machine learning research. Since its first public release (v0.1.0), the project has grown through contributions from a team of developers across UKAEA beyond its original authors, with new data loaders, annotation types, and model-training functionality added in each subsequent release. It is used internally at UKAEA to curate labelled datasets in support of machine learning models for tasks such as automated ELM and disruption detection from MAST-U and JET diagnostics, and its direct integration with UDA, SAL, and FAIR-MAST is intended to make it straightforward for other groups working with these data systems to build similar labelled datasets for their own machine learning workflows. As a young, actively-developed project, `TokTagger`'s broader adoption and citation record are still emerging; this paper accompanies its release as reusable, general-purpose infrastructure for the wider fusion data science community.

# Acknowledgements

We acknowledge the support of the UK Atomic Energy Authority and the Science and Technology Facilities Council (STFC) and the wider MAST-U and JET diagnostics and data systems teams whose infrastructure `TokTagger` builds upon.

# References
