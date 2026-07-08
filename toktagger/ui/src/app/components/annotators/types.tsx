export enum AnnotatorTypes {
  PEAK_DETECTION = "peak_detection",
  CHANGE_POINT_DETECTION = "change_point_detection",
  JUMP_DETECTION = "jump_detection",
  OUTLIER_DETECTION = "outlier_detection",
  SPECTROGRAM_THRESHOLD = "spectrogram_threshold",
}

// Mirrors the "annotators::<type>" prefix the backend stamps on annotator
// suggestions (toktagger/api/core/annotators.py), so a real user can never
// collide with it (see the reserved-prefix check in the users router).
export const annotatorCreatedBy = (type: AnnotatorTypes) =>
  `annotators::${type}`;
