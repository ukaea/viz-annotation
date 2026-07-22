"use client";

import type { ImageAnnotation } from "@annotorious/react";

/** Frame number within a video (0-based). */
export type FrameIndex = number;

export type DrawingTool = "rectangle" | "polygon" | "point";
export type AnnotoriousDrawingTool = Exclude<DrawingTool, "point">;

/**
 * Stable identifier for a tracked instance in the UI/session.
 * Format: "<className>::<trackId>"
 */
export type TrackKey = `${string}::${string}`;

/** How the current selection was produced (user picked it vs. auto-assigned). */
export type SelectionSource = "explicit" | "auto" | null;

/**
 * The "armed" selection used when creating new annotations.
 * - className: required to enable drawing
 * - trackId: optional; when null, a new id can be allocated automatically
 */
export type Selection = {
  className: string | null;
  trackId: string | null;
  source: SelectionSource;
};

/**
 * UI summary for one tracked instance across frames.
 * `frames` is unique + sorted; `count` is total boxes across all frames.
 */
export type InstanceProfile = {
  key: TrackKey;
  className: string;
  classId: number;
  trackId: string;
  frames: number[]; // unique, sorted
  count: number; // total boxes across frames
};

/** In-memory annotation storage keyed by frame, using the native Annotorious model. */
export type ByFrameMap = Map<FrameIndex, ImageAnnotation[]>;

/**
 * Minimal backend payload for a video bounding box annotation.
 * This matches what we emit back to the server.
 */
export type VideoBoundingBox = {
  type: "video_bounding_box";
  frame: number;
  track_id: string;
  label: string;
  x_min: number;
  y_min: number;
  width: number;
  height: number;
  created_by?: string;
  timestamp?: string;
  class_id?: number;
};

export type VideoPolygon = {
  type: "video_polygon";
  frame: number;
  track_id: string;
  label: string;
  segmentation: number[];
  created_by?: string;
  timestamp?: string;
  class_id?: number;
};

export type VideoPoint = {
  type: "video_point";
  frame: number;
  track_id: string;
  label: string;
  x: number;
  y: number;
  created_by?: string;
  timestamp?: string;
  class_id?: number;
};

export type VideoAnnotationShape = VideoBoundingBox | VideoPolygon | VideoPoint;

/**
 * Supported label set for this UI.
 * These ids are used when exporting to backend formats that expect a numeric class id.
 */
export const class_labels: { id: number; name: string }[] = [
  { id: 1, name: "UFO" },
  { id: 2, name: "Minor UFO" },
  { id: 3, name: "Major UFO" },
  { id: 4, name: "Disruption" },
  { id: 5, name: "Marfe" },
  { id: 6, name: "Other Anomaly" },
];

/** Resolve a human label name to a numeric class id (defaults to 1). */
export function classIdForName(className: string): number {
  const key = (className || "").trim().toLowerCase();
  const hit = class_labels.find((c) => c.name.toLowerCase() === key);
  return hit?.id ?? 1;
}

/** Build a TrackKey from a class + track id. */
export function makeTrackKey(className: string, trackId: string): TrackKey {
  return `${className}::${trackId}`;
}

/** Parse a TrackKey back into its components. */
export function parseTrackKey(key: TrackKey): {
  className: string;
  trackId: string;
} {
  const idx = key.lastIndexOf("::");
  return { className: key.slice(0, idx), trackId: key.slice(idx + 2) };
}

/**
 * Construct a stable source identifier for the current frame.
 * Used for `target.source` so downstream exports/debugging can associate an annotation
 * with a specific project/sample/frame.
 */
export function buildSourceKey(args: {
  projectId: string;
  sampleId: string;
  frame: number;
}): string {
  return `app://p/${args.projectId}/s/${args.sampleId}/f/${args.frame}`;
}
