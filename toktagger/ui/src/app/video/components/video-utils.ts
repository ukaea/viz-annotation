"use client";

import type { ImageAnnotation } from "@annotorious/react";
import type { Annotation } from "@/types";
import {
  VideoBoundingBoxSchema,
  VideoPointSchema,
  VideoPolygonSchema,
} from "@/types";
import type {
  ByFrameMap,
  FrameIndex,
  InstanceProfile,
  TrackKey,
} from "./types";
import { classIdForName, makeTrackKey, buildSourceKey } from "./types";
import { getLabelTrack } from "./anno-utils";

/**
 * Simple deep clone for annotation payloads.
 * Annotorious annotations are plain data objects, so JSON cloning is sufficient here.
 */
export function deepClone<T>(v: T): T {
  return JSON.parse(JSON.stringify(v)) as T;
}

/**
 * Normalize a track id into a stable, readable key for comparisons and lookups.
 * We keep this intentionally conservative (no heavy validation), just formatting.
 */
export function canonicalizeTrackId(trackId: string): string {
  const s = trackId.trim();
  if (!s) return "";
  return s.replace(/\s+/g, "-").toLowerCase();
}

/**
 * Best-effort coercion to a non-empty track id.
 * Used primarily in equality checks and deletion matching.
 */
export function ensureTrackId(trackId: string): string {
  const c = canonicalizeTrackId(trackId);
  return c.length > 0 ? c : "1";
}

/* ------------------------------------------------------------------ */
/* Human-readable track ids (e.g. mellow-glove-4)                      */
/* ------------------------------------------------------------------ */

const READABLE_ADJECTIVES = [
  "bright",
  "calm",
  "curious",
  "distant",
  "eager",
  "faint",
  "fast",
  "gentle",
  "ghostly",
  "glowing",
  "hidden",
  "icy",
  "lively",
  "lucky",
  "mellow",
  "mystic",
  "nimble",
  "quiet",
  "rapid",
  "shiny",
  "silent",
  "sly",
  "stealthy",
  "swift",
  "tiny",
  "wild",
  "young",
  "zealous",
  "ancient",
  "brave",
] as const;

const READABLE_NOUNS = [
  "comet",
  "signal",
  "flare",
  "shadow",
  "spark",
  "meteor",
  "photon",
  "plume",
  "echo",
  "nebula",
  "pattern",
  "speck",
  "glint",
  "trace",
  "whisper",
  "streak",
  "drift",
  "halo",
  "vortex",
  "wave",
  "pulse",
  "arc",
  "beam",
  "glow",
  "ripple",
  "stream",
  "trail",
  "blip",
  "sparkle",
  "cluster",
] as const;

function randInt(minIncl: number, maxIncl: number) {
  return Math.floor(Math.random() * (maxIncl - minIncl + 1)) + minIncl;
}

function randomReadableSlug(): string {
  const adj = READABLE_ADJECTIVES[randInt(0, READABLE_ADJECTIVES.length - 1)];
  const noun = READABLE_NOUNS[randInt(0, READABLE_NOUNS.length - 1)];
  const n = randInt(1, 9);
  return `${adj}-${noun}-${n}`;
}

/**
 * Generate a readable track id that doesn't collide with an existing set
 * (collisions checked after canonicalization).
 */
export function uniqueReadableTrackId(
  existingTrackIds: Iterable<string>,
): string {
  const used = new Set<string>();
  for (const t of existingTrackIds) {
    const c = canonicalizeTrackId(t);
    if (c) used.add(c);
  }

  let candidate = randomReadableSlug();
  let guard = 0;

  while (used.has(canonicalizeTrackId(candidate)) && guard++ < 200) {
    candidate = randomReadableSlug();
  }

  // Extremely unlikely fallback: add a short unique suffix.
  if (used.has(canonicalizeTrackId(candidate))) {
    const suffix =
      globalThis.crypto?.randomUUID?.().slice(0, 6) ??
      Math.random().toString(36).slice(2, 8);
    candidate = `${candidate}-${suffix}`;
  }

  return canonicalizeTrackId(candidate);
}

/**
 * Collect all track ids for a given class across all frames.
 * This is used to avoid allocating duplicates when creating new instances.
 */
export function existingTrackIdsForClass(
  byFrame: ByFrameMap,
  className: string,
): string[] {
  const cls = className.trim();
  if (!cls) return [];

  const out: string[] = [];
  for (const list of byFrame.values()) {
    for (const a of list) {
      const got = getLabelTrack(a);
      if ((got.className || "").trim() !== cls) continue;
      const tid = canonicalizeTrackId(got.trackId || "");
      if (tid) out.push(tid);
    }
  }
  return out;
}

/**
 * Build the current per-class numeric track-id counters from seeded annotations.
 * The returned map stores the highest numeric id seen for each class.
 */
export function buildNextTrackIdState(
  annotations: Annotation[],
): Map<string, number> {
  const nextTrackNums = new Map<string, number>();

  for (const annotation of annotations) {
    const parsed =
      annotation.type === "video_bounding_box"
        ? VideoBoundingBoxSchema.safeParse(annotation)
        : annotation.type === "video_polygon"
          ? VideoPolygonSchema.safeParse(annotation)
          : annotation.type === "video_point"
            ? VideoPointSchema.safeParse(annotation)
            : null;
    if (!parsed || !parsed.success) continue;

    const className = (parsed.data.label || "").trim();
    const numericTrackId = Number.parseInt(parsed.data.track_id || "", 10);

    if (!className || !Number.isFinite(numericTrackId)) continue;

    const prev = nextTrackNums.get(className) ?? 0;
    if (numericTrackId > prev) {
      nextTrackNums.set(className, numericTrackId);
    }
  }

  return nextTrackNums;
}

/**
 * Advance the numeric track-id counter for a class and return the allocated id.
 * The map is mutated intentionally because the owning session keeps the state.
 */
export function allocateNextTrackId(
  nextTrackNums: Map<string, number>,
  className: string,
): string {
  const key = (className || "").trim();
  const next = (nextTrackNums.get(key) ?? 0) + 1;
  nextTrackNums.set(key, next);
  return String(next);
}

/** Immutable update: set a frame's overlay list. */
export function mapSetFrame(
  prev: ByFrameMap,
  frame: FrameIndex,
  list: ImageAnnotation[],
): ByFrameMap {
  const next = new Map(prev);
  next.set(frame, list);
  return next;
}

/** Immutable update: clear a single frame to an empty overlay list. */
export function mapClearFrame(prev: ByFrameMap, frame: FrameIndex): ByFrameMap {
  const next = new Map(prev);
  next.set(frame, []);
  return next;
}

/** Immutable update: clear all known frames to empty overlay lists. */
export function mapClearAll(prev: ByFrameMap): ByFrameMap {
  const next = new Map<FrameIndex, ImageAnnotation[]>();
  for (const [f] of prev.entries()) next.set(f, []);
  return next;
}

/** Flatten all frame overlays into one list. */
export function flattenByFrame(byFrame: ByFrameMap): ImageAnnotation[] {
  const out: ImageAnnotation[] = [];
  for (const list of byFrame.values()) out.push(...list);
  return out;
}

/**
 * Derive instance profiles from the per-frame overlays.
 * Profiles are keyed by (className, trackId) and include counts + which frames they appear in.
 */
export function deriveInstances(
  byFrame: ByFrameMap,
  getLabelTrack: (a: ImageAnnotation) => {
    className?: string | null;
    trackId?: string | null;
  },
): InstanceProfile[] {
  const map = new Map<TrackKey, InstanceProfile>();

  const frames = Array.from(byFrame.keys()).sort((a, b) => a - b);

  for (const frame of frames) {
    const list = byFrame.get(frame) ?? [];
    for (const a of list) {
      const { className, trackId } = getLabelTrack(a);
      if (!className || !trackId) continue;

      const key = makeTrackKey(className, trackId);
      const existing = map.get(key);
      if (!existing) {
        map.set(key, {
          key,
          className,
          classId: classIdForName(className),
          trackId,
          frames: [frame],
          count: 1,
        });
      } else {
        existing.count += 1;
        if (!existing.frames.includes(frame)) existing.frames.push(frame);
      }
    }
  }

  const out = Array.from(map.values());
  for (const inst of out) inst.frames.sort((a, b) => a - b);

  // Show instances by the first frame they appear in, with stable ordering
  // for ties based on alphabetical track id.
  out.sort(
    (a, b) =>
      (a.frames[0] ?? Number.POSITIVE_INFINITY) -
        (b.frames[0] ?? Number.POSITIVE_INFINITY) ||
      a.trackId.localeCompare(b.trackId),
  );

  return out;
}

/**
 * Remove all annotations belonging to a specific (className, trackId) pair across all frames.
 */
export function deleteTrackAcrossFrames(
  byFrame: ByFrameMap,
  match: { className: string; trackId: string },
  getLabelTrack: (a: ImageAnnotation) => {
    className?: string | null;
    trackId?: string | null;
  },
): ByFrameMap {
  const cls = match.className.trim();
  const tid = ensureTrackId(match.trackId);

  const next = new Map<FrameIndex, ImageAnnotation[]>();

  for (const [frame, list] of byFrame.entries()) {
    const filtered = list.filter((a) => {
      const got = getLabelTrack(a);
      const c = (got.className || "").trim();
      const t = ensureTrackId(got.trackId || "");
      return !(c === cls && t === tid);
    });
    next.set(frame, filtered);
  }

  return next;
}

/**
 * Seed `nextFrame` with a copy of `frame` only if `nextFrame` is currently empty.
 * `withRetarget` is responsible for adjusting any frame-specific fields (e.g. target.source).
 */
export function forwardPropagateIfEmpty(
  byFrame: ByFrameMap,
  frame: FrameIndex,
  nextFrame: FrameIndex,
  ids: { projectId: string; sampleId: string },
): ByFrameMap {
  const cur = byFrame.get(frame) ?? [];
  if (cur.length === 0) return byFrame;

  const nxt = byFrame.get(nextFrame) ?? [];
  if (nxt.length > 0) return byFrame;

  const nextKey = buildSourceKey({
    projectId: ids.projectId,
    sampleId: ids.sampleId,
    frame: nextFrame,
  });

  const seeded = cur.map((a) => {
    const cloned = deepClone(a);
    return {
      ...cloned,
      target: { ...cloned.target, source: nextKey },
    };
  });

  const next = new Map(byFrame);
  next.set(nextFrame, seeded);
  return next;
}
