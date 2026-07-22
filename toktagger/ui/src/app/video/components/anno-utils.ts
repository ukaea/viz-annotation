"use client";

import {
  ShapeType,
  type AnnotationBody,
  type ImageAnnotation,
  type Polygon,
  type PolygonGeometry,
  type Rectangle,
  type RectangleGeometry,
  type ImageAnnotationTarget,
  type Shape,
} from "@annotorious/react";
import type { Ellipse, EllipseGeometry } from "@annotorious/annotorious";
import type {
  AnnotoriousDrawingTool,
  VideoAnnotationShape,
  VideoBoundingBox,
  VideoPoint,
  VideoPolygon,
} from "./types";
import { classIdForName } from "./types";

/**
 * Annotorious annotation helpers.
 *
 * We store two string bodies on each annotation:
 * - purpose="tagging"     -> class label
 * - purpose="identifying" -> track id
 * - purpose="creator"     -> backend created_by value
 *
 * These helpers keep the shape consistent and make conversion to/from backend boxes trivial.
 */

type UnknownRecord = Record<string, unknown>;
const POINT_BODY_PURPOSE = "shape";
const POINT_BODY_VALUE = "point";
export const POINT_MARKER_SIZE = 4;
const CREATOR_PURPOSE = "creator";

export type PointGeometry = { x: number; y: number };

export function toAnnotoriousDrawingTool(
  tool: "rectangle" | "polygon" | "point",
): AnnotoriousDrawingTool {
  return tool === "point" ? "rectangle" : tool;
}

// Our app stores a frame key on target.source (not present in upstream Annotorious types).
export type VideoImageAnnotation = ImageAnnotation & {
  target: ImageAnnotationTarget & {
    source?: string;
  };
};

export type ImageAnnotationWithSelector<TSelector extends Shape> = Omit<
  ImageAnnotation,
  "target"
> & {
  target: Omit<ImageAnnotationTarget, "selector"> & {
    selector: TSelector;
  };
};

export type RectangleAnnotation = ImageAnnotationWithSelector<Rectangle>;
export type PolygonAnnotation = ImageAnnotationWithSelector<Polygon>;
export type EllipseAnnotation = ImageAnnotationWithSelector<Ellipse>;
export type PointAnnotation = EllipseAnnotation;

function withTargetSource(
  a: ImageAnnotation,
  source: string,
): VideoImageAnnotation {
  return {
    ...(a as VideoImageAnnotation),
    target: {
      ...(a.target as VideoImageAnnotation["target"]),
      source,
    },
  };
}

function isImageAnnotationLike(v: unknown): v is ImageAnnotation {
  if (!v || typeof v !== "object") return false;

  const rec = v as UnknownRecord;
  const target = rec["target"];
  if (!target || typeof target !== "object") return false;

  const selector = (target as UnknownRecord)["selector"];
  if (!selector || typeof selector !== "object") return false;

  // Shape.type is a string enum at runtime, so just require a string here.
  const type = (selector as UnknownRecord)["type"];
  return typeof type === "string";
}

export function getBodyValue(
  a: ImageAnnotation,
  purpose: string,
): string | null {
  const v = a?.bodies?.find((b) => b?.purpose === purpose)?.value;
  return typeof v === "string" ? v : null;
}

/** Insert or replace a body by purpose (returns a new array).
 *
 * Annotorious bodies should have a unique `id`. Annotorious may auto-generate
 * missing IDs on load, but we ensure IDs here so our normalized/session state
 * is explicit and stable.
 */
export function upsertBody(
  bodies: AnnotationBody[] | undefined,
  purpose: string,
  value: string,
): AnnotationBody[] {
  const list = Array.isArray(bodies) ? bodies.slice() : [];
  const idx = list.findIndex((b) => b?.purpose === purpose);

  if (idx >= 0) {
    const prev = list[idx] as AnnotationBody;

    // Preserve existing body id if present; otherwise assign one once.
    const existingId = prev.id?.trim() ? prev.id : null;

    list[idx] = {
      ...prev,
      id: existingId ?? newBodyId(),
      purpose,
      value,
    } as AnnotationBody;

    return list;
  }

  // New body: assign an id explicitly.
  list.push({ id: newBodyId(), purpose, value } as AnnotationBody);
  return list;
}

/** Ensure the annotation has a class label body. */
export function stampLabel(
  a: ImageAnnotation,
  className: string,
): ImageAnnotation {
  const bodies = upsertBody(a.bodies, "tagging", className);
  return { ...a, bodies };
}

/** Ensure the annotation has both class label + track id bodies. */
export function stampLabelAndTrack(
  a: ImageAnnotation,
  className: string,
  trackId: string,
): ImageAnnotation {
  const bodies = upsertBody(a.bodies, "tagging", className);
  const bodies2 = upsertBody(bodies, "identifying", trackId);
  return { ...a, bodies: bodies2 };
}

/** Preserve backend creator metadata while annotations pass through Annotorious. */
export function stampCreator(
  a: ImageAnnotation,
  createdBy: string | null | undefined,
): ImageAnnotation {
  const creator = createdBy?.trim();
  if (!creator) return a;

  const bodies = upsertBody(a.bodies, CREATOR_PURPOSE, creator);
  return { ...a, bodies };
}

/** Mark a selector as the UI representation of a point. */
export function stampPoint(a: ImageAnnotation): ImageAnnotation {
  const bodies = upsertBody(a.bodies, POINT_BODY_PURPOSE, POINT_BODY_VALUE);
  return { ...a, bodies };
}

/** Read the class label + track id bodies. */
export function getLabelTrack(a: ImageAnnotation): {
  className: string | null;
  trackId: string | null;
} {
  return {
    className: getBodyValue(a, "tagging"),
    trackId: getBodyValue(a, "identifying"),
  };
}

/** Read backend creator metadata, defaulting to manual when none is stored. */
export function getAnnotationCreator(a: ImageAnnotation): string {
  return getBodyValue(a, CREATOR_PURPOSE) ?? "manual";
}

function newBodyId(): string {
  return (
    globalThis.crypto?.randomUUID?.() ??
    `body-${Math.random().toString(36).slice(2)}`
  );
}

/** True if the annotation target is a rectangle selector. */
export function isRectangleAnno(a: ImageAnnotation): a is RectangleAnnotation {
  // ShapeType.RECTANGLE is already the string literal "RECTANGLE",
  // so we don't need a separate check for the string form.
  return a.target.selector.type === ShapeType.RECTANGLE;
}

/** True if the annotation target is a polygon selector. */
export function isPolygonAnno(a: ImageAnnotation): a is PolygonAnnotation {
  return a.target.selector.type === ShapeType.POLYGON;
}

/** True if the annotation target is an ellipse selector. */
export function isEllipseAnno(a: ImageAnnotation): a is EllipseAnnotation {
  return a.target.selector.type === ShapeType.ELLIPSE;
}

/** True when a selector is being used as a point marker. */
export function isPointAnno(a: ImageAnnotation): a is PointAnnotation {
  return (
    isEllipseAnno(a) && getBodyValue(a, POINT_BODY_PURPOSE) === POINT_BODY_VALUE
  );
}

function isFiniteNumber(v: unknown): v is number {
  return Number.isFinite(v);
}

/** Read rectangle geometry (returns null if missing/invalid). */
export function readRectGeometry(
  a: RectangleAnnotation,
): RectangleGeometry | null {
  const g = a.target.selector.geometry;
  const { x, y, w, h } = g;

  // TS now narrows x/y/w/h to number after these checks.
  if (
    !isFiniteNumber(x) ||
    !isFiniteNumber(y) ||
    !isFiniteNumber(w) ||
    !isFiniteNumber(h)
  )
    return null;

  if (w <= 0 || h <= 0) return null;

  return g;
}

/** Read polygon geometry (returns null if missing/invalid). */
export function readPolygonGeometry(
  a: PolygonAnnotation,
): PolygonGeometry | null {
  const g = a.target.selector.geometry;
  const { points } = g;
  if (!Array.isArray(points) || points.length < 3) return null;

  for (const point of points) {
    if (!Array.isArray(point) || point.length !== 2) return null;

    const [x, y] = point;
    if (!isFiniteNumber(x) || !isFiniteNumber(y)) return null;
  }

  return g;
}

/** Read ellipse geometry (returns null if missing/invalid). */
export function readEllipseGeometry(
  a: EllipseAnnotation,
): EllipseGeometry | null {
  const g = a.target.selector.geometry;
  const { cx, cy, rx, ry } = g;

  if (
    !isFiniteNumber(cx) ||
    !isFiniteNumber(cy) ||
    !isFiniteNumber(rx) ||
    !isFiniteNumber(ry)
  )
    return null;

  if (rx <= 0 || ry <= 0) return null;

  return g;
}

/** Read point geometry from the tagged marker shape. */
export function readPointGeometry(a: ImageAnnotation): PointGeometry | null {
  if (!isPointAnno(a)) return null;

  const g = readEllipseGeometry(a);
  if (!g) return null;

  return {
    x: g.cx,
    y: g.cy,
  };
}

/**
 * Normalize an overlay list into a backend-safe shape:
 * - keep supported video shapes only
 * - stamp target.source with frameKey
 * - ensure class label exists (uses fallback if missing)
 * - ensure track id exists (uses fallback or allocator)
 */
export function normalizeOverlay(
  list: ImageAnnotation[],
  frameKey: string,
  fallback: { className: string | null; trackId: string | null },
  allocTrackId?: (className: string) => string,
  opts?: {
    enforceBothBodies?: boolean;
    dedupeByInstance?: boolean;
  },
): ImageAnnotation[] {
  const enforceBothBodies = opts?.enforceBothBodies ?? true;
  const dedupeByInstance = opts?.dedupeByInstance ?? true;

  // Defensive filter: keep only annotation-ish object.
  const src = list.filter(isImageAnnotationLike);
  const out: ImageAnnotation[] = [];

  for (const a of src) {
    if (!isRectangleAnno(a) && !isPolygonAnno(a) && !isPointAnno(a)) continue;

    const withSource = withTargetSource(a, frameKey);

    const got = getLabelTrack(withSource);
    const className = (got.className ?? fallback.className)?.trim() || null;
    if (!className) continue;

    let trackId = (got.trackId ?? fallback.trackId)?.trim() || null;
    if (!trackId && allocTrackId) {
      trackId = allocTrackId(className);
    }

    if (!trackId) {
      if (enforceBothBodies) continue;
      out.push(stampLabel(withSource, className));
      continue;
    }

    out.push(stampLabelAndTrack(withSource, className, trackId));
  }

  if (!dedupeByInstance) return out;

  const byKey = new Map<string, ImageAnnotation>();
  const order: string[] = [];

  for (const a of out) {
    const { className, trackId } = getLabelTrack(a);
    if (!className || !trackId) continue;
    const k = `${className}::${trackId}`;
    if (!byKey.has(k)) order.push(k);
    byKey.set(k, a);
  }

  return order.map((k) => byKey.get(k)!).filter(Boolean);
}

/** Convert a normalized ImageAnnotation -> backend VideoBoundingBox. */
export function annoToVideoBBox(
  a: RectangleAnnotation,
  frame: number,
): VideoBoundingBox | null {
  const g = readRectGeometry(a);
  if (!g) return null;

  const { className, trackId } = getLabelTrack(a);
  if (!className || !trackId) return null;

  return {
    type: "video_bounding_box",
    frame,
    track_id: String(trackId),
    label: String(className),
    class_id: classIdForName(className),
    x_min: Math.round(g.x),
    y_min: Math.round(g.y),
    width: Math.round(g.w),
    height: Math.round(g.h),
    created_by: getAnnotationCreator(a),
  };
}

/** Convert a normalized ImageAnnotation -> backend VideoPolygon. */
export function annoToVideoPolygon(
  a: PolygonAnnotation,
  frame: number,
): VideoPolygon | null {
  const g = readPolygonGeometry(a);
  if (!g) return null;

  const { className, trackId } = getLabelTrack(a);
  if (!className || !trackId) return null;

  const segmentation = g.points.flatMap(([x, y]) => [
    Math.round(x),
    Math.round(y),
  ]);

  if (segmentation.length < 6) return null;

  return {
    type: "video_polygon",
    frame,
    track_id: String(trackId),
    label: String(className),
    class_id: classIdForName(className),
    segmentation,
    created_by: getAnnotationCreator(a),
  };
}

/** Convert a normalized point annotation -> backend VideoPoint. */
export function annoToVideoPoint(
  a: ImageAnnotation,
  frame: number,
): VideoPoint | null {
  const g = readPointGeometry(a);
  if (!g) return null;

  const { className, trackId } = getLabelTrack(a);
  if (!className || !trackId) return null;

  return {
    type: "video_point",
    frame,
    track_id: String(trackId),
    label: String(className),
    class_id: classIdForName(className),
    x: Math.round(g.x),
    y: Math.round(g.y),
    created_by: getAnnotationCreator(a),
  };
}

/** Convert a normalized ImageAnnotation -> backend video annotation shape. */
export function annoToVideoAnnotation(
  a: ImageAnnotation,
  frame: number,
): VideoAnnotationShape | null {
  if (isPointAnno(a)) return annoToVideoPoint(a, frame);
  if (isRectangleAnno(a)) return annoToVideoBBox(a, frame);
  if (isPolygonAnno(a)) return annoToVideoPolygon(a, frame);
  return null;
}

/** Convert backend VideoBoundingBox -> Annotorious rectangle annotation. */
export function videoBBoxToAnno(
  b: VideoBoundingBox,
  frameKey: string,
): ImageAnnotation {
  const x = Number(b.x_min);
  const y = Number(b.y_min);
  const w = Number(b.width);
  const h = Number(b.height);

  const id =
    globalThis.crypto?.randomUUID?.() ??
    `anno-${Math.random().toString(36).slice(2)}`;

  // Annotorious Geometry type only guarantees `bounds`, but rectangle geometry
  // includes x/y/w/h at runtime. Use the concrete RectangleGeometry type.
  const geometry: RectangleGeometry = {
    x,
    y,
    w,
    h,
    bounds: { minX: x, minY: y, maxX: x + w, maxY: y + h },
  };

  const selector: Rectangle = {
    type: ShapeType.RECTANGLE,
    geometry,
  };

  const anno: VideoImageAnnotation = {
    id,
    bodies: [],
    target: {
      // Required by the upstream type; links target -> parent annotation id.
      annotation: id,
      source: frameKey,
      selector,
    },
  };

  const labelled = stampLabelAndTrack(anno, b.label, String(b.track_id));
  return stampCreator(labelled, b.created_by);
}

/** Convert backend VideoPolygon -> Annotorious polygon annotation. */
export function videoPolygonToAnno(
  p: VideoPolygon,
  frameKey: string,
): ImageAnnotation {
  const points: [number, number][] = [];

  for (let i = 0; i < p.segmentation.length - 1; i += 2) {
    const x = Number(p.segmentation[i]);
    const y = Number(p.segmentation[i + 1]);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    points.push([x, y]);
  }

  const id =
    globalThis.crypto?.randomUUID?.() ??
    `anno-${Math.random().toString(36).slice(2)}`;

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (const [x, y] of points) {
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
  }

  const geometry: PolygonGeometry = {
    points,
    bounds: { minX, minY, maxX, maxY },
  };

  const selector: Polygon = {
    type: ShapeType.POLYGON,
    geometry,
  };

  const anno: VideoImageAnnotation = {
    id,
    bodies: [],
    target: {
      annotation: id,
      source: frameKey,
      selector,
    },
  };

  const labelled = stampLabelAndTrack(anno, p.label, String(p.track_id));
  return stampCreator(labelled, p.created_by);
}

/** Convert backend VideoPoint -> tagged Annotorious circle marker. */
export function videoPointToAnno(
  p: VideoPoint,
  frameKey: string,
): ImageAnnotation {
  const x = Number(p.x);
  const y = Number(p.y);
  const half = POINT_MARKER_SIZE / 2;

  const id =
    globalThis.crypto?.randomUUID?.() ??
    `anno-${Math.random().toString(36).slice(2)}`;

  const geometry: EllipseGeometry = {
    cx: x,
    cy: y,
    rx: half,
    ry: half,
    bounds: {
      minX: x - half,
      minY: y - half,
      maxX: x + half,
      maxY: y + half,
    },
  };

  const selector: Ellipse = {
    type: ShapeType.ELLIPSE,
    geometry,
  };

  const anno: VideoImageAnnotation = {
    id,
    bodies: [],
    target: {
      annotation: id,
      source: frameKey,
      selector,
    },
  };

  const labelled = stampLabelAndTrack(anno, p.label, String(p.track_id));
  return stampPoint(stampCreator(labelled, p.created_by));
}

/**
 * Normalizes a raw overlay list into our session invariants:
 * - supported video shapes only
 * - stamps the frame source key
 * - ensures class/track bodies exist (allocating track ids when needed)
 * - optional per-instance de-duplication within the frame
 *
 * The caller is responsible for writing the returned list into session state.
 */
export function normalizeOverlayForSession(args: {
  raw: ImageAnnotation[];
  frameKey: string;
  fallback: { className: string | null; trackId: string | null };
  allocTrackId?: (className: string) => string;
  enforceBothBodies?: boolean;
  dedupeByInstance?: boolean;
}): ImageAnnotation[] {
  return normalizeOverlay(
    args.raw,
    args.frameKey,
    args.fallback,
    args.allocTrackId,
    {
      enforceBothBodies: args.enforceBothBodies ?? true,
      dedupeByInstance: args.dedupeByInstance ?? true,
    },
  );
}
