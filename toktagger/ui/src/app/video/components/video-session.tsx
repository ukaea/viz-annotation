"use client";

import React, {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  useEffect,
  useLayoutEffect,
  useRef,
} from "react";
import {
  useAnnotator,
  type AnnotoriousOpenSeadragonAnnotator,
  type ImageAnnotation,
} from "@annotorious/react";

import type { Annotation } from "@/types";
import { useSample } from "@/app/contexts/SampleContext";
import { useVideoUiState } from "@/app/video/components/video-context";
import {
  VideoBoundingBoxSchema,
  VideoPointSchema,
  VideoPolygonSchema,
} from "@/types";
import type {
  ByFrameMap,
  DrawingTool,
  FrameIndex,
  InstanceProfile,
  Selection,
  VideoBoundingBox,
  VideoPoint,
  VideoPolygon,
} from "./types";
import { buildSourceKey } from "./types";
import {
  allocateNextTrackId,
  buildNextTrackIdState,
  deleteTrackAcrossFrames,
  canonicalizeTrackId,
  deriveInstances,
  forwardPropagateIfEmpty,
  mapClearAll,
  mapClearFrame,
  mapSetFrame,
  existingTrackIdsForClass,
  uniqueReadableTrackId,
} from "./video-utils";
import {
  annoToVideoAnnotation,
  getLabelTrack,
  videoBBoxToAnno,
  videoPointToAnno,
  videoPolygonToAnno,
  stampPoint,
  stampLabelAndTrack,
  normalizeOverlayForSession,
  toAnnotoriousDrawingTool,
} from "./anno-utils";
import { clampOverlayToNaturalImage, sameOverlay } from "./overlay-sync-utils";

/**
 * Session state for the frame-by-frame annotation workflow.
 * Mirrors SampleContext annotations into per-frame overlays and keeps selection,
 * drawing mode, forward propagation, and Annotorious event wiring in one place.
 */
type VideoSessionCtx = {
  projectId: string;
  sampleId: string;

  frame: FrameIndex;
  setFrame: (n: FrameIndex) => void;

  /** Stable source key for the current frame (used as target.source on annotations). */
  frameKey: string;

  /** Per-frame editor cache derived from SampleContext annotations. */
  byFrame: ByFrameMap;

  /** True if video edits have changed context annotations since the last save. */
  dirty: boolean;
  markSaved: () => void;

  /** Current "armed" selection for drawing and instance operations. */
  selection: Selection;
  setSelection: (next: Selection) => void;

  /** Derived instance summary across all frames (used by sidebar UI). */
  instances: InstanceProfile[];

  drawingTool: DrawingTool;
  setDrawingTool: (tool: DrawingTool) => void;
  /** When true, drawing is disabled and frame drag/pan is enabled. */
  panMode: boolean;
  setPanMode: (v: boolean) => void;
  propagate: boolean;
  setPropagate: (v: boolean) => void;
  hideAnnotations: boolean;
  setHideAnnotations: (v: boolean) => void;

  /** Natural image dimensions for the currently loaded frame (used for clamping). */
  imageNatural: { w: number; h: number } | null;
  setImageNatural: (n: { w: number; h: number } | null) => void;

  /** Popup helpers so view components don't need direct access to the Annotorious API. */
  createPointAnnotation: (point: { x: number; y: number }) => void;
  deleteAnnotation: (id: string) => void;
  closePopup: () => void;
  requestFocusInstance: (
    className: string,
    trackId: string,
    opts?: { onlyIfOnCurrentFrame?: boolean; targetFrame?: FrameIndex },
  ) => void;

  // frame ops
  getFrameList: (frame: FrameIndex) => ImageAnnotation[];
  /** Commit the current Annotorious overlay into SampleContext before frame navigation. */
  flushCurrentFrameOverlay: () => void;
  clearCurrentFrame: () => void;
  clearAllFrames: () => void;

  // instance ops
  /** Allocate/select the next track id for a class (used by "new instance" flows). */
  createNewInstanceForClass: (className: string) => {
    className: string;
    trackId: string;
  };

  /**
   * Delete a specific instance across all frames (does NOT rely on selection state).
   * This avoids "needs two clicks" bugs when callers first set selection then delete.
   */
  deleteInstanceAcrossFrames: (className: string, trackId: string) => void;

  /** Delete the currently selected instance across all frames. */
  deleteSelectedInstanceAcrossFrames: () => void;

  // forward propagation
  /** Seed next frame with current overlay if the next frame has no annotations. */
  forwardPropToNextIfEmpty: (nextFrame: FrameIndex) => void;
};

const Ctx = createContext<VideoSessionCtx | null>(null);

type FocusRequest = {
  className: string;
  trackId: string;
  onlyIfOnCurrentFrame: boolean;
  targetFrame: FrameIndex | null;
};

function parseVideoAnnotation(annotation: Annotation) {
  if (annotation.type === "video_bounding_box") {
    return VideoBoundingBoxSchema.safeParse(annotation);
  }

  if (annotation.type === "video_polygon") {
    return VideoPolygonSchema.safeParse(annotation);
  }

  if (annotation.type === "video_point") {
    return VideoPointSchema.safeParse(annotation);
  }

  return null;
}

function videoAnnotationDedupeKey(annotation: Annotation): string | null {
  const parsed = parseVideoAnnotation(annotation);
  if (!parsed?.success) return null;

  const item = parsed.data;
  const label = (item.label ?? "").trim();
  const trackId = canonicalizeTrackId(item.track_id ?? "");
  if (!label || !trackId) return null;

  return `${item.frame}::${label}::${trackId}`;
}

function dedupeVideoAnnotations(annotations: Annotation[]): {
  annotations: Annotation[];
  duplicates: number;
} {
  const nonVideoAnnotations: Annotation[] = [];
  const videoAnnotationsByKey = new Map<string, Annotation>();
  const videoKeys: string[] = [];
  let duplicates = 0;

  for (const annotation of annotations ?? []) {
    if (!isVideoAnnotationType(annotation)) {
      nonVideoAnnotations.push(annotation);
      continue;
    }

    const key = videoAnnotationDedupeKey(annotation);
    if (!key) {
      nonVideoAnnotations.push(annotation);
      continue;
    }

    if (!videoAnnotationsByKey.has(key)) {
      videoKeys.push(key);
    } else {
      duplicates += 1;
    }

    // Last write wins, matching normalizeOverlay's per-frame instance behavior.
    videoAnnotationsByKey.set(key, annotation);
  }

  return {
    annotations: [
      ...nonVideoAnnotations,
      ...videoKeys
        .map((key) => videoAnnotationsByKey.get(key))
        .filter((annotation): annotation is Annotation => Boolean(annotation)),
    ],
    duplicates,
  };
}

function videoAnnotationSignature(annotations: Annotation[]): string {
  const entries: string[] = [];

  for (const annotation of annotations ?? []) {
    const parsed = parseVideoAnnotation(annotation);
    if (!parsed?.success) continue;

    const item = parsed.data;
    if (item.type === "video_bounding_box") {
      entries.push(
        JSON.stringify({
          type: item.type,
          frame: item.frame,
          track_id: item.track_id,
          label: item.label,
          x_min: item.x_min,
          y_min: item.y_min,
          width: item.width,
          height: item.height,
          created_by: item.created_by,
          timestamp: item.timestamp,
        }),
      );
      continue;
    }

    if (item.type === "video_polygon") {
      entries.push(
        JSON.stringify({
          type: item.type,
          frame: item.frame,
          track_id: item.track_id,
          label: item.label,
          segmentation: item.segmentation,
          created_by: item.created_by,
          timestamp: item.timestamp,
        }),
      );
      continue;
    }

    if (item.type === "video_point") {
      entries.push(
        JSON.stringify({
          type: item.type,
          frame: item.frame,
          track_id: item.track_id,
          label: item.label,
          x: item.x,
          y: item.y,
          created_by: item.created_by,
          timestamp: item.timestamp,
        }),
      );
    }
  }

  return entries.sort().join("\n");
}

function videoAnnotationsToByFrame(args: {
  dbAnnotations: Annotation[];
  projectId: string;
  sampleId: string;
}): ByFrameMap {
  const byFrame = new Map<number, ImageAnnotation[]>();

  for (const annotation of args.dbAnnotations ?? []) {
    const parsed = parseVideoAnnotation(annotation);

    if (!parsed) continue;

    if (!parsed.success) continue;

    const dbAnno = parsed.data;
    const key = buildSourceKey({
      projectId: args.projectId,
      sampleId: args.sampleId,
      frame: dbAnno.frame,
    });

    let anno: ImageAnnotation | null = null;
    if (dbAnno.type === "video_bounding_box") {
      anno = videoBBoxToAnno(dbAnno as VideoBoundingBox, key);
    } else if (dbAnno.type === "video_polygon") {
      anno = videoPolygonToAnno(dbAnno as VideoPolygon, key);
    } else if (dbAnno.type === "video_point") {
      anno = videoPointToAnno(dbAnno as VideoPoint, key);
    }
    if (!anno) continue;

    const cur = byFrame.get(dbAnno.frame) ?? [];
    cur.push(anno);
    byFrame.set(dbAnno.frame, cur);
  }

  return byFrame;
}

function isVideoAnnotationType(annotation: Annotation): boolean {
  return (
    annotation.type === "video_bounding_box" ||
    annotation.type === "video_polygon" ||
    annotation.type === "video_point"
  );
}

function videoAnnotationsFromByFrame(byFrame: ByFrameMap): Annotation[] {
  const out: Annotation[] = [];

  for (const [frame, list] of byFrame.entries()) {
    for (const annotation of list ?? []) {
      const shape = annoToVideoAnnotation(annotation, frame);
      if (!shape) continue;

      let parsed:
        | ReturnType<typeof VideoBoundingBoxSchema.safeParse>
        | ReturnType<typeof VideoPointSchema.safeParse>
        | ReturnType<typeof VideoPolygonSchema.safeParse>
        | null = null;
      if (shape.type === "video_bounding_box") {
        parsed = VideoBoundingBoxSchema.safeParse(shape);
      } else if (shape.type === "video_polygon") {
        parsed = VideoPolygonSchema.safeParse(shape);
      } else if (shape.type === "video_point") {
        parsed = VideoPointSchema.safeParse(shape);
      }

      if (parsed?.success) out.push(parsed.data);
    }
  }

  return out;
}

function replaceVideoAnnotations(
  annotations: Annotation[],
  byFrame: ByFrameMap,
): Annotation[] {
  return [
    ...(annotations ?? []).filter(
      (annotation) => !isVideoAnnotationType(annotation),
    ),
    ...videoAnnotationsFromByFrame(byFrame),
  ];
}

export function useVideoSession(): VideoSessionCtx {
  const v = useContext(Ctx);
  if (!v)
    throw new Error(
      "useVideoSession must be used within <VideoSessionProvider>",
    );
  return v;
}

/**
 * Provides shared session state for the video annotation UI.
 * This provider mirrors SampleContext annotations into Annotorious-friendly
 * per-frame state and writes video edits back to SampleContext.
 */
export function VideoSessionProvider(props: {
  projectId: string;
  sampleId: string;
  propagate: boolean;
  setPropagate: (v: boolean) => void;
  children: React.ReactNode;
}) {
  const { projectId, sampleId, propagate, setPropagate, children } = props;
  const {
    data,
    dataParams,
    annotations,
    setAnnotations: setSampleAnnotations,
  } = useSample();
  const {
    videoPanMode,
    setVideoPanMode,
    videoDrawingTool,
    setVideoDrawingTool,
  } = useVideoUiState();

  const api = useAnnotator<AnnotoriousOpenSeadragonAnnotator>();

  /**
   * Annotorious emits create/update/delete events even when annotations are set
   * programmatically (e.g. when we sync the overlay on frame change).
   *
   * This ref guards against re-entrant feedback loops during those programmatic writes.
   * SampleContext annotations remain the source of truth; `byFrame` is the editor cache.
   */
  const isProgrammaticAnnoSyncRef = useRef(false);
  const commitFromAnnotoriousRef = useRef<
    (rawOverride?: ImageAnnotation[]) => void
  >(() => {});
  const pendingFocusRef = useRef<FocusRequest | null>(null);
  const nextTrackNumsRef = useRef<Map<string, number>>(
    buildNextTrackIdState(annotations),
  );
  const lastExternalAnnotationSignatureRef = useRef<string | null>(null);
  const lastLocalAnnotationSignatureRef = useRef<string | null>(null);

  const [imageNatural, setImageNatural] = useState<{
    w: number;
    h: number;
  } | null>(null);
  const [focusNonce, setFocusNonce] = useState(0);

  const [videoFrame, setVideoFrame] = useState<number | null>(null);

  // For video we assume backend returns { frame: number, values: base64, ... }
  const frameFromBackend = useMemo(() => {
    if (!data) return 0;
    const maybe = data as { frame?: number };
    return maybe?.frame ?? 0;
  }, [data]);

  useLayoutEffect(() => {
    if (!data) return;

    const dp = dataParams as {
      name?: string;
      frame?: number | null;
    };

    if (dp.name === "image" && dp.frame != null) {
      if (frameFromBackend !== dp.frame) return;
    }

    setVideoFrame(frameFromBackend);
  }, [data, dataParams, frameFromBackend]);

  const frame: FrameIndex = (videoFrame ?? frameFromBackend) as FrameIndex;

  const setFrame = useCallback((n: FrameIndex) => {
    setVideoFrame(n);
  }, []);

  const [byFrame, setByFrame] = useState<ByFrameMap>(() => new Map());
  const byFrameRef = useRef<ByFrameMap>(new Map());
  const [dirty, setDirty] = useState(false);

  const [selection, setSelectionState] = useState<Selection>({
    className: null,
    trackId: null,
    source: null,
  });
  const [drawingTool, setDrawingToolState] =
    useState<DrawingTool>(videoDrawingTool);
  const [panMode, setPanModeState] = useState(videoPanMode);
  const [hideAnnotations, setHideAnnotationsState] = useState(false);
  const hideAnnotationsRef = useRef(false);

  const setHideAnnotations = useCallback(
    (v: boolean) => {
      hideAnnotationsRef.current = v;
      setHideAnnotationsState(v);

      if (v) {
        setPanModeState(true);
        setVideoPanMode(true);
      }
    },
    [setVideoPanMode],
  );

  const frameKey = useMemo(
    () => buildSourceKey({ projectId, sampleId, frame }),
    [projectId, sampleId, frame],
  );

  // Instances are derived from byFrame so UI can render a stable sidebar list.
  const instances = useMemo(
    () => deriveInstances(byFrame, getLabelTrack),
    [byFrame],
  );

  const getFrameList = useCallback(
    (f: FrameIndex) => byFrame.get(f) ?? [],
    [byFrame],
  );

  useEffect(() => {
    byFrameRef.current = byFrame;
  }, [byFrame]);

  // Keep the editor cache synchronized with SampleContext.annotations.
  useEffect(() => {
    const { annotations: dbAnnotations, duplicates } =
      dedupeVideoAnnotations(annotations);
    const signature = videoAnnotationSignature(dbAnnotations);

    if (signature === lastExternalAnnotationSignatureRef.current) return;
    if (signature === lastLocalAnnotationSignatureRef.current) {
      nextTrackNumsRef.current = buildNextTrackIdState(dbAnnotations);
      lastExternalAnnotationSignatureRef.current = signature;
      return;
    }

    const nextByFrame = videoAnnotationsToByFrame({
      dbAnnotations,
      projectId,
      sampleId,
    });

    pendingFocusRef.current = null;
    byFrameRef.current = nextByFrame;
    setByFrame(nextByFrame);
    nextTrackNumsRef.current = buildNextTrackIdState(dbAnnotations);
    lastExternalAnnotationSignatureRef.current = signature;
    lastLocalAnnotationSignatureRef.current = null;
    if (duplicates > 0) {
      setSampleAnnotations(() => dbAnnotations);
    }
  }, [annotations, projectId, sampleId, setSampleAnnotations]);

  const commitByFrame = useCallback(
    (next: ByFrameMap, opts?: { markDirty?: boolean }) => {
      byFrameRef.current = next;
      setByFrame(next);
      setSampleAnnotations((prev) => {
        const nextAnnotations = replaceVideoAnnotations(prev, next);
        lastLocalAnnotationSignatureRef.current =
          videoAnnotationSignature(nextAnnotations);
        return nextAnnotations;
      });
      if (opts?.markDirty) setDirty(true);
    },
    [setSampleAnnotations],
  );

  const updateByFrame = useCallback(
    (
      updater: (prev: ByFrameMap) => ByFrameMap,
      opts?: { markDirty?: boolean },
    ) => {
      const prev = byFrameRef.current;
      const next = updater(prev);
      if (next === prev) return;
      commitByFrame(next, opts);
    },
    [commitByFrame],
  );

  const markSaved = useCallback(() => setDirty(false), []);

  const setSelection = useCallback((next: Selection) => {
    setSelectionState(next);
  }, []);

  const finishProgrammaticAnnotationSync = useCallback(() => {
    requestAnimationFrame(() => {
      isProgrammaticAnnoSyncRef.current = false;
    });
  }, []);

  const flushPendingOverlay = useCallback(() => {
    if (isProgrammaticAnnoSyncRef.current) return;
    if (hideAnnotationsRef.current) return;
    const raw = api?.getAnnotations?.();
    if (!raw) return;
    commitFromAnnotoriousRef.current(raw);
  }, [api]);

  const flushCurrentFrameOverlay = useCallback(() => {
    api?.setSelected?.();
    flushPendingOverlay();
  }, [api, flushPendingOverlay]);

  const setDrawingTool = useCallback(
    (tool: DrawingTool) => {
      api?.setSelected?.();
      flushPendingOverlay();
      setDrawingToolState(tool);
      setVideoDrawingTool(tool);
    },
    [api, flushPendingOverlay, setVideoDrawingTool],
  );

  const setPanMode = useCallback(
    (v: boolean) => {
      api?.setSelected?.();
      flushPendingOverlay();
      setPanModeState(v);
      setVideoPanMode(v);
    },
    [api, flushPendingOverlay, setVideoPanMode],
  );

  const clearCurrentFrame = useCallback(() => {
    api?.setSelected?.();
    flushPendingOverlay();
    pendingFocusRef.current = null;

    updateByFrame((prev) => mapClearFrame(prev, frame), { markDirty: true });

    isProgrammaticAnnoSyncRef.current = true;
    try {
      api?.cancelDrawing?.();
      api?.setSelected?.();
      api?.setAnnotations?.([], true);
    } finally {
      finishProgrammaticAnnotationSync();
    }
  }, [
    api,
    finishProgrammaticAnnotationSync,
    flushPendingOverlay,
    frame,
    updateByFrame,
  ]);

  const clearAllFrames = useCallback(() => {
    api?.setSelected?.();
    flushPendingOverlay();
    pendingFocusRef.current = null;

    updateByFrame((prev) => mapClearAll(prev), { markDirty: true });

    isProgrammaticAnnoSyncRef.current = true;
    try {
      api?.cancelDrawing?.();
      api?.setSelected?.();
      api?.setAnnotations?.([], true);
    } finally {
      finishProgrammaticAnnotationSync();
    }
  }, [
    api,
    finishProgrammaticAnnotationSync,
    flushPendingOverlay,
    updateByFrame,
  ]);

  const applyAnnotatorInteractionMode = useCallback(() => {
    if (!api) return;

    const hasSelected = (api.getSelected?.() ?? []).length > 0;
    const canDraw =
      !panMode &&
      !hideAnnotations &&
      Boolean(selection.className) &&
      !hasSelected &&
      drawingTool !== "point";
    api.setDrawingTool(toAnnotoriousDrawingTool(drawingTool));
    api.setDrawingEnabled(canDraw);

    if (!canDraw) {
      api.cancelDrawing?.();
    }
  }, [api, drawingTool, hideAnnotations, panMode, selection.className]);

  const createNewInstanceForClass = useCallback((className: string) => {
    const cname = (className || "").trim();
    const trackId = allocateNextTrackId(nextTrackNumsRef.current, cname);

    setSelectionState({ className: cname, trackId, source: "auto" });
    return { className: cname, trackId };
  }, []);

  /**
   * Delete a specific (className, trackId) across all frames.
   * IMPORTANT: does not depend on selection being updated first.
   */
  const deleteInstanceAcrossFrames = useCallback(
    (className: string, trackId: string) => {
      const cls = (className || "").trim();
      const tid = canonicalizeTrackId(trackId || "");
      if (!cls || !tid) return;

      updateByFrame(
        (prev) =>
          deleteTrackAcrossFrames(
            prev,
            { className: cls, trackId: tid },
            getLabelTrack,
          ),
        { markDirty: true },
      );

      // If the deleted instance is currently selected, clear selection.trackId
      setSelectionState((prev) => {
        const prevCls = (prev.className || "").trim();
        const prevTid = canonicalizeTrackId(prev.trackId || "");
        if (prevCls === cls && prevTid === tid) {
          return { ...prev, trackId: null };
        }
        return prev;
      });
    },
    [updateByFrame],
  );

  /**
   * Backwards-compatible helper: delete the currently selected instance.
   * Now implemented via deleteInstanceAcrossFrames to avoid stale selection issues.
   */
  const deleteSelectedInstanceAcrossFrames = useCallback(() => {
    if (!selection.className || !selection.trackId) return;
    deleteInstanceAcrossFrames(selection.className, selection.trackId);
  }, [deleteInstanceAcrossFrames, selection.className, selection.trackId]);

  /**
   * Copies current frame annotations into `nextFrame` if it's empty.
   * Any copied annotations get their target.source updated to match the destination frame.
   */
  const forwardPropToNextIfEmpty = useCallback(
    (nextFrame: FrameIndex) => {
      updateByFrame(
        (prev) =>
          forwardPropagateIfEmpty(prev, frame, nextFrame, {
            projectId,
            sampleId,
          }),
        { markDirty: true },
      );
    },
    [frame, projectId, sampleId, updateByFrame],
  );

  const createPointAnnotation = useCallback(
    (point: { x: number; y: number }) => {
      if (!api?.getAnnotations) return;

      const cls = (selection.className ?? "").trim();
      if (!cls) return;

      const rawX = Number(point.x);
      const rawY = Number(point.y);
      if (!Number.isFinite(rawX) || !Number.isFinite(rawY)) return;

      const viewerNatural = (() => {
        const item = api.viewer?.world?.getItemAt?.(0);
        if (!item) return null;
        const size = item.getContentSize?.();
        const w = Math.round(Number(size?.x ?? 0));
        const h = Math.round(Number(size?.y ?? 0));
        if (!(w > 0 && h > 0)) return null;
        return { w, h };
      })();
      const natural = imageNatural ?? viewerNatural;
      const x = natural ? Math.max(0, Math.min(natural.w, rawX)) : rawX;
      const y = natural ? Math.max(0, Math.min(natural.h, rawY)) : rawY;

      const raw = api.getAnnotations();
      const used = new Set<string>();

      for (const tid of existingTrackIdsForClass(byFrameRef.current, cls)) {
        const c = canonicalizeTrackId(tid);
        if (c) used.add(c);
      }

      for (const annotation of raw) {
        const got = getLabelTrack(annotation);
        if ((got.className ?? "").trim() !== cls) continue;

        const tid = canonicalizeTrackId(got.trackId ?? "");
        if (tid) used.add(tid);
      }

      const trackId = selection.trackId ?? uniqueReadableTrackId(used);
      const dbPoint: VideoPoint = {
        type: "video_point",
        frame,
        track_id: String(trackId),
        label: cls,
        x: Math.round(x),
        y: Math.round(y),
        created_by: "manual",
      };
      const pointAnnotation = videoPointToAnno(dbPoint, frameKey);
      const normalized = normalizeOverlayForSession({
        raw: [...raw, pointAnnotation],
        frameKey,
        fallback: { className: cls, trackId: String(trackId) },
        enforceBothBodies: true,
        dedupeByInstance: true,
      });

      updateByFrame(
        (prevByFrame) => mapSetFrame(prevByFrame, frame, normalized),
        { markDirty: true },
      );

      isProgrammaticAnnoSyncRef.current = true;
      try {
        api.cancelDrawing?.();
        api.setSelected?.();
        api.setAnnotations?.(normalized, true);
        applyAnnotatorInteractionMode();
      } finally {
        finishProgrammaticAnnotationSync();
      }
    },
    [
      api,
      applyAnnotatorInteractionMode,
      finishProgrammaticAnnotationSync,
      frame,
      frameKey,
      imageNatural,
      selection.className,
      selection.trackId,
      updateByFrame,
    ],
  );

  /**
   * Single commit point for all Annotorious mutations (create/update/delete).
   * This normalizes bodies, clamps geometry to image bounds, and syncs the session store.
   */
  const commitFromAnnotorious = useCallback(
    (rawOverride?: ImageAnnotation[]) => {
      if (!api?.getAnnotations) return;
      if (isProgrammaticAnnoSyncRef.current) return;
      if (hideAnnotationsRef.current) return;

      // Some Annotorious events pass a single annotation (not an array).
      const raw = rawOverride ?? api.getAnnotations();

      // Enforce image bounds so shapes can't persist outside the frame.
      // In OSD mode, imageNatural can occasionally lag; fall back to viewer content size.
      const viewerNatural = (() => {
        const item = api.viewer?.world?.getItemAt?.(0);
        if (!item) return null;
        const size = item.getContentSize?.();
        const w = Math.round(Number(size?.x ?? 0));
        const h = Math.round(Number(size?.y ?? 0));
        if (!(w > 0 && h > 0)) return null;
        return { w, h };
      })();

      const clampNatural = imageNatural ?? viewerNatural;
      const clamped = clampOverlayToNaturalImage(raw, clampNatural);

      const firstClassFrom = (list: ImageAnnotation[]) => {
        for (const a of list ?? []) {
          const { className } = getLabelTrack(a);
          const s = (className ?? "").trim();
          if (s) return s;
        }
        return null;
      };

      const cls = selection.className ?? firstClassFrom(clamped);

      const fallbackTrackId = selection.trackId ?? null;

      // Allocator is only needed within THIS normalization pass.
      // It allocates unique ids per-class for annotations missing trackId.
      const allocTrackId = fallbackTrackId
        ? undefined
        : (() => {
            const usedByClass = new Map<string, Set<string>>();

            return (className: string) => {
              const c = (className || "").trim() || "UFO";

              let used = usedByClass.get(c);
              if (!used) {
                const existing = existingTrackIdsForClass(
                  byFrameRef.current,
                  c,
                );
                used = new Set(existing.map((t) => canonicalizeTrackId(t)));
                usedByClass.set(c, used);
              }

              const next = uniqueReadableTrackId(used);
              used.add(canonicalizeTrackId(next));
              return next;
            };
          })();

      const normalized = normalizeOverlayForSession({
        raw: clamped,
        frameKey,
        fallback: { className: cls, trackId: fallbackTrackId },
        allocTrackId,
        enforceBothBodies: true,
        dedupeByInstance: true,
      });

      // Push corrected overlay back into Annotorious when it diverges.
      if (!sameOverlay(raw, normalized)) {
        isProgrammaticAnnoSyncRef.current = true;
        try {
          const rawById = new Map(
            raw
              .filter((annotation) => annotation.id)
              .map((annotation) => [String(annotation.id), annotation]),
          );
          const canPatchInPlace =
            Boolean(api.updateAnnotation) &&
            raw.length === normalized.length &&
            normalized.every((annotation) =>
              annotation.id ? rawById.has(String(annotation.id)) : false,
            );

          if (canPatchInPlace) {
            for (const annotation of normalized) {
              const prev = rawById.get(String(annotation.id));
              if (!prev || sameOverlay([prev], [annotation])) continue;
              api.updateAnnotation?.(annotation);
            }
          } else {
            api.setSelected?.();
            api.setAnnotations?.(normalized, true);
          }

          applyAnnotatorInteractionMode();
        } finally {
          finishProgrammaticAnnotationSync();
        }
      }

      // Persist normalized overlay for this frame in the session store.
      const prev = getFrameList(frame);
      if (!sameOverlay(prev, normalized)) {
        updateByFrame(
          (prevByFrame) => mapSetFrame(prevByFrame, frame, normalized),
          { markDirty: true },
        );
      }
    },
    [
      api,
      finishProgrammaticAnnotationSync,
      frame,
      frameKey,
      getFrameList,
      imageNatural,
      selection.className,
      selection.trackId,
      applyAnnotatorInteractionMode,
      updateByFrame,
    ],
  );

  useEffect(() => {
    commitFromAnnotoriousRef.current = commitFromAnnotorious;
  }, [commitFromAnnotorious]);

  /**
   * Keep the Annotorious overlay in sync with the per-frame editor cache.
   *
   * Annotorious maintains its own internal annotation state and does not automatically
   * swap overlays when our notion of "current frame" changes (or when session state
   * changes due to context sync, forward-prop, clear/delete actions).
   *
   * So when either:
   *  - the active frame changes, OR
   *  - the session overlay for the active frame changes,
   * we push the session overlay into Annotorious.
   */
  const desiredOverlay = useMemo(() => {
    if (hideAnnotations) return [];
    return byFrame.get(frame) ?? [];
  }, [byFrame, frame, hideAnnotations]);

  const overlayHasInstance = useCallback(
    (list: ImageAnnotation[], req: FocusRequest) => {
      const wantClass = (req.className || "").trim();
      const wantTrackId = canonicalizeTrackId(req.trackId || "");
      if (!wantClass || !wantTrackId) return false;

      return list.some((annotation) => {
        const got = getLabelTrack(annotation);
        return (
          (got.className ?? "").trim() === wantClass &&
          canonicalizeTrackId(got.trackId ?? "") === wantTrackId
        );
      });
    },
    [],
  );

  const selectAnnotationById = useCallback(
    (id: string) => {
      if (!id) return false;
      if (!api) return false;

      api.setSelected(id, !panMode);
      return true;
    },
    [api, panMode],
  );

  const tryFocusPending = useCallback(
    (rawOverride?: ImageAnnotation[]) => {
      if (!api) return false;

      const pending = pendingFocusRef.current;
      if (!pending) return false;
      if (pending.targetFrame != null && frame !== pending.targetFrame) {
        return false;
      }

      const raw = rawOverride ?? api.getAnnotations();
      const hit = raw.find((annotation) => {
        const got = getLabelTrack(annotation);
        return (
          (got.className ?? "").trim() === pending.className &&
          canonicalizeTrackId(got.trackId ?? "") === pending.trackId
        );
      });

      if (!hit?.id) {
        if (
          pending.onlyIfOnCurrentFrame &&
          !overlayHasInstance(desiredOverlay, pending)
        ) {
          pendingFocusRef.current = null;
        }
        if (
          pending.targetFrame != null &&
          frame === pending.targetFrame &&
          !overlayHasInstance(desiredOverlay, pending)
        ) {
          pendingFocusRef.current = null;
        }
        return false;
      }

      if (!selectAnnotationById(hit.id)) return false;

      pendingFocusRef.current = null;
      return true;
    },
    [api, desiredOverlay, frame, overlayHasInstance, selectAnnotationById],
  );

  const requestFocusInstance = useCallback(
    (
      className: string,
      trackId: string,
      opts?: { onlyIfOnCurrentFrame?: boolean; targetFrame?: FrameIndex },
    ) => {
      const cls = (className || "").trim();
      const tid = canonicalizeTrackId(trackId || "");

      if (!cls || !tid) {
        pendingFocusRef.current = null;
        return;
      }

      pendingFocusRef.current = {
        className: cls,
        trackId: tid,
        onlyIfOnCurrentFrame: Boolean(opts?.onlyIfOnCurrentFrame),
        targetFrame:
          typeof opts?.targetFrame === "number" &&
          Number.isFinite(opts.targetFrame)
            ? opts.targetFrame
            : null,
      };
      setFocusNonce((n) => n + 1);
    },
    [],
  );

  // Required integration boundary: Annotorious keeps its own internal overlay
  // state, so we must push session state on frame/overlay changes.
  useEffect(() => {
    if (!api) return;

    let rafId: number | null = null;
    const cur = api.getAnnotations();
    if (sameOverlay(cur, desiredOverlay)) {
      applyAnnotatorInteractionMode();
      rafId = requestAnimationFrame(() => {
        tryFocusPending();
      });
      return () => {
        if (rafId != null) cancelAnimationFrame(rafId);
      };
    }

    isProgrammaticAnnoSyncRef.current = true;
    try {
      // Clear selection so popup closes when switching frames / overlays
      api.setSelected();
      api.setAnnotations(desiredOverlay, true);
      applyAnnotatorInteractionMode();
      rafId = requestAnimationFrame(() => {
        tryFocusPending();
      });
    } finally {
      finishProgrammaticAnnotationSync();
    }
    return () => {
      if (rafId != null) cancelAnimationFrame(rafId);
    };
  }, [
    api,
    applyAnnotatorInteractionMode,
    desiredOverlay,
    finishProgrammaticAnnotationSync,
    tryFocusPending,
  ]);

  /**
   * Event wiring:
   * - create/update/delete all funnel through commitFromAnnotorious
   * - selectionChanged kept only for the "deselect commits" behavior
   */
  useEffect(() => {
    if (!api?.on || !api?.off || !api?.getAnnotations) return;

    const onClickAnnotation = (
      clicked: ImageAnnotation,
      _originalEvent: PointerEvent,
    ) => {
      if (isProgrammaticAnnoSyncRef.current) return;
      if (hideAnnotations) return;

      const id = clicked?.id;
      if (id) {
        api.setSelected(id, !panMode);
      }

      const got = getLabelTrack(clicked);
      const className = (got.className ?? "").trim();
      const trackId = canonicalizeTrackId(got.trackId ?? "");
      if (className) {
        setSelectionState({
          className,
          trackId: trackId || null,
          source: "explicit",
        });
      }

      // While an annotation is selected, prioritize reshape/move over new drawing.
      api.cancelDrawing?.();
      api.setDrawingEnabled(false);
    };

    const onSelectionChanged = (arr: ImageAnnotation[]) => {
      if (isProgrammaticAnnoSyncRef.current) return;

      if (arr.length > 0) {
        if (!hideAnnotations) {
          const selected = arr[0];
          const got = getLabelTrack(selected);
          const className = (got.className ?? "").trim();
          const trackId = canonicalizeTrackId(got.trackId ?? "");

          if (className) {
            setSelectionState({
              className,
              trackId: trackId || null,
              source: "explicit",
            });
          }

          // While selected, keep drawing off so edit handles work predictably.
          api.cancelDrawing?.();
          api.setDrawingEnabled(false);
        }
        return;
      }

      if (arr.length === 0) {
        if (hideAnnotations) {
          setSelectionState((prev) =>
            prev.trackId ? { ...prev, trackId: null } : prev,
          );
          applyAnnotatorInteractionMode();
          return;
        }

        commitFromAnnotorious();
        setSelectionState((prev) =>
          prev.trackId ? { ...prev, trackId: null } : prev,
        );
        applyAnnotatorInteractionMode();
      }
    };

    const onCreate = (created: ImageAnnotation) => {
      if (isProgrammaticAnnoSyncRef.current) return;

      const cls = selection.className;
      if (!cls) return;

      const createdId = created?.id;
      if (!createdId) {
        commitFromAnnotorious();
        return;
      }

      const raw = api.getAnnotations();

      // Track ids already used for this class (session + current overlay).
      const used = new Set<string>();

      for (const tid of existingTrackIdsForClass(byFrameRef.current, cls)) {
        const c = canonicalizeTrackId(tid);
        if (c) used.add(c);
      }

      for (const a of raw) {
        const got = getLabelTrack(a);
        if ((got.className ?? "").trim() !== cls) continue;

        const tid = canonicalizeTrackId(got.trackId ?? "");
        if (tid) used.add(tid);
      }

      let trackId = selection.trackId ?? null;
      if (!trackId) {
        trackId = uniqueReadableTrackId(used);
      }

      // Important: patch only the newly created annotation via updateAnnotation.
      // Avoid full clear/set rewrite during create, which can break OSD draw state.
      const createdForTool =
        drawingTool === "point" ? stampPoint(created) : created;
      const patched = normalizeOverlayForSession({
        raw: [stampLabelAndTrack(createdForTool, cls, String(trackId))],
        frameKey,
        fallback: { className: cls, trackId: String(trackId) },
        enforceBothBodies: true,
        dedupeByInstance: false,
      })[0];

      if (patched?.id === createdId) {
        api.updateAnnotation?.(patched);
        const nextRaw = raw.map((annotation) =>
          annotation.id === createdId ? patched : annotation,
        );
        commitFromAnnotorious(nextRaw);
      } else {
        commitFromAnnotorious();
      }
      api.setSelected?.();
      applyAnnotatorInteractionMode();
    };

    const onUpdate = (
      _updated: ImageAnnotation,
      _previous: ImageAnnotation,
    ) => {
      commitFromAnnotorious();
      applyAnnotatorInteractionMode();
    };

    const onDelete = (_deleted: ImageAnnotation) => {
      commitFromAnnotorious();
    };

    api.on("clickAnnotation", onClickAnnotation);
    api.on("createAnnotation", onCreate);
    api.on("updateAnnotation", onUpdate);
    api.on("deleteAnnotation", onDelete);
    api.on("selectionChanged", onSelectionChanged);

    return () => {
      api.off("clickAnnotation", onClickAnnotation);
      api.off("createAnnotation", onCreate);
      api.off("updateAnnotation", onUpdate);
      api.off("deleteAnnotation", onDelete);
      api.off("selectionChanged", onSelectionChanged);
    };
  }, [
    api,
    applyAnnotatorInteractionMode,
    commitFromAnnotorious,
    drawingTool,
    frameKey,
    hideAnnotations,
    panMode,
    selection.className,
    selection.trackId,
  ]);

  const closePopup = useCallback(() => {
    api?.setSelected?.();
  }, [api]);

  const deleteAnnotation = useCallback(
    (id: string) => {
      if (!id) return;
      api?.removeAnnotation?.(id);
      api?.setSelected?.();
    },
    [api],
  );

  useEffect(() => {
    if (focusNonce === 0) return;

    const rafId = requestAnimationFrame(() => {
      tryFocusPending();
    });

    return () => cancelAnimationFrame(rafId);
  }, [focusNonce, tryFocusPending]);

  useEffect(() => {
    const rafId = requestAnimationFrame(() => {
      tryFocusPending();
    });

    return () => cancelAnimationFrame(rafId);
  }, [frame, tryFocusPending]);

  const value = useMemo<VideoSessionCtx>(
    () => ({
      projectId,
      sampleId,
      frame,
      setFrame,
      frameKey,
      byFrame,
      dirty,
      markSaved,
      selection,
      setSelection,
      instances,
      drawingTool,
      setDrawingTool,
      panMode,
      setPanMode,
      propagate,
      setPropagate,
      hideAnnotations,
      setHideAnnotations,
      imageNatural,
      setImageNatural,
      createPointAnnotation,
      deleteAnnotation,
      closePopup,
      requestFocusInstance,
      getFrameList,
      flushCurrentFrameOverlay,
      clearCurrentFrame,
      clearAllFrames,
      createNewInstanceForClass,
      deleteInstanceAcrossFrames,
      deleteSelectedInstanceAcrossFrames,
      forwardPropToNextIfEmpty,
    }),
    [
      projectId,
      sampleId,
      frame,
      setFrame,
      frameKey,
      byFrame,
      dirty,
      markSaved,
      selection,
      setSelection,
      instances,
      drawingTool,
      setDrawingTool,
      panMode,
      setPanMode,
      propagate,
      setPropagate,
      hideAnnotations,
      setHideAnnotations,
      imageNatural,
      setImageNatural,
      createPointAnnotation,
      deleteAnnotation,
      closePopup,
      requestFocusInstance,
      getFrameList,
      flushCurrentFrameOverlay,
      clearCurrentFrame,
      clearAllFrames,
      createNewInstanceForClass,
      deleteInstanceAcrossFrames,
      deleteSelectedInstanceAcrossFrames,
      forwardPropToNextIfEmpty,
    ],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
