"use client";

import React, {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
} from "react";
import OpenSeadragon from "openseadragon";
import type {
  AnnotoriousOpenSeadragonAnnotator,
  ImageAnnotation,
} from "@annotorious/react";

import { readPointGeometry } from "@/app/video/components/anno-utils";
import { POINT_MARKER } from "./marker-style";

/** A point marker stored in image coordinates and projected on viewport updates. */
type MarkerPoint = {
  id: string;
  imageX: number;
  imageY: number;
  selected: boolean;
};

export function PointCenterDotOverlay(props: {
  api: AnnotoriousOpenSeadragonAnnotator | undefined;
  annotations: ImageAnnotation[];
  selectedAnnotations?: ImageAnnotation[];
  hidden: boolean;
  isEditMode: boolean;
}) {
  const {
    api,
    annotations,
    selectedAnnotations = [],
    hidden,
    isEditMode,
  } = props;
  const markerRefs = useRef(new Map<string, SVGGElement>());
  const pointsRef = useRef<MarkerPoint[]>([]);

  const selectedAnnotationIds = useMemo(
    () => new Set(selectedAnnotations.map((annotation) => annotation.id)),
    [selectedAnnotations],
  );

  const skippedAnnotationIds = useMemo(() => {
    // In edit mode, PointEditor.svelte draws the selected annotation so it can
    // track live drag geometry without a duplicated overlay marker.
    if (!isEditMode) return new Set<string>();
    return selectedAnnotationIds;
  }, [isEditMode, selectedAnnotationIds]);

  const points = useMemo<MarkerPoint[]>(() => {
    if (hidden || !api?.viewer) return [];

    const annotationsById = new Map<string, ImageAnnotation>();

    // Selected annotations can carry fresher geometry than the session store
    // while editing, so they intentionally overwrite frame-list entries.
    for (const annotation of [...annotations, ...selectedAnnotations]) {
      annotationsById.set(annotation.id, annotation);
    }

    return [...annotationsById.values()].flatMap((annotation) => {
      if (skippedAnnotationIds.has(annotation.id)) return [];

      const geometry = readPointGeometry(annotation);
      if (!geometry) return [];

      return [
        {
          id: annotation.id,
          imageX: geometry.x,
          imageY: geometry.y,
          selected: selectedAnnotationIds.has(annotation.id),
        },
      ];
    });
  }, [
    api,
    annotations,
    hidden,
    selectedAnnotations,
    selectedAnnotationIds,
    skippedAnnotationIds,
  ]);
  pointsRef.current = points;

  const updateMarkerPositions = useCallback(() => {
    const viewer = api?.viewer;
    if (!viewer) return;

    for (const point of pointsRef.current) {
      const marker = markerRefs.current.get(point.id);
      if (!marker) continue;

      const screenPoint = viewer.viewport.imageToViewerElementCoordinates(
        new OpenSeadragon.Point(point.imageX, point.imageY),
      );

      marker.setAttribute(
        "transform",
        `translate(${screenPoint.x} ${screenPoint.y})`,
      );
    }
  }, [api]);

  useLayoutEffect(() => {
    updateMarkerPositions();
  }, [points, updateMarkerPositions]);

  useEffect(() => {
    const viewer = api?.viewer;
    if (!viewer) return;

    viewer.addHandler("open", updateMarkerPositions);
    viewer.addHandler("viewport-change", updateMarkerPositions);
    viewer.addHandler("update-viewport", updateMarkerPositions);
    viewer.addHandler("animation-finish", updateMarkerPositions);
    viewer.addHandler("resize", updateMarkerPositions);
    updateMarkerPositions();

    return () => {
      viewer.removeHandler("open", updateMarkerPositions);
      viewer.removeHandler("viewport-change", updateMarkerPositions);
      viewer.removeHandler("update-viewport", updateMarkerPositions);
      viewer.removeHandler("animation-finish", updateMarkerPositions);
      viewer.removeHandler("resize", updateMarkerPositions);
    };
  }, [api, updateMarkerPositions]);

  return (
    <svg
      className="pointer-events-none absolute inset-0 z-10 h-full w-full"
      width="100%"
      height="100%"
      aria-hidden="true"
    >
      {points.map((point) => (
        <g
          key={point.id}
          ref={(node) => {
            if (node) {
              markerRefs.current.set(point.id, node);
            } else {
              markerRefs.current.delete(point.id);
            }
          }}
        >
          <circle
            r={
              point.selected
                ? POINT_MARKER.selectedRingRadiusPx
                : POINT_MARKER.ringRadiusPx
            }
            fill={
              point.selected
                ? POINT_MARKER.selectedRingFill
                : POINT_MARKER.ringFill
            }
            stroke={
              point.selected
                ? POINT_MARKER.selectedRingStroke
                : POINT_MARKER.ringStroke
            }
            strokeWidth={
              point.selected
                ? POINT_MARKER.selectedRingStrokePx
                : POINT_MARKER.ringStrokePx
            }
          />
          <circle
            r={POINT_MARKER.dotRadiusPx}
            fill={POINT_MARKER.dotFill}
            stroke={POINT_MARKER.dotStroke}
            strokeWidth={POINT_MARKER.dotStrokePx}
          />
        </g>
      ))}
    </svg>
  );
}
