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

import {
  isPointAnno,
  readPointGeometry,
} from "@/app/video/components/anno-utils";

type ScreenPoint = {
  id: string;
  imageX: number;
  imageY: number;
  selected: boolean;
};

const RING_RADIUS_PX = 9;
const DOT_RADIUS_PX = 2.25;
const POINT_RING_STROKE = "#ffffff";
const SELECTED_POINT_RING_STROKE = "#38bdf8";

export function PointCenterDotOverlay(props: {
  api: AnnotoriousOpenSeadragonAnnotator | undefined;
  annotations: ImageAnnotation[];
  selectedAnnotations?: ImageAnnotation[];
  hidden: boolean;
  skipSelectedEditable?: boolean;
}) {
  const {
    api,
    annotations,
    selectedAnnotations = [],
    hidden,
    skipSelectedEditable = false,
  } = props;
  const markerRefs = useRef(new Map<string, SVGGElement>());

  const selectedAnnotationIds = useMemo(
    () =>
      new Set(
        selectedAnnotations
          .map((annotation) =>
            annotation.id == null ? null : String(annotation.id),
          )
          .filter((id): id is string => id !== null),
      ),
    [selectedAnnotations],
  );

  const skippedAnnotationIds = useMemo(() => {
    if (!skipSelectedEditable) return new Set<string>();
    return selectedAnnotationIds;
  }, [selectedAnnotationIds, skipSelectedEditable]);

  const points = useMemo<ScreenPoint[]>(() => {
    if (hidden || !api?.viewer) return [];

    const annotationsById = new Map<string, ImageAnnotation>();

    for (const annotation of annotations) {
      annotationsById.set(
        String(annotation.id ?? annotationsById.size),
        annotation,
      );
    }

    for (const annotation of selectedAnnotations) {
      annotationsById.set(
        String(annotation.id ?? annotationsById.size),
        annotation,
      );
    }

    return [...annotationsById.values()].flatMap((annotation) => {
      if (
        annotation.id != null &&
        skippedAnnotationIds.has(String(annotation.id))
      ) {
        return [];
      }

      if (!isPointAnno(annotation)) return [];

      const geometry = readPointGeometry(annotation);
      if (!geometry) return [];

      return [
        {
          id: String(annotation.id ?? `${geometry.x}-${geometry.y}`),
          imageX: geometry.x,
          imageY: geometry.y,
          selected:
            annotation.id != null &&
            selectedAnnotationIds.has(String(annotation.id)),
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

  const updateMarkerPositions = useCallback(() => {
    const viewer = api?.viewer;
    if (!viewer) return;

    for (const point of points) {
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
  }, [api, points]);

  useLayoutEffect(() => {
    updateMarkerPositions();
  }, [updateMarkerPositions]);

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
            r={point.selected ? RING_RADIUS_PX + 2 : RING_RADIUS_PX}
            fill={
              point.selected
                ? "rgba(56, 189, 248, 0.18)"
                : "rgba(255, 255, 255, 0.12)"
            }
            stroke={
              point.selected ? SELECTED_POINT_RING_STROKE : POINT_RING_STROKE
            }
            strokeWidth={point.selected ? 2.5 : 2}
          />
          <circle
            r={DOT_RADIUS_PX}
            fill="#ffffff"
            stroke="rgba(0, 0, 0, 0.9)"
            strokeWidth={1.1}
          />
        </g>
      ))}
    </svg>
  );
}
