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
};

const RING_RADIUS_PX = 9;
const DOT_RADIUS_PX = 2.25;

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
  const svgRef = useRef<SVGSVGElement | null>(null);
  const markerRefs = useRef(new Map<string, SVGGElement>());

  const skippedAnnotationIds = useMemo(() => {
    if (!skipSelectedEditable) return new Set<string>();

    return new Set(
      selectedAnnotations
        .map((annotation) =>
          annotation.id == null ? null : String(annotation.id),
        )
        .filter((id): id is string => id !== null),
    );
  }, [selectedAnnotations, skipSelectedEditable]);

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
        },
      ];
    });
  }, [api, annotations, hidden, selectedAnnotations, skippedAnnotationIds]);

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
      ref={svgRef}
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
            r={RING_RADIUS_PX}
            fill="rgba(255, 255, 255, 0.12)"
            stroke="#ffffff"
            strokeWidth={2}
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
