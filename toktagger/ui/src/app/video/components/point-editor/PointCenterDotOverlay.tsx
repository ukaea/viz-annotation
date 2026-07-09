"use client";

import React, { useEffect, useRef, useState } from "react";
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
  x: number;
  y: number;
};

export function PointCenterDotOverlay(props: {
  api: AnnotoriousOpenSeadragonAnnotator | undefined;
  annotations: ImageAnnotation[];
  selectedAnnotations?: ImageAnnotation[];
  hidden: boolean;
}) {
  const { api, annotations, selectedAnnotations = [], hidden } = props;
  const svgRef = useRef<SVGSVGElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const [, setViewportTick] = useState(0);

  useEffect(() => {
    const viewer = api?.viewer;
    if (!viewer) return;

    const scheduleUpdate = () => {
      if (rafRef.current != null) return;

      rafRef.current = requestAnimationFrame(() => {
        rafRef.current = null;
        setViewportTick((tick) => tick + 1);
      });
    };

    viewer.addHandler("open", scheduleUpdate);
    viewer.addHandler("viewport-change", scheduleUpdate);
    viewer.addHandler("update-viewport", scheduleUpdate);
    viewer.addHandler("animation-finish", scheduleUpdate);
    viewer.addHandler("resize", scheduleUpdate);
    scheduleUpdate();

    return () => {
      viewer.removeHandler("open", scheduleUpdate);
      viewer.removeHandler("viewport-change", scheduleUpdate);
      viewer.removeHandler("update-viewport", scheduleUpdate);
      viewer.removeHandler("animation-finish", scheduleUpdate);
      viewer.removeHandler("resize", scheduleUpdate);

      if (rafRef.current != null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [api]);

  const points: ScreenPoint[] =
    hidden || !api?.viewer || !svgRef.current
      ? []
      : (() => {
          const viewer = api.viewer;
          const svgBounds = svgRef.current.getBoundingClientRect();
          const viewerBounds = (
            viewer.element as HTMLElement
          ).getBoundingClientRect();
          const offsetX = viewerBounds.left - svgBounds.left;
          const offsetY = viewerBounds.top - svgBounds.top;

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
            if (!isPointAnno(annotation)) return [];

            const point = readPointGeometry(annotation);
            if (!point) return [];

            const screenPoint = viewer.viewport.imageToViewerElementCoordinates(
              new OpenSeadragon.Point(point.x, point.y),
            );

            return [
              {
                id: String(annotation.id ?? `${point.x}-${point.y}`),
                x: offsetX + screenPoint.x,
                y: offsetY + screenPoint.y,
              },
            ];
          });
        })();

  return (
    <svg
      ref={svgRef}
      className="pointer-events-none absolute inset-0 z-10"
      aria-hidden="true"
    >
      {points.map((point) => (
        <circle
          key={point.id}
          cx={point.x}
          cy={point.y}
          r={2.75}
          fill="#ffffff"
          stroke="rgba(0, 0, 0, 0.9)"
          strokeWidth={1.25}
        />
      ))}
    </svg>
  );
}
