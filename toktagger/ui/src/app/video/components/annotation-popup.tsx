"use client";

import React from "react";
import { Button } from "@adobe/react-spectrum";

/**
 * Lightweight floating UI for the currently selected annotation.
 *
 * Positioning is handled by Annotorious' ImageAnnotationPopup wrapper.
 * This component stays purely presentational.
 */
export function AnnotationPopup(props: {
  className: string | null;
  trackId: string | null;
  heading?: string;
  geometry?: { x: number; y: number; w: number; h: number } | null;
  details?: string | null;
  deleteDisabled?: boolean;
  onDeleteBox: () => void;
  onClose: () => void;
}) {
  const { className, trackId, geometry, details } = props;

  const label = className ?? "—";
  const detailText =
    details ??
    (geometry
      ? `x=${Math.round(geometry.x)}, y=${Math.round(geometry.y)}, w=${Math.round(
          geometry.w,
        )}, h=${Math.round(geometry.h)}`
      : null);

  return (
    <div
      className="z-[60] pointer-events-auto"
      role="dialog"
      aria-label="Annotation actions"
    >
      <div className="rounded-lg border border-white/10 bg-black/80 backdrop-blur px-3 py-2 shadow-lg min-w-[220px]">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <div className="text-[11px] text-white/70">
              {props.heading ?? "Selected"}
            </div>
            <div className="text-sm font-semibold text-white truncate">
              {label}
              {trackId && (
                <>
                  {" "}
                  <span className="text-white/70">/</span> {trackId}
                </>
              )}
            </div>

            {detailText && (
              <div className="mt-1 text-[11px] text-white/60">{detailText}</div>
            )}
          </div>

          <button
            onClick={props.onClose}
            className="shrink-0 rounded-md px-2 py-1 text-white/80 hover:text-white hover:bg-white/10"
            title="Close"
            aria-label="Close popup"
          >
            ✕
          </button>
        </div>

        <div className="mt-2 flex gap-2">
          <Button
            variant="negative"
            isDisabled={props.deleteDisabled}
            onPress={props.onDeleteBox}
          >
            Delete
          </Button>
        </div>
      </div>
    </div>
  );
}
