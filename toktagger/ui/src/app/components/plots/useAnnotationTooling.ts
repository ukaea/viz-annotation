"use client";

import { PlotlyHTMLElement, Layout, relayout } from "plotly.js";
import { useEffect, useRef } from "react";
import {
  useTimeSeriesActions,
  useTimeSeriesState,
} from "@/app/contexts/TimeSeriesContext";
import { ExtendedPlotlyHTMLElement, TimeSeriesAnnotationPoint } from "@/types";
import { ToastQueue } from "@adobe/react-spectrum";

/**
 * Wires a Plotly plot up to the annotation tools registered on the TimeSeriesContext.
 *
 * Translates pointer events on the plot's drag layers into data-space tooling
 * callbacks, so that any plot can host the D3 annotation tools. Shared by the base
 * time series plot and the profile 2D plot.
 *
 * @param plotId Id of the element the plot is rendered into
 * @param plotReady Signal from the plot that it has finished rendering
 * @param idleDragMode Drag mode to restore when no annotation is being drawn
 */
export const useAnnotationTooling = ({
  plotId,
  plotReady,
  idleDragMode = "pan",
}: {
  plotId: string;
  plotReady: boolean;
  idleDragMode?: Layout["dragmode"];
}) => {
  const { findSelectedAnnotations, setOngoingAction } = useTimeSeriesActions();
  const {
    activeAnnotationTool,
    toolingCallbacks,
    isDrawing,
    ongoingAction,
    editMode,
  } = useTimeSeriesState();

  const isDraggingRef = useRef(false);
  const lastHoverTime = useRef(0);
  const lockedSubplotElementRef = useRef<HTMLElement | null>(null); // used to track which subplot an annotation was started on

  if (!isDrawing) isDraggingRef.current = false;

  // Plotly's own drag interactions must be suspended while a tool is drawing
  useEffect(() => {
    if (!plotReady) {
      // Plot may not have loaded yet - this will rerun after loading
      return;
    }

    const plot = document.getElementById(plotId);

    if (!plot) {
      console.error("Could not locate plot to set drag mode");
      return;
    }

    if (isDrawing) {
      relayout(plot, { dragmode: false });
      return;
    }

    relayout(plot, { dragmode: idleDragMode });
  }, [isDrawing, plotId, plotReady, idleDragMode]);

  // The subplot lock whilst drawing is cleared once an action is no longer ongoing
  useEffect(() => {
    if (ongoingAction) return;
    lockedSubplotElementRef.current = null;
  }, [ongoingAction]);

  useEffect(() => {
    if (!plotReady) {
      // Plot may not have loaded yet - this will rerun after loading
      return;
    }
    const plot = document.getElementById(plotId) as PlotlyHTMLElement;
    if (!plot) {
      console.error("Could not locate plot to assign click handler");
      return;
    }

    function getClickData(
      event: PointerEvent,
      _plot: PlotlyHTMLElement,
      resolveAgainst?: HTMLElement | null, // This is used to ensure the annotation is resolved against the starting subplot
    ): TimeSeriesAnnotationPoint & { axisSize: { x: number; y: number } } {
      const plot = _plot as ExtendedPlotlyHTMLElement;
      let xaxis = plot._fullLayout.xaxis; // x-axis descriptor
      let yaxis = plot._fullLayout.yaxis; // y-axis descriptor

      const target = resolveAgainst ?? (event.target as HTMLElement); // If a resolve target is not set, the event target is used instead
      const bb = target.getBoundingClientRect();
      const relX = event.clientX - bb.left; // click X in pixels, relative to plot
      const relY = event.clientY - bb.top; // click Y in pixels, relative to plot

      const subplotId = target.dataset.subplot; // e.g. "x2y2"
      if (subplotId) {
        const m = subplotId.match(/^x(\d*)y(\d*)$/); // ['', '2', '2']
        // m[1]/m[2] hold numeric suffixes empty string -> primary axis
        if (m) {
          const suffixX = m[1] ?? ""; // '' -> xaxis
          const suffixY = m[2] ?? ""; // '' -> yaxis
          // Swap to subplot-specific axes if they exist
          xaxis = plot._fullLayout[`xaxis${suffixX}`] ?? plot._fullLayout.xaxis;
          yaxis = plot._fullLayout[`yaxis${suffixY}`] ?? plot._fullLayout.yaxis;
        }
      }
      // final catch-all fallback – runs whether or not we found a subplotId
      xaxis = xaxis ?? plot._fullLayout.xaxis;
      yaxis = yaxis ?? plot._fullLayout.yaxis;

      // Coordinates in data space
      const x = xaxis.p2d(relX); // data-space X at click
      const y = yaxis.p2d(relY); // data-space Y at click

      const axisSize = {
        x: Math.abs(xaxis.range[1] - xaxis.range[0]),
        y: Math.abs(yaxis.range[1] - yaxis.range[0]),
      };

      return { x, y, axisSize };
    }

    const draggableElements =
      plot.querySelectorAll<HTMLDivElement>(".nsewdrag");
    if (draggableElements.length === 0) {
      console.error("Could not locate drag element to assign click handler");
      return;
    }

    const handleContextMenu = (event: MouseEvent) => {
      event.preventDefault();
    };

    const handleCancelSelection = (event: PointerEvent) => {
      if (!event.ctrlKey) {
        findSelectedAnnotations(null);
      }
    };

    const startAnnotationCreation = (event: PointerEvent) => {
      if (event.ctrlKey) {
        if (!editMode) {
          ToastQueue.info(
            "Change to Edit Mode to draw annotations - see help popup in annotation toolbar for more info",
            { timeout: 5000 },
          );
          return;
        }
        if (activeAnnotationTool) {
          // If a subplot has not been locked yet (e.g the annotation has just started) the current subplot should be stored
          if (!lockedSubplotElementRef.current) {
            lockedSubplotElementRef.current =
              event.currentTarget as HTMLElement;
          }
          setOngoingAction(true);
          isDraggingRef.current = true;
          const clickLocation = getClickData(
            event,
            plot,
            lockedSubplotElementRef.current,
          );
          toolingCallbacks
            .get(activeAnnotationTool.type)
            ?.start(
              clickLocation.x,
              clickLocation.y,
              activeAnnotationTool.label,
              clickLocation.axisSize,
            );
        } else {
          ToastQueue.info(
            "Select a tool to draw annotation - see help popup in annotation toolbar for more info",
            { timeout: 5000 },
          );
        }
      }
    };

    const updateAnnotation = (event: PointerEvent) => {
      if (activeAnnotationTool && isDraggingRef.current) {
        const clickLocation = getClickData(
          event,
          plot,
          lockedSubplotElementRef.current,
        );
        toolingCallbacks
          .get(activeAnnotationTool.type)
          ?.move(clickLocation.x, clickLocation.y);
      }
    };

    const hoverAnnotation = (event: PointerEvent) => {
      if (!activeAnnotationTool) return;
      const now = Date.now();
      if (now - lastHoverTime.current < 20) return;
      lastHoverTime.current = now;
      const clickLocation = getClickData(
        event,
        plot,
        lockedSubplotElementRef.current,
      );
      toolingCallbacks
        .get(activeAnnotationTool.type)
        ?.hover?.(clickLocation.x, clickLocation.y);
    };

    const finishAnnotationCreation = (event: PointerEvent) => {
      isDraggingRef.current = false;
      // Subplot lock release is handled by the ongoingAction effect above -
      // for hover-based tools (e.g. polygon) the session continues past this pointerup
      if (activeAnnotationTool) {
        const clickLocation = getClickData(
          event,
          plot,
          lockedSubplotElementRef.current,
        );
        const callback = toolingCallbacks.get(activeAnnotationTool.type);

        // If hover behaviour is specified the callbacks should handle finishing the ongoing action
        if (!callback?.hover) {
          setOngoingAction(false);
        }
        callback?.end(clickLocation.x, clickLocation.y);
        return;
      }

      setOngoingAction(false); // Ensure this is always called even if the tool callback isn't found
    };

    draggableElements.forEach((element) => {
      element.addEventListener("contextmenu", handleContextMenu);
      element.addEventListener("pointerdown", handleCancelSelection);
      element.addEventListener("pointerdown", startAnnotationCreation);
      element.addEventListener("pointermove", hoverAnnotation);

      if (editMode) {
        element.addEventListener("pointermove", updateAnnotation);
        element.addEventListener("pointerup", finishAnnotationCreation);
      }
    });

    return () => {
      draggableElements.forEach((element) => {
        element.removeEventListener("contextmenu", handleContextMenu);
        element.removeEventListener("pointerdown", handleCancelSelection);
        element.removeEventListener("pointerdown", startAnnotationCreation);
        element.removeEventListener("pointermove", hoverAnnotation);
        element.removeEventListener("pointermove", updateAnnotation);
        element.removeEventListener("pointerup", finishAnnotationCreation);
      });
    };
  }, [
    activeAnnotationTool,
    editMode,
    findSelectedAnnotations,
    plotId,
    plotReady,
    setOngoingAction,
    toolingCallbacks,
  ]);
};
