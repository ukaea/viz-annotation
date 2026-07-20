"use client";

import {
  TIME_SERIES_ANNOTATION_MENU,
  useTimeSeriesActions,
  useTimeSeriesState,
} from "@/app/contexts/TimeSeriesContext";
import {
  ExtendedPlotlyHTMLElement,
  TimeSeriesAnnotation,
  TimeSeriesAnnotationType,
  ToolingCallbacks,
  ToolingProps,
} from "@/types";
import * as d3 from "d3";
import { useEffect, useRef } from "react";
import { useContextMenu } from "react-contexify";

export const TimeRegion = ({ plotId, plotReady }: ToolingProps) => {
  const {
    registerTooling,
    createAnnotation,
    addAnnotation,
    removeAnnotation,
    updateAnnotation,
    setOngoingAction,
    selectAnnotations,
  } = useTimeSeriesActions();
  const { annotations, forceUpdate, isDrawing, categories, editMode } =
    useTimeSeriesState();

  const currentAnnotation = useRef<TimeSeriesAnnotation | null>(null);
  const dragOffset = useRef(0);

  // Hook to trigger the context provider to render context menu
  const { show } = useContextMenu({
    id: TIME_SERIES_ANNOTATION_MENU,
  });

  useEffect(() => {
    const toolingCallbacks: ToolingCallbacks = {
      start: (x, y, label, _axisSize) => {
        const annotation = createAnnotation(
          TimeSeriesAnnotationType.TIME_REGION,
          label,
        );
        currentAnnotation.current = annotation;
        annotation.points.push({ x, y });
        annotation.points.push({ x, y });
        addAnnotation(annotation);
      },
      move(x, y) {
        if (!currentAnnotation.current) {
          console.warn(
            "Could not update annotation as ID reference has been lost",
          );
          return;
        }
        if (!currentAnnotation.current.points[1]) {
          console.warn("Could not update zone as data points are invalid");
          return;
        }
        currentAnnotation.current.points[1] = { x, y };
        updateAnnotation(currentAnnotation.current);
      },
      end(_x, _y) {},
      cancel() {
        if (currentAnnotation.current) {
          removeAnnotation(currentAnnotation.current.id);
        }
        currentAnnotation.current = null;
      },
    };
    registerTooling(TimeSeriesAnnotationType.TIME_REGION, toolingCallbacks);
  }, [
    addAnnotation,
    createAnnotation,
    registerTooling,
    removeAnnotation,
    setOngoingAction,
    updateAnnotation,
  ]);

  // Main rendering effect
  useEffect(() => {
    // This shall not run until the target plot is initialised
    if (!plotId || !plotReady) {
      return;
    }

    // Grab the handle set up in the main plot for D3 rendering
    const plot = document.getElementById(plotId) as ExtendedPlotlyHTMLElement;

    // Rendering should not be attempted if the required handles are not found
    if (!plot) {
      console.error("Could not locate plot to generate vspans");
      return;
    }

    // Get a reference to all subplots and find the name of the axis
    const subplots = plot.querySelectorAll(".subplot");
    const subplotNames = [...subplots].map((el) =>
      [...el.classList].find((cls) => cls !== "subplot"),
    );

    // For each subplot carry out the tooling generation
    subplotNames.forEach((subplotId) => {
      if (subplotId === undefined) {
        console.error("Could not find valid subplot ID");
        return;
      }

      const overplot = document.getElementsByClassName(
        `${plotId}-overplot-${subplotId}`,
      )[0];

      if (!overplot) {
        // Silently skip if overplot not found yet - it will be available on next render
        console.warn("Could not find overplot for vspan rendering");
        return;
      }

      // Find the y axis ID relating to this subplot
      const yAxisID = subplotId.match(/y(.*)$/)?.[1];
      if (!yAxisID && yAxisID !== "") {
        console.error("Could not find valid subplot y-axis ID");
        return;
      }
      // Use the axis information to calculate the upper and lower limits of the zone
      const axis = plot._fullLayout[`yaxis${yAxisID}`];
      const range = axis._tmax - axis._tmin;
      const upperLimit = axis.d2p(axis._tmax + 2 * range);
      const lowerLimit = axis.d2p(axis._tmin - 2 * range);
      const height = lowerLimit - upperLimit;

      const xaxis = plot._fullLayout.xaxis;

      // Minimum width in data units: 0.1% of current x-range
      const [xMin, xMax] = xaxis.range;
      const MIN_WIDTH_FRACTION = 0.001; // 0.1%
      const minWidth = (xMax - xMin) * MIN_WIDTH_FRACTION;

      const graphGroup = d3.select(overplot);
      graphGroup.selectAll(".time-region").remove(); // All VSpans are removed each render cycle

      // Prevents a little bit of repetition by auto-configuring the resize handler
      const getBoundaryHandler = (isLeft: boolean) => {
        // Handles the dragging of the boundaries of the zone
        const resize = d3
          .drag<SVGRectElement, TimeSeriesAnnotation>()
          .on("start", function (event, d) {
            selectAnnotations([d.id]);
          })
          .on("drag", function (event, d) {
            // Convert pointer X (pixels) → data units; allow wrap while dragging (no clamp here)
            const x = xaxis.p2d(event.x);
            if (isLeft) d.points[0].x = x;
            else d.points[1].x = x; // live-update only the boundary being dragged
            updateAnnotation(d);
            setOngoingAction(true);
          })
          .on("end", function (_event, d) {
            // On drag end: enforce minimum width and normalize orientation
            // minWidth is in data units (computed from current x-range above)
            let changed = false;
            const width = Math.abs(d.points[1].x - d.points[0].x);
            if (width < minWidth) {
              // Clamp to min width by moving ONLY the boundary the user dragged.
              // Keep the opposite boundary fixed so the zone’s "anchor"/center doesn’t jump.
              changed = true;
              if (isLeft) {
                // If the left boundary has crossed to the right of x1, place it to the right; otherwise to the left.
                if (d.points[0].x > d.points[1].x) {
                  d.points[0].x = d.points[1].x + minWidth; // wrapped past right → clamp on the right side of x1
                } else {
                  d.points[0].x = d.points[1].x - minWidth; // normal case → clamp on the left side of x1
                }
              } else {
                // Symmetric logic for right boundary relative to fixed left boundary (x0)
                if (d.points[1].x < d.points[0].x) {
                  d.points[1].x = d.points[0].x - minWidth; // wrapped past left → clamp on the left side of x0
                } else {
                  d.points[1].x = d.points[0].x + minWidth; // normal case → clamp on the right side of x0
                }
              }
            }
            // Always normalize so downstream logic sees x0 <= x1
            if (d.points[1].x < d.points[0].x) {
              const t = d.points[0].x;
              d.points[0].x = d.points[1].x;
              d.points[1].x = t;
              changed = true;
            }
            if (changed) {
              updateAnnotation(d);
            }
            setOngoingAction(false);
          });
        return resize;
      };

      const translateHandler = d3
        .drag<SVGRectElement, TimeSeriesAnnotation>()
        .on("start", function (event, d) {
          selectAnnotations([d.id]);
          const leftBoundary = Math.min(d.points[0].x, d.points[1].x);
          dragOffset.current = xaxis.d2p(leftBoundary) - event.x;
        })
        .on("drag", function (event, d) {
          const newX = event.x + dragOffset.current;
          d3.select(this).attr("x", newX);

          const x0 = xaxis.p2d(newX);
          const x1 = xaxis.p2d(
            newX +
              Math.abs(xaxis.d2p(d.points[1].x) - xaxis.d2p(d.points[0].x)),
          );
          const x0Left = d.points[0].x < d.points[1].x;
          d.points[0].x = x0Left ? x0 : x1;
          d.points[1].x = x0Left ? x1 : x0;
          updateAnnotation(d); // Global refresh must be triggered to update all linked plots
          setOngoingAction(true);
        })
        .on("end", function (_event, _d) {
          setOngoingAction(false);
        });

      function handleContextMenu(
        event: MouseEvent,
        annotation: TimeSeriesAnnotation,
      ) {
        event.preventDefault(); // Prevent default context menu
        const isRightClickEvent = event.button === 2 && !event.ctrlKey;
        if (isRightClickEvent) {
          show({
            event,
            props: {
              annotation,
            },
          });
        }
      }

      // Create a line and a transparent drag handle for each VSpan
      for (const zone of annotations) {
        if (zone.type !== TimeSeriesAnnotationType.TIME_REGION) continue;
        const opacity = zone.selected ? 0.8 : 0.5;

        // pixel positions for the two data boundaries
        const px0 = xaxis.d2p(zone.points[0].x);
        const px1 = xaxis.d2p(zone.points[1].x);
        const pointerEvent = isDrawing || !editMode ? "none" : "all";

        // render span using left-most x and absolute width, span in pixels
        const spanLeft = Math.min(px0, px1);
        const spanRight = Math.max(px0, px1);
        const spanWidth = spanRight - spanLeft;

        // handle layout: fixed outside strip + variable inside strip
        const OUTER_HANDLE_PX = 10; // fixed, always clickable outside the zone
        const INNER_HANDLE_MAX_PX = 10; // cap inside portion so handles don't dominate
        const MIN_CENTER_DRAG_PX = 6; // keep a gap so the middle stays draggable

        // inside portion per side; shrink when zone is tiny to keep a center gap
        const inner = Math.max(
          0,
          Math.min(INNER_HANDLE_MAX_PX, (spanWidth - MIN_CENTER_DRAG_PX) / 2),
        );
        const totalHandleWidth = OUTER_HANDLE_PX + inner;

        const x0IsLeft = px0 <= px1;

        const categoryId = `${zone.type}_${zone.label}`;
        const color = categories.get(categoryId)?.color || "black";

        // Span (center drag target)
        graphGroup
          .append("rect")
          .attr("aria-label", "time-zone")
          .attr(
            "class",
            "annotation time-region span cursor-grab disable-on-modifier",
          )
          .attr("x", spanLeft)
          .attr("y", upperLimit)
          .attr("width", spanWidth)
          .attr("height", height)
          .attr("fill", color)
          .attr("opacity", opacity)
          .attr("style", `pointer-events: ${pointerEvent}`)
          .attr("stroke", "black")
          .attr("stroke-width", 1)
          .attr("stroke", "gray")
          .style("cursor", "move")
          .attr("stroke-width", 1)
          .datum(zone)
          .call(translateHandler)
          .on("contextmenu", handleContextMenu);

        // x0 handle (moves x0): outside is away from the zone, inside points toward the other end
        const x0HandleX = x0IsLeft ? px0 - OUTER_HANDLE_PX : px0 - inner;
        graphGroup
          .append("rect")
          .attr("aria-label", "zone.leftHandle")
          .attr(
            "class",
            "annotation time-region leftHandle disable-on-modifier",
          )
          .attr("x", x0HandleX)
          .attr("y", upperLimit)
          .attr("width", totalHandleWidth)
          .attr("height", height)
          .attr("fill", "transparent")
          .attr("style", `pointer-events: ${pointerEvent}`)
          .style("cursor", "w-resize")
          .datum(zone)
          .call(getBoundaryHandler(true))
          .on("contextmenu", handleContextMenu);

        // x1 handle (moves x1)
        const x1HandleX = x0IsLeft ? px1 - inner : px1 - OUTER_HANDLE_PX;
        graphGroup
          .append("rect")
          .attr("aria-label", "zone.rightHandle")
          .attr("class", "time-region rightHandle disable-on-modifier")
          .attr("x", x1HandleX)
          .attr("y", upperLimit)
          .attr("width", totalHandleWidth)
          .attr("height", height)
          .attr("fill", "transparent")
          .attr("style", `pointer-events: ${pointerEvent}`)
          .style("cursor", "e-resize")
          .datum(zone)
          .call(getBoundaryHandler(false))
          .on("contextmenu", handleContextMenu);
      }
    });
  }, [
    annotations,
    isDrawing,
    plotId,
    plotReady,
    forceUpdate,
    updateAnnotation,
    categories,
    show,
    editMode,
    setOngoingAction,
    selectAnnotations,
  ]); // forceUpdate is required here to keep tooling correctly positioned

  return <div />;
};
