"use client";

import {
  TIME_SERIES_ANNOTATION_MENU,
  useTimeSeriesActions,
  useTimeSeriesState,
} from "@/app/contexts/TimeSeriesContext";
import {
  ExtendedPlotlyHTMLElement,
  TimeSeriesAnnotation,
  TimeSeriesAnnotationPoint,
  TimeSeriesAnnotationType,
  ToolingCallbacks,
  ToolingProps,
} from "@/types";
import * as d3 from "d3";
import { useEffect, useRef } from "react";
import { useContextMenu } from "react-contexify";

enum Side {
  LEFT,
  BOTTOM,
  TOP,
  RIGHT,
}

export const BoundingBox = ({ plotId, plotReady }: ToolingProps) => {
  const {
    registerTooling,
    createAnnotation,
    addAnnotation,
    updateAnnotation,
    setOngoingAction,
    selectAnnotations,
  } = useTimeSeriesActions();
  const { annotations, forceUpdate, isDrawing, categories, editMode } =
    useTimeSeriesState();

  const currentAnnotation = useRef<TimeSeriesAnnotation | null>(null);
  const dragOffset = useRef({ x: 0, y: 0 });

  // Hook to trigger the context provider to render context menu
  const { show } = useContextMenu({
    id: TIME_SERIES_ANNOTATION_MENU,
  });

  useEffect(() => {
    const toolingCallbacks: ToolingCallbacks = {
      start: (x, y, label) => {
        const annotation = createAnnotation(
          TimeSeriesAnnotationType.BOUNDING_BOX,
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
      end(_x, _y) {
        if (!currentAnnotation.current) return;

        // Once the user has drawn the annotation the points are recalculated to ensure the bottom left and top right are stored
        const origin: TimeSeriesAnnotationPoint = {
          x: Math.min(
            currentAnnotation.current.points[0].x,
            currentAnnotation.current.points[1].x,
          ),
          y: Math.min(
            currentAnnotation.current.points[0].y,
            currentAnnotation.current.points[1].y,
          ),
        };
        const extreme: TimeSeriesAnnotationPoint = {
          x: Math.max(
            currentAnnotation.current.points[0].x,
            currentAnnotation.current.points[1].x,
          ),
          y: Math.max(
            currentAnnotation.current.points[0].y,
            currentAnnotation.current.points[1].y,
          ),
        };
        currentAnnotation.current.points[0] = origin;
        currentAnnotation.current.points[1] = extreme;
        updateAnnotation(currentAnnotation.current);
      },
    };
    registerTooling(TimeSeriesAnnotationType.BOUNDING_BOX, toolingCallbacks);
  }, [
    addAnnotation,
    createAnnotation,
    registerTooling,
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
      console.error("Could not locate plot to generate bounding box");
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
        console.warn("Could not find overplot for bounding box rendering");
        return;
      }

      // Find the y axis ID relating to this subplot
      const yAxisID = subplotId.match(/y(.*)$/)?.[1];
      if (!yAxisID && yAxisID !== "") {
        console.error("Could not find valid subplot y-axis ID");
        return;
      }
      // Use the axis information to calculate the upper and lower limits of the zone
      const yAxis = plot._fullLayout[`yaxis${yAxisID}`];
      const xAxis = plot._fullLayout.xaxis;

      // Minimum width in data units: 0.5% of current x-range
      const [xMin, xMax] = xAxis.range;
      const MIN_WIDTH_FRACTION = 0.005; // 0.5%
      const minWidth = (xMax - xMin) * MIN_WIDTH_FRACTION;

      // Minimum height in data units: 0.5% of current x-range
      const [yMin, yMax] = yAxis.range;
      const MIN_HEIGHT_FRACTION = 0.005; // 0.5%
      const minHeight = (yMax - yMin) * MIN_HEIGHT_FRACTION;

      const graphGroup = d3.select(overplot);
      graphGroup.selectAll(".bounding-box").remove(); // All VSpans are removed each render cycle

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

      const translateHandler = d3
        .drag<SVGRectElement, TimeSeriesAnnotation>()
        .on("start", function (event, d) {
          selectAnnotations([d.id]); // Visually selects the annotation when editting starts
          // Calculate the mouse offset from the origin when the annotation is clicked
          dragOffset.current = {
            x: xAxis.d2p(d.points[0].x) - event.x,
            y: event.y - yAxis.d2p(d.points[0].y),
          };
        })
        .on("drag", function (event, d) {
          const newX = event.x + dragOffset.current.x;
          const newY = event.y + dragOffset.current.y;

          // This ensure the visual rendering is updated in real-time
          d3.select(this).attr("x", newX);
          d3.select(this).attr("y", newY);

          // Bottom-left corner
          const origin = {
            x: xAxis.p2d(newX),
            y: yAxis.p2d(newY),
          };

          // Top-right corner
          const extreme = {
            x: xAxis.p2d(
              newX +
                Math.abs(xAxis.d2p(d.points[1].x) - xAxis.d2p(d.points[0].x)),
            ),
            y: yAxis.p2d(
              newY +
                Math.abs(yAxis.d2p(d.points[1].y) - yAxis.d2p(d.points[0].y)),
            ),
          };

          d.points[0] = origin;
          d.points[1] = extreme;
          updateAnnotation(d); // Global refresh must be triggered to update all linked plots
          setOngoingAction(true);
        })
        .on("end", function (_event, _d) {
          setOngoingAction(false);
        });

      const getBoundaryHandler = (side: Side) => {
        // Used to reduce code repetition
        let point_id: number;
        if (side === Side.LEFT || side === Side.BOTTOM) {
          point_id = 0;
        } else {
          point_id = 1;
        }

        const resize = d3
          .drag<SVGRectElement, TimeSeriesAnnotation>()
          .on("start", function (event, d) {
            selectAnnotations([d.id]); // Generate visual indication that editting is happening
          })
          .on("drag", function (event, d) {
            // Convert pointer X (pixels) → data units; allow wrap while dragging (no clamp here)
            const x = xAxis.p2d(event.x);
            const y = yAxis.p2d(event.y);

            if (side === Side.LEFT || side === Side.RIGHT) {
              d.points[point_id].x = x;
            } else {
              d.points[point_id].y = y;
            }
            updateAnnotation(d);
            setOngoingAction(true);
          })
          .on("end", function (_event, d) {
            // On drag end: enforce minimum width and height
            const width = Math.abs(d.points[1].x - d.points[0].x);
            if (width < minWidth) {
              if (side === Side.LEFT) {
                if (d.points[0].x > d.points[1].x) {
                  d.points[0].x = d.points[1].x + minWidth; // wrapped past right → clamp on the right side of right edge
                } else {
                  d.points[0].x = d.points[1].x - minWidth; // normal case → clamp on the left side of left edge
                }
              } else {
                // Symmetric logic for right edge relative to fixed left edge
                if (d.points[1].x < d.points[0].x) {
                  d.points[1].x = d.points[0].x - minWidth; // wrapped past left → clamp on the left side of edge
                } else {
                  d.points[1].x = d.points[0].x + minWidth; // normal case → clamp on the right side of edge
                }
              }
            }

            const height = Math.abs(d.points[1].y - d.points[0].y);
            if (height < minHeight) {
              if (side === Side.BOTTOM) {
                if (d.points[0].y > d.points[1].y) {
                  d.points[0].y = d.points[1].y + minHeight; // wrapped past top → clamp on the top side of top edge
                } else {
                  d.points[0].y = d.points[1].y - minHeight; // normal case → clamp on the bottom side of top edge
                }
              } else {
                // Symmetric logic for top edge relative to fixed bottom edge
                if (d.points[1].y < d.points[0].y) {
                  d.points[1].y = d.points[0].y - minHeight; // wrapped past bottom → clamp on the bottom side of edge
                } else {
                  d.points[1].y = d.points[0].y + minHeight; // normal case → clamp on the top side of edge
                }
              }
            }

            updateAnnotation(d);
            setOngoingAction(false);
          });
        return resize;
      };

      // Create a line and a transparent drag handle for each VSpan
      for (const boundingBox of annotations) {
        if (boundingBox.type !== TimeSeriesAnnotationType.BOUNDING_BOX)
          continue;
        const opacity = boundingBox.selected ? 0.8 : 0.5;

        // pixel positions for the two definiting points
        const px0 = xAxis.d2p(boundingBox.points[0].x);
        const py0 = yAxis.d2p(boundingBox.points[0].y);
        const px1 = xAxis.d2p(boundingBox.points[1].x);
        const py1 = yAxis.d2p(boundingBox.points[1].y);
        const pointerEvent = isDrawing || !editMode ? "none" : "all";

        const originPoint: TimeSeriesAnnotationPoint = {
          x: Math.min(px0, px1),
          y: Math.min(py0, py1),
        };

        const boxWidth = Math.abs(px1 - px0);
        const boxHeight = Math.abs(py1 - py0);

        // handle layout: fixed outside strip + variable inside strip
        const OUTER_HANDLE_PX = 10; // fixed, always clickable outside the zone
        const INNER_HANDLE_MAX_PX = 10; // cap inside portion so handles don't dominate
        const MIN_CENTER_DRAG_PX = 6; // keep a gap so the middle stays draggable
        const EDGE_PERCENTAGE = 0.5;
        const EDGE_BUFFER = (1 - EDGE_PERCENTAGE) / 2;

        // inside portion per side; shrink when zone is tiny to keep a center gap
        const innerWidth = Math.max(
          0,
          Math.min(INNER_HANDLE_MAX_PX, (boxWidth - MIN_CENTER_DRAG_PX) / 2),
        );
        const totalHandleWidth = OUTER_HANDLE_PX + innerWidth;

        // inside portion per side; shrink when zone is tiny to keep a center gap
        const innerHeight = Math.max(
          0,
          Math.min(INNER_HANDLE_MAX_PX, (boxHeight - MIN_CENTER_DRAG_PX) / 2),
        );
        const totalHandleHeight = OUTER_HANDLE_PX + innerHeight;

        const categoryId = `${boundingBox.type}_${boundingBox.label}`;
        const color = categories.get(categoryId)?.color || "black";

        // Bounding box (center drag target)
        graphGroup
          .append("rect")
          .attr("aria-label", "bounding-box")
          .attr(
            "class",
            "annotation bounding-box cursor-grab disable-on-modifier",
          )
          .attr("x", originPoint.x)
          .attr("y", originPoint.y)
          .attr("width", boxWidth)
          .attr("height", boxHeight)
          .attr("fill", color)
          .attr("opacity", opacity)
          .attr("style", `pointer-events: ${pointerEvent}`)
          .attr("stroke", "black")
          .attr("stroke-width", 1)
          .attr("stroke", "gray")
          .style("cursor", "move")
          .attr("stroke-width", 1)
          .datum(boundingBox)
          .call(translateHandler)
          .on("contextmenu", handleContextMenu);

        // Side handles
        graphGroup
          .append("rect")
          .attr("aria-label", "box.bottomHandle")
          .attr(
            "class",
            "annotation bounding-box bottomHandle disable-on-modifier",
          )
          .attr("x", px0 + boxWidth * EDGE_BUFFER)
          .attr("y", py0 - (totalHandleHeight - OUTER_HANDLE_PX))
          .attr("width", boxWidth * EDGE_PERCENTAGE)
          .attr("height", totalHandleHeight)
          .attr("fill", "transparent")
          .attr("style", `pointer-events: ${pointerEvent}`)
          .style("cursor", "n-resize")
          .datum(boundingBox)
          .call(getBoundaryHandler(Side.BOTTOM))
          .on("contextmenu", handleContextMenu);

        graphGroup
          .append("rect")
          .attr("aria-label", "box.topHandle")
          .attr(
            "class",
            "annotation bounding-box topHandle disable-on-modifier",
          )
          .attr("x", px0 + boxWidth * EDGE_BUFFER)
          .attr("y", py1 - OUTER_HANDLE_PX)
          .attr("width", boxWidth * EDGE_PERCENTAGE)
          .attr("height", totalHandleHeight)
          .attr("fill", "transparent")
          .attr("style", `pointer-events: ${pointerEvent}`)
          .style("cursor", "n-resize")
          .datum(boundingBox)
          .call(getBoundaryHandler(Side.TOP))
          .on("contextmenu", handleContextMenu);

        graphGroup
          .append("rect")
          .attr("aria-label", "box.leftHandle")
          .attr(
            "class",
            "annotation bounding-box leftHandle disable-on-modifier",
          )
          .attr("x", px0 - OUTER_HANDLE_PX)
          .attr("y", py1 + boxHeight * EDGE_BUFFER)
          .attr("width", totalHandleWidth)
          .attr("height", boxHeight * EDGE_PERCENTAGE)
          .attr("fill", "transparent")
          .attr("style", `pointer-events: ${pointerEvent}`)
          .style("cursor", "w-resize")
          .datum(boundingBox)
          .call(getBoundaryHandler(Side.LEFT))
          .on("contextmenu", handleContextMenu);

        graphGroup
          .append("rect")
          .attr("aria-label", "box.rightHandle")
          .attr(
            "class",
            "annotation bounding-box rightHandle disable-on-modifier",
          )
          .attr("x", px1 - (totalHandleWidth - OUTER_HANDLE_PX))
          .attr("y", py1 + boxHeight * EDGE_BUFFER)
          .attr("width", totalHandleWidth)
          .attr("height", boxHeight * EDGE_PERCENTAGE)
          .attr("fill", "transparent")
          .attr("style", `pointer-events: ${pointerEvent}`)
          .style("cursor", "w-resize")
          .datum(boundingBox)
          .call(getBoundaryHandler(Side.RIGHT))
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
