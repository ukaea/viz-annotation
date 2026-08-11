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
import { ToastQueue } from "@adobe/react-spectrum";
import * as d3 from "d3";
import { useEffect, useRef } from "react";
import { useContextMenu } from "react-contexify";

// A polygon being drawn carries two helper vertices: one that tracks the cursor and a
// temporary vertex that keeps the outline closed. Both are dropped once it is committed.
const HELPER_VERTEX_COUNT = 2;

// Fewer real vertices than this does not describe a shape, so it cannot be committed.
const MIN_POLYGON_VERTICES = 3;

export const Polygon = ({ plotId, plotReady, subplot }: ToolingProps) => {
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
  const isUpdatingPolygon = useRef(false);
  const hasDeletedVertex = useRef(false);

  // Hook to trigger the context provider to render context menu
  const { show } = useContextMenu({
    id: TIME_SERIES_ANNOTATION_MENU,
  });

  useEffect(() => {
    const toolingCallbacks: ToolingCallbacks = {
      // For polgons the start callback is responsible for creating the polygon but also appeanding vertices and closing shape too
      start: (x, y, label, axisSize) => {
        if (isUpdatingPolygon.current) {
          if (!currentAnnotation.current) {
            console.error("Could not find current annotation to add point to");
            return;
          }

          if (
            currentAnnotation.current.type !== TimeSeriesAnnotationType.POLYGON
          ) {
            console.error("Could not add point to non-polygon annotation");
            return;
          }

          const pointArrayLength = currentAnnotation.current.points.length;
          const closeThreshold = { x: axisSize.x * 0.02, y: axisSize.y * 0.02 };

          // Logic used to close the polygon when a vertex is added near the start point
          if (
            Math.abs(x - currentAnnotation.current.points[0].x) <
              closeThreshold.x &&
            Math.abs(y - currentAnnotation.current.points[0].y) <
              closeThreshold.y
          ) {
            // The polygon should only be allowed to close if there are 4 points - if not this would reduce to 2 vertices which is not allowed
            if (pointArrayLength > 4) {
              setOngoingAction(false); // Since this has hover functionality the callback must end the ongoing action
              isUpdatingPolygon.current = false;
              currentAnnotation.current.points.splice(pointArrayLength - 2, 2); // This removes the temmporary point added to prevent polygons with 2 vertices
              updateAnnotation(currentAnnotation.current);
            }
            return;
          }

          currentAnnotation.current.points[pointArrayLength - 2] = { x, y }; // Set penultimate point to be the click point
          currentAnnotation.current.points.splice(pointArrayLength - 1, 0, {
            x,
            y,
          }); // Add a new temporary point to follow hover
          updateAnnotation(currentAnnotation.current);
          return;
        }

        const annotation = createAnnotation(
          TimeSeriesAnnotationType.POLYGON,
          label,
        );
        currentAnnotation.current = annotation;
        // Polygons need at least three points so the following are added:
        annotation.points.push({ x, y }); // Start vertex (persistent)
        annotation.points.push({ x, y }); // Point to follow mouse hover - persistent whilst polygon not closed
        annotation.points.push({ x, y }); // Temporary vertex to close polygon whilst it is being drawn - removed when shape is actually closed
        isUpdatingPolygon.current = true;
        addAnnotation(annotation);
      },
      move(_x, _y) {}, // Points are added using clicks so this is not needed
      end(_x, _y) {}, // Shape is closed inside the start callback as it relies on clicks
      hover(x, y) {
        if (!currentAnnotation.current || !isUpdatingPolygon.current) return;
        const pointArrayLength = currentAnnotation.current.points.length;
        currentAnnotation.current.points[pointArrayLength - 2] = { x, y }; // Ensure the temporary hover vertex is kept up-to-date with the mouse position
        updateAnnotation(currentAnnotation.current);
      },
      cancel() {
        // Discards the half-drawn polygon if the tool is switched or drawing is escaped mid-draw
        if (isUpdatingPolygon.current && currentAnnotation.current) {
          removeAnnotation(currentAnnotation.current.id);
        }
        currentAnnotation.current = null;
        isUpdatingPolygon.current = false;
      },
    };
    registerTooling(TimeSeriesAnnotationType.POLYGON, toolingCallbacks);
  }, [
    addAnnotation,
    createAnnotation,
    registerTooling,
    removeAnnotation,
    setOngoingAction,
    updateAnnotation,
  ]);

  // Releasing the modifier key ends the drawing gesture. Every other tool has already
  // committed its shape by that point, so an in-progress polygon is committed here too
  // rather than being left with its open edge tracking the cursor. A shape that does
  // not yet have enough real vertices cannot be committed, so it is discarded instead.
  const wasDrawing = useRef(isDrawing);
  useEffect(() => {
    const drawingEnded = wasDrawing.current && !isDrawing;
    wasDrawing.current = isDrawing;

    if (!drawingEnded) return;

    const annotation = currentAnnotation.current;
    if (!isUpdatingPolygon.current || !annotation) return;

    isUpdatingPolygon.current = false;
    currentAnnotation.current = null;
    setOngoingAction(false);

    if (annotation.points.length - HELPER_VERTEX_COUNT < MIN_POLYGON_VERTICES) {
      removeAnnotation(annotation.id);
      return;
    }

    annotation.points.splice(
      annotation.points.length - HELPER_VERTEX_COUNT,
      HELPER_VERTEX_COUNT,
    );
    updateAnnotation(annotation);
  }, [isDrawing, removeAnnotation, setOngoingAction, updateAnnotation]);

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
    const targetSubplots = subplot
      ? subplotNames.filter((name) => name === subplot)
      : subplotNames;

    targetSubplots.forEach((subplotId) => {
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

      const graphGroup = d3.select(overplot);
      graphGroup.selectAll(".polygon").remove();
      graphGroup.selectAll(".polygon-point").remove();
      graphGroup.selectAll(".polygon-vertex").remove();
      graphGroup.selectAll(".polygon-edge").remove();

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

      function handleClick(
        event: MouseEvent,
        annotation: TimeSeriesAnnotation,
      ) {
        selectAnnotations([annotation.id]); // Visually selects the annotation when editting starts
      }

      const translateHandler = d3
        .drag<SVGPolygonElement, TimeSeriesAnnotation>()
        .on("start", function (_event, d) {
          selectAnnotations([d.id]); // Visually selects the annotation when editting starts
          setOngoingAction(true);
        })
        .on("drag", function (event, d) {
          // d3 drag events expose dx/dy as pixel deltas since the last event,
          // convert using the linear axis scale so it applies irrespective of origin
          const dx = xAxis.p2d(event.dx) - xAxis.p2d(0);
          const dy = yAxis.p2d(event.dy) - yAxis.p2d(0);
          d.points = d.points.map((p) => ({ x: p.x + dx, y: p.y + dy }));
          updateAnnotation(d); // Global refresh must be triggered to update all linked plots
        })
        .on("end", function () {
          setOngoingAction(false);
        });

      // Stores the vertex delete handlers so that they can be deleted later
      const deleteHandlers = new WeakMap<
        SVGCircleElement,
        (e: KeyboardEvent) => void
      >();

      const getVertexHandler = (index: number) =>
        d3
          .drag<SVGCircleElement, TimeSeriesAnnotation>()
          .on("start", function (_, d) {
            setOngoingAction(true);
            const onKeyDown = (e: KeyboardEvent) => {
              if (e.key === "Delete") {
                e.stopPropagation(); // prevents the whole annotation being deleted if selected and dragging vertex

                if (d.points.length < 4) {
                  ToastQueue.info(
                    "Cannot delete a vertex for a polygon with fewer than four points, click on the polygon and press delete to remove the annotation",
                    { timeout: 5000 },
                  );
                  return;
                }

                d.points.splice(index, 1); // Deletes the point at the index
                updateAnnotation(d);
                hasDeletedVertex.current = true; // Required to prevent the drag handler from readding the point if the mouse is moved
              }
            };

            // Store the handler keyed by the element so we can remove it later
            deleteHandlers.set(this, onKeyDown);

            // Run during capture-phase to prevent document-level listeners from getting the event
            window.addEventListener("keydown", onKeyDown, true);
          })
          .on("drag", (event, d) => {
            if (hasDeletedVertex.current) return; // If during this drag the vertex was deleted no action should be taken
            d.points[index] = { x: xAxis.p2d(event.x), y: yAxis.p2d(event.y) };
            updateAnnotation(d);
          })
          .on("end", function () {
            hasDeletedVertex.current = false;
            setOngoingAction(false);
            const handler = deleteHandlers.get(this);
            if (handler) {
              window.removeEventListener("keydown", handler, true);
              deleteHandlers.delete(this);
            }
          });

      for (const polygon of annotations) {
        if (polygon.type !== TimeSeriesAnnotationType.POLYGON) continue;
        const opacity = polygon.selected ? 0.8 : 0.5;
        const pointerEvent = isDrawing || !editMode ? "none" : "all";
        const isInProgress =
          currentAnnotation.current?.id === polygon.id &&
          isUpdatingPolygon.current;

        const convertedPoints = polygon.points.map((point) => ({
          x: xAxis.d2p(point.x),
          y: yAxis.d2p(point.y),
        }));

        const categoryId = `${polygon.type}_${polygon.label}`;
        const color = categories.get(categoryId)?.color || "black";

        // Render the actual polygon - noting this is why there can't be less than 3 vertices
        const polygonShape = graphGroup
          .append("polygon")
          .attr("aria-label", "polygon")
          .attr("class", "annotation polygon cursor-grab disable-on-modifier")
          .attr(
            "points",
            convertedPoints.map((p) => `${p.x}, ${p.y}`).join(" "),
          ) // x1, y1 x2, y2 ...
          .attr("fill", color)
          .attr("opacity", opacity)
          .attr("style", `pointer-events: ${pointerEvent}`)
          .attr("stroke-width", 1)
          .attr("stroke", "gray")
          .datum(polygon)
          .on("click", handleClick)
          .on("contextmenu", handleContextMenu);

        if (editMode && !isInProgress) {
          polygonShape.style("cursor", "move").call(translateHandler);
        }

        // Edit geometery
        if (editMode) {
          // Edit geometery only rendered when in edit mode
          // When an annotation is being drawn only the start vertex is rendered (makes it easy to see where to end the drawing)
          // otherwise all vertices are drawn with accompanying handles plus the edge handles
          if (!isInProgress) {
            // Edge hit targets (for adding new points) — appended before vertex circles so vertices take priority
            convertedPoints.forEach((p, i) => {
              const next = convertedPoints[(i + 1) % convertedPoints.length];
              graphGroup
                .append("line")
                .attr("aria-label", "polygon-edge-handle")
                .attr("class", "annotation polygon-edge disable-on-modifier")
                .attr("x1", p.x)
                .attr("y1", p.y)
                .attr("x2", next.x)
                .attr("y2", next.y)
                .attr("stroke", "transparent")
                .attr("stroke-width", 10)
                .attr("style", `pointer-events: ${pointerEvent}; cursor: cell`)
                .datum(polygon)
                .on("click", (event, d) => {
                  // Find halfway point along edge
                  const x = xAxis.p2d((p.x + next.x) / 2);
                  const y = yAxis.p2d((p.y + next.y) / 2);

                  d.points.splice(i + 1, 0, { x, y }); // Add new point at halfway point
                  updateAnnotation(d);
                });
            });

            // Vertex rendering (non-functional)
            convertedPoints.forEach((p, i) => {
              graphGroup
                .append("circle")
                .attr("aria-label", "polygon-vertex")
                .attr("class", "annotation polygon-vertex disable-on-modifier")
                .attr("cx", p.x)
                .attr("cy", p.y)
                .attr("r", 3)
                .attr("fill", "grey")
                .attr("stroke", "grey")
                .attr("stroke-width", 1)
                .attr("style", `pointer-events: ${pointerEvent}; cursor: move`);

              // Transparent vertex handles (functional)
              graphGroup
                .append("circle")
                .attr("aria-label", "polygon-vertex-handle")
                .attr("class", "annotation polygon-vertex disable-on-modifier")
                .attr("cx", p.x)
                .attr("cy", p.y)
                .attr("r", 15)
                .attr("fill", "transparent")
                .attr("stroke", "transparent")
                .attr("stroke-width", 1)
                .attr("style", `pointer-events: ${pointerEvent}; cursor: move`)
                .datum(polygon)
                .call(getVertexHandler(i))
                .on("contextmenu", handleContextMenu);
            });
          } else {
            // Start vertex rendering
            graphGroup
              .append("circle")
              .attr("aria-label", "polygon-vertex")
              .attr("class", "annotation polygon-vertex disable-on-modifier")
              .attr("cx", convertedPoints[0].x)
              .attr("cy", convertedPoints[0].y)
              .attr("r", 3)
              .attr("fill", "grey")
              .attr("stroke", "grey")
              .attr("stroke-width", 1)
              .attr("style", `pointer-events: ${pointerEvent}; cursor: move`);
          }
        }
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
    subplot,
  ]); // forceUpdate is required here to keep tooling correctly positioned

  return <div />;
};
