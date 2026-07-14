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
import { ToastQueue } from "@adobe/react-spectrum";
import * as d3 from "d3";
import { useEffect, useRef } from "react";
import { useContextMenu } from "react-contexify";

export const Polygon = ({ plotId, plotReady }: ToolingProps) => {
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
  const isUpdatingPolygon = useRef(false);
  const hasDeletedVertex = useRef(false);

  // Hook to trigger the context provider to render context menu
  const { show } = useContextMenu({
    id: TIME_SERIES_ANNOTATION_MENU,
  });

  useEffect(() => {
    const toolingCallbacks: ToolingCallbacks = {
      start: (x, y, label, axisSize) => {
        if (isUpdatingPolygon.current) {
            if (!currentAnnotation.current) {
                console.error("Could not find current annotation to add point to")
                return
            }

            if (currentAnnotation.current.type !== TimeSeriesAnnotationType.POLYGON) {
                console.error("Could not add point to non-polygon annotation")
                return
            }

            const pointArrayLength = currentAnnotation.current.points.length
            const closeThreshold = { x: axisSize.x * 0.02, y: axisSize.y * 0.02};

            if (Math.abs(x - currentAnnotation.current.points[0].x) < closeThreshold.x && Math.abs(y - currentAnnotation.current.points[0].y) < closeThreshold.y) {
                if (pointArrayLength > 4) {
                    console.log("End 1")
                    setOngoingAction(false); // Since this has hover functionality the callback must end the ongoing action
                    isUpdatingPolygon.current = false
                    currentAnnotation.current.points.splice(pointArrayLength-2, 2)
                    updateAnnotation(currentAnnotation.current);
                }
                return
            }

            currentAnnotation.current.points[pointArrayLength - 2] = { x, y };
            currentAnnotation.current.points.splice(pointArrayLength-1, 0, {x, y})
            updateAnnotation(currentAnnotation.current);
            return
        }
        
        const annotation = createAnnotation(
          TimeSeriesAnnotationType.POLYGON,
          label,
        );
        currentAnnotation.current = annotation;
        annotation.points.push({ x, y });
        annotation.points.push({ x, y });
        annotation.points.push({ x, y });
        isUpdatingPolygon.current = true;
        addAnnotation(annotation);
      },
      move(_x, _y) {},
      end(_x, _y) {},
      hover(x, y) {
        if (!currentAnnotation.current || !isUpdatingPolygon.current) return;
        const pointArrayLength = currentAnnotation.current.points.length
        currentAnnotation.current.points[pointArrayLength - 2] = { x, y };
        updateAnnotation(currentAnnotation.current);
      },
    };
    registerTooling(TimeSeriesAnnotationType.POLYGON, toolingCallbacks);
  }, [addAnnotation, createAnnotation, registerTooling, setOngoingAction, updateAnnotation]);

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

      const getVertexHandler = (index: number) =>
        d3.drag<SVGCircleElement, TimeSeriesAnnotation>()
          .on("start", function (_, d) { 
            setOngoingAction(true);
            const onKeyDown = (e: KeyboardEvent) => {
              if (e.key === "Delete") {

                if (d.points.length < 4) {
                  ToastQueue.info(
                    "Cannot delete a vertex for a polygon with fewer than four points, click on the polygon and press delete to remove the annotation",
                    { timeout: 5000 },
                  );
                  return
                }

                d.points.splice(index, 1);
                updateAnnotation(d);
                hasDeletedVertex.current = true;
              }
            };

          // Store the handler on the element so we can remove it later
          (this as SVGCircleElement).__deleteHandler = onKeyDown;

          window.addEventListener("keydown", onKeyDown); 
          })
          .on("drag", (event, d) => {
            if (hasDeletedVertex.current) return // If during this drag the vertex was deleted no action should be taken
            d.points[index] = { x: xAxis.p2d(event.x), y: yAxis.p2d(event.y) };
            updateAnnotation(d);
          })
          .on("end", function () {
            hasDeletedVertex.current = false;
            setOngoingAction(false)
            console.log("End 2")
            const handler = (this as SVGCircleElement).__deleteHandler;
            if (handler) {
              window.removeEventListener("keydown", handler);
            }
          });

      for (const polygon of annotations) {
        if (polygon.type !== TimeSeriesAnnotationType.POLYGON)
          continue;
        const opacity = polygon.selected ? 0.8 : 0.5;
        const pointerEvent = isDrawing || !editMode ? "none" : "all";
        const isInProgress = currentAnnotation.current?.id === polygon.id && isUpdatingPolygon.current;

        const convertedPoints = polygon.points.map((point) => ({
            x: xAxis.d2p(point.x),
            y: yAxis.d2p(point.y),
        }))

        const categoryId = `${polygon.type}_${polygon.label}`;
        const color = categories.get(categoryId)?.color || "black";

        graphGroup
            .append("polygon")
            .attr("aria-label", "polygon")
            .attr(
                "class",
                "annotation polygon cursor-grab disable-on-modifier",
            )
            .attr("points", convertedPoints.map(p => (`${p.x}, ${p.y}`)).join(" "))
            .attr("fill", color)
            .attr("opacity", opacity)
            .attr("style", `pointer-events: ${pointerEvent}`)
            .attr("stroke-width", 1)
            .attr("stroke", "gray")
            .datum(polygon)
            .on("click", handleClick)
            .on("contextmenu", handleContextMenu);

        if (editMode) {
          if (!isInProgress) {
            // Edge hit targets — appended before vertex circles so vertices take priority
            convertedPoints.forEach((p, i) => {
              const next = convertedPoints[(i + 1) % convertedPoints.length];
              graphGroup
                .append("line")
                .attr("aria-label", "polygon-edge-handle")
                .attr("class", "annotation polygon-edge disable-on-modifier")
                .attr("x1", p.x).attr("y1", p.y)
                .attr("x2", next.x).attr("y2", next.y)
                .attr("stroke", "transparent")
                .attr("stroke-width", 10)
                .attr("style", `pointer-events: ${pointerEvent}; cursor: cell`)
                .datum(polygon)
                .on("click", (event, d) => {
                  const x = xAxis.p2d((p.x + next.x) / 2)
                  const y = yAxis.p2d((p.y + next.y) / 2)

                  d.points.splice(i + 1, 0, { x, y });
                  updateAnnotation(d);
                });
            });

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
                .attr("style", `pointer-events: ${pointerEvent}; cursor: move`)

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
  ]); // forceUpdate is required here to keep tooling correctly positioned

  return <div />;
};
