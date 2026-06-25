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
  const dragOffset = useRef(0);

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
      end(_x, _y) {},
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

      // Minimum width in data units: 0.1% of current x-range
      const [xMin, xMax] = xAxis.range;
      const MIN_WIDTH_FRACTION = 0.001; // 0.1%
      const minWidth = (xMax - xMin) * MIN_WIDTH_FRACTION;

      // Minimum height in data units: 0.1% of current x-range
      const [yMin, yMax] = yAxis.range;
      const MIN_HEIGHT_FRACTION = 0.001; // 0.1%
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

      // Create a line and a transparent drag handle for each VSpan
      for (const boundingBox of annotations) {
        if (boundingBox.type !== TimeSeriesAnnotationType.BOUNDING_BOX) continue;
        const opacity = boundingBox.selected ? 0.8 : 0.5;

        // pixel positions for the two data boundaries
        const px0 = xAxis.d2p(boundingBox.points[0].x);
        const py0 = yAxis.d2p(boundingBox.points[0].y);
        const px1 = xAxis.d2p(boundingBox.points[1].x);
        const py1 = yAxis.d2p(boundingBox.points[1].y);
        const pointerEvent = isDrawing || !editMode ? "none" : "all";

        const originPoint: TimeSeriesAnnotationPoint = {
          x: Math.min(px0, px1),
          y: Math.min(py0, py1)
        }
        
        const boxWidth = Math.abs(px1 - px0)
        const boxHeight = Math.abs(py1 - py0)

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
