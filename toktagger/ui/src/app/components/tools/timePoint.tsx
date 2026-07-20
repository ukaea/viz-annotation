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

export const TimePoint = ({ plotId, plotReady }: ToolingProps) => {
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
          TimeSeriesAnnotationType.TIME_POINT,
          label,
        );
        currentAnnotation.current = annotation;
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
        currentAnnotation.current.points[0] = { x, y };
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
    registerTooling(TimeSeriesAnnotationType.TIME_POINT, toolingCallbacks);
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

      const graphGroup = d3.select(overplot);
      graphGroup.selectAll(".time-point").remove(); // All VSpans are removed each render cycle

      // Create a line and a transparent drag handle for each VSpan
      for (const vspan of annotations) {
        if (vspan.type !== TimeSeriesAnnotationType.TIME_POINT) continue;
        const opacity = vspan.selected ? 0.8 : 0.5;

        const drag = d3
          .drag<SVGRectElement, TimeSeriesAnnotation>()
          .on("start", function (event, d) {
            selectAnnotations([d.id]);
            dragOffset.current = xaxis.d2p(d.points[0].x) - event.x;
            setOngoingAction(true);
          })
          .on("drag", function (event, d) {
            const newX = event.x + dragOffset.current;
            d3.select(this).attr("x", newX);

            const x = xaxis.p2d(newX); // The context provider stores the decimal value rather than pixel
            d.points[0].x = x;
            updateAnnotation(d); // Global refresh must be triggered to update all linked plots
          })
          .on("end", function (_event, _d) {
            setOngoingAction(false);
          });

        const categoryId = `${vspan.type}_${vspan.label}`;
        const color = categories.get(categoryId)?.color || "black";

        const x = xaxis.d2p(vspan.points[0].x);
        const pointerEvent = isDrawing || !editMode ? "none" : "all";
        graphGroup
          .append("line")
          .attr("class", "annotation time-point disable-on-modifier")
          .attr("x1", x)
          .attr("x2", x)
          .attr("y1", upperLimit)
          .attr("y2", upperLimit + height)
          .attr("opacity", opacity)
          .attr("stroke", color)
          .attr("stroke-width", 6)
          .attr("style", `pointer-events: ${pointerEvent}`)
          .style("cursor", "move");

        graphGroup
          .append("rect")
          .attr("aria-label", "time-point")
          .attr("class", "annotation time-point disable-on-modifier")
          .attr("x", x - 10)
          .attr("y", upperLimit)
          .attr("width", 20)
          .attr("height", height)
          .attr("fill", "transparent")
          .attr("style", `pointer-events: ${pointerEvent}`)
          .style("cursor", "move")
          .datum(vspan)
          .call(drag)
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
