"use client";

import { useContextMenuProvider } from "@/app/components/providers/annotation-provider";
import { useSample } from "@/app/contexts/SampleContext";
import { arrayMax, arrayMin } from "@/app/utils";
import {
  Annotation,
  BoundingBoxAnnotation,
  BoundingBoxAnnotationSchema,
  PolygonAnnotation,
  PolygonAnnotationSchema,
  Profile2DViewParams,
  VSpan,
  Zone,
} from "@/types";
import Plotly, {
  Config,
  Layout,
  PlotData,
  relayout,
  react,
  PlotRelayoutEvent,
} from "plotly.js-dist-min";
import React, { useCallback, useEffect, useRef, useState } from "react";
import { Item, Submenu } from "react-contexify";
import { useVSpanContext } from "../providers/vpsan-provider";
import { useZoneContext } from "../providers/zone-provider";
import { useBoundingBoxContext } from "../providers/bounding-box-provider";
import { usePolygonContext } from "../providers/polygon-provider";

type InjectedProps = {
  plotId: string;
  plotReady: boolean;
  forceUpdate: number;
  selectedXRange: [number, number] | null;
};

type ShapeContextMenuProps = {
  x: number;
  y: number;
  xRange: number;
  yRange: number;
  xLimits: [number, number];
  yLimits: [number, number];
  shapeIndex: number | undefined;
  selectedShapeIndices: number[];
};

interface PlotConfiguration {
  data: Partial<PlotData>[];
  layout: Partial<Layout>;
  config?: Partial<Config>;
}

type PlotlyWidgetProps = {
  plotId?: string;
  plotConfig: PlotConfiguration;
  rescaleOnZoom?: boolean;
  children:
    | React.ReactElement<InjectedProps>
    | React.ReactElement<InjectedProps>[];
};

function getBoxBounds(eventData: PlotRelayoutEvent) {
  return {
    x0: Math.min(...eventData.range.x),
    x1: Math.max(...eventData.range.x),
    y0: Math.min(...eventData.range.y2),
    y1: Math.max(...eventData.range.y2),
  };
}

function shapeIntersectsBox(
  shape: Plotly.Shape,
  box: { x0: number; x1: number; y0: number; y1: number },
): boolean {
  if (shape.type === "rect") {
    // For rectangles, check if the shape is entirely within the box
    const shapeX0 = Math.min(shape.x0 as number, shape.x1 as number);
    const shapeX1 = Math.max(shape.x0 as number, shape.x1 as number);
    const shapeY0 = Math.min(shape.y0 as number, shape.y1 as number);
    const shapeY1 = Math.max(shape.y0 as number, shape.y1 as number);

    // Rectangle is entirely within the box if all corners are inside
    const xContained = shapeX0 >= box.x0 && shapeX1 <= box.x1;
    const yContained = shapeY0 >= box.y0 && shapeY1 <= box.y1;

    return xContained && yContained;
  } else if (shape.type === "path" && shape.path) {
    // For polygons (path shapes), extract points and check if all are within the box
    const pathCommands = shape.path.match(/[ML][^MLZ]+/g);
    if (!pathCommands) return false;

    const points: { x: number; y: number }[] = [];
    pathCommands.forEach((command) => {
      const coords = command
        .slice(1)
        .trim()
        .split(",")
        .map((coord) => parseFloat(coord));
      if (coords.length === 2) {
        points.push({ x: coords[0], y: coords[1] });
      }
    });

    if (points.length === 0) return false;

    // Check if all points of the polygon are inside the selection box
    const allPointsInside = points.every(
      (point) =>
        point.x >= box.x0 &&
        point.x <= box.x1 &&
        point.y >= box.y0 &&
        point.y <= box.y1,
    );

    return allPointsInside;
  }

  return false;
}

const shapeToAnnotation = (
  shape: Plotly.Shape,
  signalName: string | null,
): Annotation => {
  if (shape.type === "rect") {
    const boundingBox: BoundingBoxAnnotation =
      BoundingBoxAnnotationSchema.parse({
        x_min: shape.x0 as number,
        y_min: shape.y0 as number,
        width: (shape.x1 as number) - (shape.x0 as number),
        height: (shape.y1 as number) - (shape.y0 as number),
        signal_name: signalName,
        label: shape.meta?.label || "Unknown",
        created_by: "manual",
        type: "bounding_box",
      });
    return boundingBox;
  } else if (shape.type === "path" && shape.path) {
    // extract points from path string
    const pathCommands = shape.path.match(/[ML][^MLZ]+/g); // match 'M x y' or 'L x y'
    if (pathCommands) {
      const xPoints: number[] = [];
      const yPoints: number[] = [];
      pathCommands.forEach((command) => {
        const coords = command
          .slice(1)
          .trim()
          .split(",")
          .map((coord) => parseFloat(coord));
        if (coords.length === 2) {
          xPoints.push(coords[0]);
          yPoints.push(coords[1]);
        }
      });
      if (xPoints.length > 0 && yPoints.length > 0) {
        const polygon: PolygonAnnotation = PolygonAnnotationSchema.parse({
          segmentation: [xPoints.flatMap((x, i) => [x, yPoints[i]])],
          area: 0, // area can be computed server-side if needed
          bbox: [
            arrayMin(xPoints),
            arrayMin(yPoints),
            arrayMax(xPoints) - arrayMin(xPoints),
            arrayMax(yPoints) - arrayMin(yPoints),
          ],
          signal_name: signalName,
          label: shape.meta?.label || "Unknown",
          created_by: "manual",
          type: "polygon",
        });
        return polygon;
      }
    }
  }
  throw new Error("Unsupported shape type for annotation conversion");
};

/**
 * Component that handles the plotly and context menu rendering
 *
 * @param data Disruption time series data
 * @param plotId Set plot id externally in case multiple plots are used
 */
export const PlotlyWidget = ({
  plotId: externalId,
  plotConfig: {
    data,
    layout,
    config = {
      modeBarButtons: [
        [
          "toImage",
          "zoom2d",
          "select2d",
          "pan2d",
          "autoScale2d",
          "resetScale2d",
        ],
      ],
      dragmode: "pan",
      displaylogo: false,
      displayModeBar: true,
      scrollZoom: true,
      responsive: true,
    },
  },
  rescaleOnZoom = true,
  children,
}: PlotlyWidgetProps) => {
  const { viewParams, setAnnotations } = useSample();
  const { vspans, handleVSpanDelete } = useVSpanContext();
  const { zones, handleZoneDelete } = useZoneContext();
  const { categories: polygonCategories } = usePolygonContext();
  const { categories: boundingBoxCategories } = useBoundingBoxContext();

  const [selectedXRange, setSelectedXRange] = useState<[number, number] | null>(
    null,
  );
  const [updateTools, setUpdateTools] = useState(0);
  const [plotReady, setPlotReady] = useState(false);
  const isDraggingRef = useRef(false);
  const plotId = externalId || "time-series"; // Facilitate an external or default ID

  const { toolingCallbacks, disableToolingInteraction } =
    useContextMenuProvider();

  const { show: showContextMenu, registerMenuItem } = useContextMenuProvider();
  const showContextMenuRef = useRef(showContextMenu);

  // Keep the ref in sync with the latest show function
  useEffect(() => {
    showContextMenuRef.current = showContextMenu;
  }, [showContextMenu]);

  const allowRelayout = useRef(true);

  const triggerToolUpdate = () => {
    setUpdateTools((current) => (current + 1) % 100);
  };

  const updateShapeAnnotations = useCallback(
    (shapes: Plotly.Shape[]) => {
      const signalName =
        (viewParams as Profile2DViewParams).signal_name || null;

      const newAnnotations: Annotation[] = shapes.map((shape) => {
        const annotation = shapeToAnnotation(shape, signalName);
        return annotation;
      });

      // Update annotations with new polygons/bounding boxes
      setAnnotations((previousAnnotations: Annotation[]) => {
        const otherAnnotations = previousAnnotations.filter(
          (annotation: Annotation) =>
            annotation.type !== "polygon" && annotation.type !== "bounding_box",
        );
        return otherAnnotations.concat(newAnnotations);
      });
    },
    [viewParams, setAnnotations],
  );

  const handleShapeDelete = useCallback(
    (shapeIndex: number) => {
      const plot = document.getElementById(plotId) as Plotly.PlotlyHTMLElement;

      if (!plot) {
        return;
      }

      const shapes = plot.layout.shapes as Plotly.Shape[] | null;
      if (!shapes) {
        return;
      }

      if (shapeIndex < 0 || shapeIndex >= shapes.length) {
        return;
      }

      const updatedShapes = shapes.filter(
        (_shape, index) => index !== shapeIndex,
      );

      const newLayout = { ...plot.layout, shapes: updatedShapes };
      Plotly.react(plot, plot.data, newLayout, plot.config);
      updateShapeAnnotations(updatedShapes);
    },
    [plotId, updateShapeAnnotations],
  );
  // Main plotly rendering
  useEffect(() => {
    const overplots: string[] = [];

    const renderZones = (plot: Plotly.PlotlyHTMLElement) => {
      // Get all subplot elements and extract the subplot name (xy for example) from the class list
      const subplots = plot.querySelectorAll(".subplot");
      const subplotNames = [...subplots].map((el) =>
        [...el.classList].find((cls) => cls !== "subplot"),
      );

      // For each subplot identified generate a D3 overplot with the subplot name appended so that tooling can reference it
      subplotNames.forEach((coordinateSystem) => {
        const subplot = plot
          .querySelector(`.subplot.${coordinateSystem}`)
          ?.querySelector(".overplot")
          ?.querySelector(`.${coordinateSystem}`) as HTMLElement;
        if (!subplot) {
          console.error("Cannot locate plotly subplot");
          return;
        }

        if (!subplot.querySelector(`.${plotId}-overplot-${coordinateSystem}`)) {
          // ensure only one custom overlay group is present
          const svg = document.createElementNS(
            "http://www.w3.org/2000/svg",
            "g",
          );
          svg.setAttribute("class", `${plotId}-overplot-${coordinateSystem}`);
          svg.setAttribute("fill", "none");
          subplot.appendChild(svg);
          overplots.push(`${plotId}-overplot-${coordinateSystem}`); // Store overplots for removal
        }
      });

      // Use setTimeout to ensure DOM has fully updated before signaling ready
      setPlotReady(true);
    };

    // Sets the y axis range required for the current x range for each subplot
    const rescale = (x0?: number, x1?: number) => {
      const plot = document.getElementById(plotId) as Plotly.PlotlyHTMLElement;
      if (!plot) {
        return;
      }

      if (!allowRelayout.current) return; // Prevents relayout triggering itself

      if (data.length === 0) {
        return;
      }
      allowRelayout.current = false;

      // If no x range is passed, then the min/max is used
      if (!x0) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        x0 = (plot as any)._fullData[0]._extremes.x.min[0].val as number;
      }
      if (!x1) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        x1 = (plot as any)._fullData[0]._extremes.x.max[0].val as number;
      }

      let configUpdate = {};

      // Ensure each data set is handled (ensures all subplots are zoomed correctly)
      data.forEach((dataSet) => {
        let yAxisID = "";

        if (dataSet.yaxis) {
          // Find the y axis ID relating to this subplot
          const locatedID = dataSet.yaxis.match(/y(.*)$/)?.[1];
          if (locatedID) {
            yAxisID = locatedID;
          }
        }

        const xArray = (dataSet as PlotData).x as number[];
        const yArray = (dataSet as PlotData).y as number[];

        // Find min and max y data values
        const yValues: number[] = [];
        for (let i = 0; i < xArray.length; i++) {
          const xVal = xArray[i];
          if (xVal >= x0 && xVal <= x1) {
            yValues.push(yArray[i]);
          }
        }

        if (yValues.length > 0) {
          const yMin = arrayMin(yValues);
          const yMax = arrayMax(yValues);
          const offset = 0.1 * (yMax - yMin); // 10 % offset

          configUpdate = {
            ...configUpdate,
            [`yaxis${yAxisID}.range`]: [yMin - offset, yMax + offset],
          };
        }
      });

      relayout(plot, configUpdate);

      // Debounce the relayout calls
      setTimeout(() => {
        allowRelayout.current = true;
      }, 100);
    };

    const relayoutHandler = (eventData: PlotRelayoutEvent) => {
      if ("shapes" in eventData) {
        // shapes may have been modified - update annotations
        const shapes = eventData.shapes as Plotly.Shape[];
        updateShapeAnnotations(shapes);
      }

      if (rescaleOnZoom) {
        // This makes use of the first graph displayed but this should be fine
        if ("xaxis.range[0]" in eventData && "xaxis.range[1]" in eventData) {
          // for zoom and pan events
          rescale(eventData["xaxis.range[0]"], eventData["xaxis.range[1]"]);
        } else if ("xaxis.range" in eventData) {
          // for range slider events
          rescale(eventData["xaxis.range"][0], eventData["xaxis.range"][1]);
        } else {
          rescale(); // for initial load & autoscale
        }
      }
      triggerToolUpdate();
    };

    const plot = document.getElementById(plotId) as Plotly.PlotlyHTMLElement;

    if (!plot) {
      return;
    }

    const initGraph = async () => {
      react(plot, data, layout, config).then(renderZones);

      plot.removeAllListeners("plotly_relayout"); // remove any existing listeners
      plot.removeAllListeners("plotly_doubleclick");
      plot.removeAllListeners("plotly_selected");
      plot.removeAllListeners("plotly_deselect");
      plot.on("plotly_relayout", relayoutHandler); // attach listener so it can be removed
      plot.on("plotly_doubleclick", rescale);

      // attach listener for selection events
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      plot.on("plotly_selected", function (eventData: any) {
        if (eventData && eventData.range) {
          setSelectedXRange(eventData.range.x);
        }

        const shapes = plot.layout.shapes as Plotly.Shape[] | null;
        if (!shapes) return;

        if (!eventData || !eventData.range) return;

        const selectionBounds = getBoxBounds(eventData);
        const updatedShapes = shapes.map((shape) => {
          if (shapeIntersectsBox(shape, selectionBounds)) {
            // Mark shape as selected
            shape.line.color = "rgba(150, 150, 150, 0.3)"; // example selected style
            shape.meta.selected = true;
          }
          return shape;
        });

        const newLayout = { ...plot.layout, shapes: updatedShapes };
        Plotly.react(plot, plot.data, newLayout, plot.config);
      });
      plot.on("plotly_deselect", function () {
        setSelectedXRange(null);
        const shapes = plot.layout.shapes as Plotly.Shape[] | null;
        if (!shapes) return;

        const updatedShapes = shapes.map((shape) => {
          // Reset shape style
          shape.line.color = "rgba(150, 150, 150, 1.0)"; // example default style
          shape.meta.selected = false;
          return shape;
        });

        const newLayout = {
          ...plot.layout,
          shapes: updatedShapes,
          selections: [],
        };
        Plotly.react(plot, plot.data, newLayout, plot.config);
      });
    };

    initGraph();

    if (!disableToolingInteraction) {
      plot.style.pointerEvents = "auto";
    } else {
      plot.style.pointerEvents = "none";
    }

    return () => {
      // cleanup on unmount / Fast-Refresh
      overplots.forEach((overplot) => {
        plot?.querySelector(`.${overplot}`)?.remove(); // remove custom overlay group
      });
      setPlotReady(false); // reset ready state
    };
  }, [
    plotId,
    config,
    data,
    layout,
    allowRelayout,
    disableToolingInteraction,
    rescaleOnZoom,
    updateShapeAnnotations,
  ]);

  useEffect(() => {
    const plot = document.getElementById(plotId) as Plotly.PlotlyHTMLElement;
    if (!plot) {
      return;
    }
    if (allowRelayout.current) {
      relayout(plot, { selections: [] }); // clear selection
      setSelectedXRange(null);
    }
  }, [config, plotId]);

  // Handles context menu creation
  useEffect(() => {
    if (!plotReady) {
      // Plot may not have loaded yet - this will rerun after loading
      return;
    }

    const plot = document.getElementById(plotId);

    if (!plot) {
      console.error("Could not locate plot to assign context menu");
      return;
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    function getClickData(event: MouseEvent, plot: any): [number, number] {
      const xaxis = plot._fullLayout.xaxis; // x-axis descriptor
      const yaxis = plot._fullLayout.yaxis; // y-axis descriptor

      const bb = (event.target as HTMLElement).getBoundingClientRect();
      const relX = event.clientX - bb.left; // click X in pixels, relative to plot
      const relY = event.clientY - bb.top; // click Y in pixels, relative to plot

      // Coordinates in data space
      const x = xaxis.p2d(relX); // data-space X at click
      const y = yaxis.p2d(relY); // data-space Y at click

      return [x, y];
    }

    /* 
        Context-menu dispatcher

            Converts the mouse click (pixel-space) to data-space coordinates (x, y) using Plotly’s axis converters.

            Derives the current axis ranges (xRange, yRange) so tools can size new elements as a fraction of the view, independent of zoom level.
    
            information delivered to the menu is  { x, y, xScale, yScale, xRange, yRange, xLimits: [xMin, xMax], yLimits: [yMin, yMax] }

            The dispatcher now auto-detects which subplot was clicked (via the element data-subplot attribute or nearest .subplot group) and 
            picks the matching xaxisN / yaxisN, so the props are correct for any subplot.
        */
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    function handleContextMenu(event: MouseEvent, plot: any) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      let xaxis: any; // will be assigned to the subplot-specific or primary x-axis below
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      let yaxis: any; // will be assigned to the subplot-specific or primary y-axis below

      const bb = (event.target as HTMLElement).getBoundingClientRect();
      const relX = event.clientX - bb.left; // click X in pixels, relative to plot
      const relY = event.clientY - bb.top; // click Y in pixels, relative to plot

      /* 
            determine local axes for the subplot clicked
            Prefer the data-subplot attribute available on drag layers;
            */
      const subplotId = (event.target as HTMLElement).dataset.subplot; // e.g. "x2y2"
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

      // compute full data range spans from axis.range
      const [xMin, xMax] = xaxis.range as [number, number]; // data-space limits on x
      const [yMin, yMax] = yaxis.range as [number, number]; // data-space limits on y
      const xRange = xMax - xMin; // total span on x axis
      const yRange = yMax - yMin; // total span on y axis

      const shapes = plot.layout.shapes;
      let shapeIndex: number | undefined = undefined;
      const selectedShapeIndices: number[] = [];

      // First, collect all selected shapes
      for (let i = 0; i < shapes?.length; i++) {
        const shape = shapes[i];
        if (shape.meta && shape.meta.selected === true) {
          selectedShapeIndices.push(i);
        }
      }

      // Then check if a specific shape was clicked.
      // Each shape may reference a different subplot (e.g. yref: "y2"). We look up
      // the drag element for that subplot and use its bounding rect as the pixel
      // reference for p2d(), which is calibrated relative to the drag layer origin.
      for (let i = 0; i < shapes?.length; i++) {
        const shape = shapes[i];

        const xSuffix =
          (shape.xref as string | undefined)?.replace("x", "") ?? "";
        const ySuffix =
          (shape.yref as string | undefined)?.replace("y", "") ?? "";

        const fullLayout = (
          plot as {
            _fullLayout: Record<string, { p2d: (v: number) => number }>;
          }
        )._fullLayout;
        const shapeXaxis = fullLayout[`xaxis${xSuffix}`] ?? fullLayout.xaxis;
        const shapeYaxis = fullLayout[`yaxis${ySuffix}`] ?? fullLayout.yaxis;

        const subplotKey = `x${xSuffix}y${ySuffix}`;
        const shapeDragEl = (plot as HTMLElement).querySelector(
          `.drag[data-subplot="${subplotKey}"]`,
        ) as HTMLElement | null;
        const shapeBB = shapeDragEl ? shapeDragEl.getBoundingClientRect() : bb; // fall back to whatever the click landed on
        const sx = shapeXaxis.p2d(event.clientX - shapeBB.left);
        const sy = shapeYaxis.p2d(event.clientY - shapeBB.top);

        if (shape.type === "rect") {
          const rectXMin = Math.min(shape.x0, shape.x1);
          const rectXMax = Math.max(shape.x0, shape.x1);
          const rectYMin = Math.min(shape.y0, shape.y1);
          const rectYMax = Math.max(shape.y0, shape.y1);

          if (
            sx >= rectXMin &&
            sx <= rectXMax &&
            sy >= rectYMin &&
            sy <= rectYMax
          ) {
            shapeIndex = i;
            break;
          }
        } else if (shape.type === "path" && shape.path) {
          const pathCommands = shape.path.match(/[ML][^MLZ]+/g);
          if (pathCommands) {
            const xPoints: number[] = [];
            const yPoints: number[] = [];
            pathCommands.forEach((command) => {
              const coords = command
                .slice(1)
                .trim()
                .split(",")
                .map((coord) => parseFloat(coord));
              if (coords.length === 2) {
                xPoints.push(coords[0]);
                yPoints.push(coords[1]);
              }
            });

            const polyXMin = Math.min(...xPoints);
            const polyXMax = Math.max(...xPoints);
            const polyYMin = Math.min(...yPoints);
            const polyYMax = Math.max(...yPoints);

            if (
              sx >= polyXMin &&
              sx <= polyXMax &&
              sy >= polyYMin &&
              sy <= polyYMax
            ) {
              shapeIndex = i;
              break;
            }
          }
        }
      }

      const menuProps: ShapeContextMenuProps = {
        x,
        y, // generic data-space click position
        xRange,
        yRange, // current axis spans
        xLimits: [xMin, xMax],
        yLimits: [yMin, yMax], // explicit axis limits
        shapeIndex, // will be undefined if no bbox was clicked
        selectedShapeIndices, // array of all selected shape indices
      };

      showContextMenuRef.current({
        event,
        props: menuProps,
      });
    }

    const dragElements = plot.querySelectorAll<HTMLDivElement>(".drag");

    if (dragElements.length === 0) {
      console.error("Could not locate drag element to assign context menu");
      return;
    }

    const contextHandler = (event: MouseEvent) => {
      //  wrap handler so we can remove it
      event.preventDefault(); // Prevent default context menu
      const isRightClickEvent = event.button === 2 && !event.ctrlKey;
      if (isRightClickEvent) {
        handleContextMenu(event, plot);
      }
    };

    const startToolCreation = (event: MouseEvent) => {
      if (toolingCallbacks && event.ctrlKey) {
        isDraggingRef.current = true;
        const [x, y] = getClickData(event, plot);
        toolingCallbacks.start(x, y);
      }
    };

    // This is a backup listener in case the user lifts the control key first - this isn't ideal as a final update won't be sent
    const cancelToolCreation = (event: KeyboardEvent) => {
      if (event.key === "Alt" && toolingCallbacks && isDraggingRef.current) {
        isDraggingRef.current = false;
      }
    };

    const finishToolCreation = (event: MouseEvent) => {
      if (toolingCallbacks && isDraggingRef.current) {
        isDraggingRef.current = false;
        const [x, y] = getClickData(event, plot);
        toolingCallbacks.end(x, y);
      }
    };

    const updateTool = (event: MouseEvent) => {
      if (toolingCallbacks && isDraggingRef.current) {
        const [x, y] = getClickData(event, plot);
        toolingCallbacks.move(x, y);
      }
    };

    // Delete selected spans on Delete/Backspace keypress
    document.addEventListener("keydown", (e) => {
      if (e.key === "Delete" || e.key == "Backspace") {
        const selectedSpans = vspans.filter((span: VSpan) => span.selected);
        for (const span of selectedSpans) {
          handleVSpanDelete(span);
        }

        const selectedZones = zones.filter((zone: Zone) => zone.selected);
        for (const zone of selectedZones) {
          handleZoneDelete(zone);
        }

        const plot = document.getElementById(
          plotId,
        ) as Plotly.PlotlyHTMLElement;
        if (!plot) return;

        const shapes = plot?.layout.shapes as Plotly.Shape[] | null;
        if (!shapes) return;

        const updatedShapes = shapes.filter((_shape) => {
          // Keep shape if not selected
          return !(_shape.meta && _shape.meta.selected === true);
        });

        const newLayout = { ...plot.layout, shapes: updatedShapes };
        Plotly.react(plot, plot.data, newLayout, plot.config);
        updateShapeAnnotations(updatedShapes);
      }
    });

    dragElements.forEach((dragElement) => {
      dragElement.addEventListener("mousedown", startToolCreation);
      dragElement.addEventListener("mouseup", finishToolCreation);
      dragElement.addEventListener("mousemove", updateTool);
    });

    // Add context menu to the entire plot container to catch events even when selection overlays are present
    plot.addEventListener("contextmenu", contextHandler);

    document.addEventListener("keyup", cancelToolCreation);

    return () => {
      // remove listener on effect cleanup
      dragElements.forEach((dragElement) => {
        dragElement.removeEventListener("mousedown", startToolCreation);
        dragElement.removeEventListener("mouseup", finishToolCreation);
        dragElement.removeEventListener("mousemove", updateTool);
      });
      plot.removeEventListener("contextmenu", contextHandler);
      document.removeEventListener("keyup", cancelToolCreation);
    };
  }, [
    plotId,
    plotReady,
    toolingCallbacks,
    updateTools,
    setAnnotations,
    rescaleOnZoom,
    viewParams,
    handleVSpanDelete,
    vspans,
    handleZoneDelete,
    zones,
    updateShapeAnnotations,
  ]);

  const handleShapeContextMenuClick = useCallback(
    (id: string | undefined, params: ShapeContextMenuProps) => {
      if (!id) {
        return;
      }

      const plot = document.getElementById(plotId) as Plotly.PlotlyHTMLElement;

      if (!plot) {
        return;
      }

      const shapes = plot.layout.shapes as Plotly.Shape[] | undefined;
      if (!shapes) {
        return;
      }

      const shapeIndex = params.shapeIndex;
      if (
        shapeIndex === undefined ||
        shapeIndex < 0 ||
        shapeIndex >= shapes.length
      ) {
        return;
      }

      const shape = shapes[shapeIndex];
      if (shape.type === "rect") {
        const category = boundingBoxCategories?.find((cat) => cat.name === id);
        if (!category) return;

        const updatedShapes = [...shapes];
        updatedShapes[shapeIndex] = {
          ...updatedShapes[shapeIndex],
          meta: { label: category.name },
          line: { color: "rgb(150, 150, 150)", width: 5 },
          fillcolor: category.color
            ?.replace("rgb(", "rgba(")
            .replace(")", ", 0.5)"),
        };

        const newLayout = { ...plot.layout, shapes: updatedShapes };
        Plotly.react(plot, plot.data, newLayout, plot.config);
        updateShapeAnnotations(updatedShapes);
      } else if (shape.type === "path") {
        const category = polygonCategories?.find((cat) => cat.name === id);
        if (!category) return;

        const updatedShapes = [...shapes];
        updatedShapes[shapeIndex] = {
          ...updatedShapes[shapeIndex],
          meta: { label: category.name },
          line: { color: "rgb(150, 150, 150)", width: 5 },
          fillcolor: category.color
            ?.replace("rgb(", "rgba(")
            .replace(")", ", 0.5)"),
        };

        const newLayout = { ...plot.layout, shapes: updatedShapes };
        Plotly.react(plot, plot.data, newLayout, plot.config);
        updateShapeAnnotations(updatedShapes);
      }
    },
    [plotId, boundingBoxCategories, polygonCategories, updateShapeAnnotations],
  );

  useEffect(() => {
    const deleteItem = (
      <Item
        id="delete-shape"
        key="delete-shape"
        onClick={({ props }) => {
          const shapeIndex = props?.shapeIndex;
          if (shapeIndex !== undefined) {
            handleShapeDelete(shapeIndex);
          }
        }}
        hidden={({ props }) => props?.shapeIndex === undefined}
      >
        Delete Shape
      </Item>
    );

    const deleteSelectedItem = (
      <Item
        id="delete-selected-shapes"
        key="delete-selected-shapes"
        onClick={({ props }) => {
          const selectedIndices = props?.selectedShapeIndices || [];
          if (selectedIndices.length > 0) {
            const plot = document.getElementById(
              plotId,
            ) as Plotly.PlotlyHTMLElement;
            if (!plot) return;

            const shapes = plot.layout.shapes as Plotly.Shape[] | null;
            if (!shapes) return;

            const updatedShapes = shapes.filter(
              (_shape, index) => !selectedIndices.includes(index),
            );

            const newLayout = { ...plot.layout, shapes: updatedShapes };
            Plotly.react(plot, plot.data, newLayout, plot.config);
            updateShapeAnnotations(updatedShapes);
          }
        }}
        hidden={({ props }) =>
          !props?.selectedShapeIndices ||
          props.selectedShapeIndices.length === 0
        }
      >
        Delete Selected
      </Item>
    );

    const setTypeSubmenu = (
      <Submenu
        id="set-shape-type"
        key="set-shape-type"
        label="Set type"
        hidden={({ props }) => props?.shapeIndex === undefined}
      >
        {boundingBoxCategories?.map((category) => (
          <Item
            id={category.name}
            key={category.name}
            onClick={({ id, props }) => {
              handleShapeContextMenuClick(id, props);
            }}
          >
            {category.name}
          </Item>
        ))}
      </Submenu>
    );

    const setTypeSelectedSubmenu = (
      <Submenu
        id="set-selected-type"
        key="set-selected-type"
        label="Set type for selected"
        hidden={({ props }) =>
          !props?.selectedShapeIndices ||
          props.selectedShapeIndices.length === 0
        }
      >
        {boundingBoxCategories?.map((category) => (
          <Item
            id={category.name}
            key={category.name}
            onClick={({ id, props }) => {
              const selectedIndices = props?.selectedShapeIndices || [];
              if (selectedIndices.length === 0) return;

              const plot = document.getElementById(
                plotId,
              ) as Plotly.PlotlyHTMLElement;
              if (!plot) return;

              const shapes = plot.layout.shapes as Plotly.Shape[] | undefined;
              if (!shapes) return;

              const selectedCategory = boundingBoxCategories?.find(
                (cat) => cat.name === id,
              );
              if (!selectedCategory) return;

              const updatedShapes = [...shapes];
              selectedIndices.forEach((index) => {
                if (index >= 0 && index < updatedShapes.length) {
                  updatedShapes[index] = {
                    ...updatedShapes[index],
                    meta: {
                      ...updatedShapes[index].meta,
                      label: selectedCategory.name,
                    },
                    line: { color: "rgb(150, 150, 150)", width: 5 },
                    fillcolor: selectedCategory.color
                      ?.replace("rgb(", "rgba(")
                      .replace(")", ", 0.5)"),
                  };
                }
              });

              const newLayout = { ...plot.layout, shapes: updatedShapes };
              Plotly.react(plot, plot.data, newLayout, plot.config);
              updateShapeAnnotations(updatedShapes);
            }}
          >
            {category.name}
          </Item>
        ))}
      </Submenu>
    );

    registerMenuItem("shape-delete", deleteItem);
    registerMenuItem("shape-delete-selected", deleteSelectedItem);
    registerMenuItem("shape-type", setTypeSubmenu);
    registerMenuItem("shape-type-selected", setTypeSelectedSubmenu);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    handleShapeDelete,
    handleShapeContextMenuClick,
    boundingBoxCategories,
    plotId,
    updateShapeAnnotations,
  ]);

  return (
    <div className="px-6 py-3 space-y-3 flex-col">
      {/* Div where plot is inserted */}
      <div id={plotId} className="" />
      <>
        {React.Children.map(children, (child) => {
          return React.isValidElement(child)
            ? React.cloneElement(child, {
                plotId,
                plotReady,
                forceUpdate: updateTools,
                selectedXRange: selectedXRange,
              })
            : child;
        })}
      </>
    </div>
  );
};
