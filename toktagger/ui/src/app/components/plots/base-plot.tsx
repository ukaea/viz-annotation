"use client";
import {
  PlotlyHTMLElement,
  PlotData,
  Layout,
  Config,
  react,
  relayout,
  PlotRelayoutEvent,
  PlotSelectionEvent,
} from "plotly.js";
import { useEffect, useRef, useState } from "react";
import {
  useTimeSeriesActions,
  useTimeSeriesState,
} from "@/app/contexts/TimeSeriesContext";
import { ExtendedPlotlyHTMLElement, TimeSeriesAnnotationPoint } from "@/types";
import React from "react";
import { arrayMax, arrayMin } from "@/app/utils";
import { ToastQueue } from "@adobe/react-spectrum";

const DEFAULT_PLOTLY_CONFIG: Partial<Config> = {
  modeBarButtons: [
    ["toImage", "zoom2d", "select2d", "pan2d", "autoScale2d", "resetScale2d"],
  ],
  displaylogo: false,
  displayModeBar: true,
  scrollZoom: true,
  responsive: true,
};

// The typing for plotly's selection relayout is not great - this avoids errors and ensures the correct object is used
const EMPTY_PLOTLY_SELECTION = { selections: [] } as Partial<Layout>;

interface PlotConfiguration {
  data: Partial<PlotData>[];
  layout: Partial<Layout>;
  config?: Partial<Config>;
}

type InjectedProps = {
  plotId: string;
  plotReady: boolean;
};

type TimeSeriesPlotProps = {
  plotId?: string;
  plotConfig: PlotConfiguration;
  rescaleOnZoom?: boolean;
  children:
    | React.ReactElement<InjectedProps>
    | React.ReactElement<InjectedProps>[];
};

export const BaseTimeSeriesPlot = ({
  plotId: externalId,
  plotConfig: { data, layout, config = DEFAULT_PLOTLY_CONFIG },
  rescaleOnZoom = true,
  children,
}: TimeSeriesPlotProps) => {
  const [plotReady, setPlotReady] = useState(false);

  const {
    createAnnotation,
    addAnnotation,
    triggerUpdate,
    findSelectedAnnotations,
    setOngoingAction,
  } = useTimeSeriesActions();
  const { activeAnnotationTool, toolingCallbacks, isDrawing, editMode } =
    useTimeSeriesState();

  const isDraggingRef = useRef(false);
  const allowRelayout = useRef(true);

  const plotId = externalId || "time-series";

  if (!isDrawing) isDraggingRef.current = false;

  useEffect(() => {
    const plot = document.getElementById(plotId) as PlotlyHTMLElement;
    if (!plot) {
      console.warn(
        "Base plot element could not be located, skipping plot render",
      );
      return;
    }

    const overplots: string[] = []; // store IDs of overplots to allow D3 to draw on subplots

    const generateOverplots = (plot: PlotlyHTMLElement) => {
      // Get all subplot elements and extract the subplot name (xy for example) from the class list
      const subplots = plot.querySelectorAll(".subplot");
      const subplotNames = [...subplots].map((el) =>
        [...el.classList].find((cls) => cls !== "subplot"),
      );

      // For each subplot identified generate a D3 overplot with the subplot name appended so that tooling can reference it
      subplotNames.forEach((coordinateSystem) => {
        // Find subplot if it exists
        const subplot = plot
          .querySelector(`.subplot.${coordinateSystem}`)
          ?.querySelector(".overplot")
          ?.querySelector(`.${coordinateSystem}`) as HTMLElement;
        if (!subplot) {
          console.error("Cannot locate plotly subplot");
          return;
        }

        // ensure only one custom overlay group is present
        if (!subplot.querySelector(`.${plotId}-overplot-${coordinateSystem}`)) {
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
      if (rescaleOnZoom) {
        // This makes use of the first graph displayed but this should be fine
        // Note that the event fired by plotly is a bit strange hence the different handlers
        if ("xaxis.range[0]" in eventData && "xaxis.range[1]" in eventData) {
          // This logic is triggered after a normal zoom/pan event
          rescale(eventData["xaxis.range[0]"], eventData["xaxis.range[1]"]);
        } else if (
          eventData["xaxis.range"] &&
          eventData["xaxis.range"].length === 2
        ) {
          // This logic is triggered after a range bar event
          const x0 = eventData["xaxis.range"][0] as number;
          const x1 = eventData["xaxis.range"][1] as number;
          rescale(x0, x1);
        } else if (
          Object.keys(eventData).some((key) => key.startsWith("xaxis"))
        ) {
          rescale(); // Handle other updates like auto-scale button (e.g. xaxis.autorange: true)
        }
      }
      triggerUpdate();
    };

    const initGraph = async () => {
      react(plot, data, layout, config).then(generateOverplots);

      plot.removeAllListeners("plotly_relayout"); // remove any existing listeners
      plot.removeAllListeners("plotly_selected");
      plot.on("plotly_relayout", relayoutHandler);
    };
    initGraph();

    return () => {
      // cleanup on unmount / Fast-Refresh
      overplots.forEach((overplot) => {
        plot?.querySelector(`.${overplot}`)?.remove(); // remove custom overlay group
      });
      setPlotReady(false); // reset ready state
    };
  }, [config, data, layout, plotId, rescaleOnZoom, triggerUpdate]);

  useEffect(() => {
    if (!plotReady) {
      // Plot may not have loaded yet - this will rerun after loading
      return;
    }

    const plot = document.getElementById(plotId) as PlotlyHTMLElement;

    if (!plot) {
      console.error("Could not locate plot to set selection listener");
      return;
    }

    const onSelection = (eventData: PlotSelectionEvent) => {
      if (eventData?.range) {
        if (!editMode) {
          ToastQueue.info(
            "Change to Edit Mode to select annotations - see help popup in annotation toolbar for more info",
            { timeout: 5000 },
          );
        }
        findSelectedAnnotations({
          low: eventData.range.x[0],
          high: eventData.range.x[1],
        });
      }
      relayout(plot, EMPTY_PLOTLY_SELECTION); // Immediately remove selection indicator
    };

    plot.on("plotly_selected", onSelection);

    return () => {
      plot.removeAllListeners("plotly_selected");
    };
  }, [editMode, findSelectedAnnotations, plotId, plotReady]);

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

    relayout(plot, { dragmode: "pan" });
  }, [isDrawing, plotId, plotReady]);

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
      event: MouseEvent,
      _plot: PlotlyHTMLElement,
    ): TimeSeriesAnnotationPoint {
      const plot = _plot as ExtendedPlotlyHTMLElement;
      let xaxis = plot._fullLayout.xaxis; // x-axis descriptor
      let yaxis = plot._fullLayout.yaxis; // y-axis descriptor

      const bb = (event.target as HTMLElement).getBoundingClientRect();
      const relX = event.clientX - bb.left; // click X in pixels, relative to plot
      const relY = event.clientY - bb.top; // click Y in pixels, relative to plot

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

      return { x, y };
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

    const handleCancelSelection = (event: MouseEvent) => {
      if (!event.ctrlKey) {
        findSelectedAnnotations(null);
      }
    };

    const startAnnotationCreation = (event: MouseEvent) => {
      if (event.ctrlKey) {
        console.log(editMode);
        if (!editMode) {
          ToastQueue.info(
            "Change to Edit Mode to draw annotations - see help popup in annotation toolbar for more info",
            { timeout: 5000 },
          );
          return;
        }
        if (activeAnnotationTool) {
          setOngoingAction(true);
          isDraggingRef.current = true;
          const clickLocation = getClickData(event, plot);
          toolingCallbacks
            .get(activeAnnotationTool.type)
            ?.start(
              clickLocation.x,
              clickLocation.y,
              activeAnnotationTool.label,
            );
        } else {
          ToastQueue.info(
            "Select a tool to draw annotation - see help popup in annotation toolbar for more info",
            { timeout: 5000 },
          );
        }
      }
    };

    const updateAnnotation = (event: MouseEvent) => {
      if (activeAnnotationTool && isDraggingRef.current) {
        const clickLocation = getClickData(event, plot);
        toolingCallbacks
          .get(activeAnnotationTool.type)
          ?.move(clickLocation.x, clickLocation.y);
      }
    };

    const finishAnnotationCreation = (event: MouseEvent) => {
      setOngoingAction(false);
      isDraggingRef.current = false;
      if (activeAnnotationTool) {
        const clickLocation = getClickData(event, plot);
        toolingCallbacks
          .get(activeAnnotationTool.type)
          ?.end(clickLocation.x, clickLocation.y);
      }
    };

    draggableElements.forEach((element) => {
      element.addEventListener("contextmenu", handleContextMenu);
      element.addEventListener("mousedown", handleCancelSelection);
      element.addEventListener("mousedown", startAnnotationCreation);

      if (editMode) {
        element.addEventListener("mousemove", updateAnnotation);
        element.addEventListener("mouseup", finishAnnotationCreation);
      }
    });

    return () => {
      draggableElements.forEach((element) => {
        element.removeEventListener("contextmenu", handleContextMenu);
        element.removeEventListener("mousedown", handleCancelSelection);
        element.removeEventListener("mousedown", startAnnotationCreation);
        element.removeEventListener("mousemove", updateAnnotation);
        element.removeEventListener("mouseup", finishAnnotationCreation);
      });
    };
  }, [
    activeAnnotationTool,
    addAnnotation,
    createAnnotation,
    editMode,
    findSelectedAnnotations,
    plotId,
    plotReady,
    setOngoingAction,
    toolingCallbacks,
  ]);

  return (
    <div className="w-full px-6 py-3 space-y-3 flex-col">
      {/* Div where plot is inserted */}
      <div id={plotId} className="" aria-label="time-series">
        <>
          {React.Children.map(children, (child) => {
            return React.isValidElement(child)
              ? React.cloneElement(child, {
                  plotId,
                  plotReady,
                })
              : child;
          })}
        </>
      </div>
    </div>
  );
};
