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
import { SelectionRange } from "@/types";
import React from "react";
import { arrayMax, arrayMin } from "@/app/utils";
import { ToastQueue } from "@adobe/react-spectrum";
import { useAnnotationTooling } from "./useAnnotationTooling";

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
  ariaLabel?: string;
  children:
    | React.ReactElement<InjectedProps>
    | React.ReactElement<InjectedProps>[];
};

export const BaseTimeSeriesPlot = ({
  plotId: externalId,
  plotConfig: { data, layout, config = DEFAULT_PLOTLY_CONFIG },
  rescaleOnZoom = true,
  ariaLabel = "time-series",
  children,
}: TimeSeriesPlotProps) => {
  const [plotReady, setPlotReady] = useState(false);

  const { triggerUpdate, findSelectedAnnotations } = useTimeSeriesActions();
  const { editMode } = useTimeSeriesState();

  const allowRelayout = useRef(true);

  const plotId = externalId || "time-series";

  useAnnotationTooling({ plotId, plotReady });

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
        // Plotly keys the range by axis id, so a selection on a subplot other than the
        // first reports e.g. y2 rather than y. Resolve the keys rather than assuming x/y.
        const range = eventData.range as Record<string, number[]>;
        const xRange =
          range[Object.keys(range).find((k) => k[0] === "x") ?? ""];
        const yRange =
          range[Object.keys(range).find((k) => k[0] === "y") ?? ""];

        if (xRange && yRange) {
          const selection: SelectionRange = {
            x: {
              low: Math.min(xRange[0], xRange[1]),
              high: Math.max(xRange[0], xRange[1]),
            },
            y: {
              low: Math.min(yRange[0], yRange[1]),
              high: Math.max(yRange[0], yRange[1]),
            },
          };
          findSelectedAnnotations(selection);
        }
      }
      relayout(plot, EMPTY_PLOTLY_SELECTION); // Immediately remove selection indicator
    };

    plot.on("plotly_selected", onSelection);

    return () => {
      plot.removeAllListeners("plotly_selected");
    };
  }, [editMode, findSelectedAnnotations, plotId, plotReady]);

  return (
    <div className="w-full px-6 py-3 space-y-3 flex-col">
      {/* Div where plot is inserted */}
      <div id={plotId} className="" aria-label={ariaLabel}>
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
