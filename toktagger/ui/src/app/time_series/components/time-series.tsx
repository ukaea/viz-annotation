"use client";
import { MultiVariateTimeSeriesData, TimeSeriesData } from "@/types";
import { BaseTimeSeriesPlot } from "@/app/components/plots/base-plot";
import { TimeSeriesProvider } from "@/app/contexts/TimeSeriesContext";
import { TimeRegion } from "@/app/components/tools/timeRegion";
import "react-contexify/ReactContexify.css";

import { applyGlobalStyle, arrayMax, arrayMin } from "@/app/utils";
import { useEffect, useMemo, useState } from "react";
import { TimePoint } from "@/app/components/tools/timePoint";
import { useSample } from "@/app/contexts/SampleContext";
import { View } from "@adobe/react-spectrum";
import { AnnotationsTable } from "@/app/components/ui/annotationsTable";
import { AnnotationToolbar } from "@/app/components/tools/annotationToolbar";
import { useElementHeight } from "@/app/hooks/useElementHeight";

// The traces are stacked, so each one needs its own share of the height. Below
// this the subplots are too squashed to read, so the plot keeps its size and the
// page scrolls instead. The rangeslider takes 10% of the plot, hence the extra.
const MIN_TRACE_HEIGHT = 105;
const RANGESLIDER_FRACTION = 0.1;

// Floor for a single trace, also used before the container has been measured.
const MIN_PLOT_HEIGHT = 320;

export const TimeSeriesView = () => {
  const { data } = useSample();

  const [plotData, setPlotData] = useState<Partial<Plotly.PlotData>[]>([]);

  const viewData = data as MultiVariateTimeSeriesData | null;

  // Tracked as state so the plot restyles live when the OS theme changes.
  const [isDarkMode, setIsDarkMode] = useState(
    () => window.matchMedia("(prefers-color-scheme: dark)").matches,
  );
  useEffect(() => {
    const query = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = (event: MediaQueryListEvent) =>
      setIsDarkMode(event.matches);
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, []);

  useEffect(() => {
    if (!viewData) return;

    const numRows = Object.keys(viewData.values).length;

    let plotData: Partial<Plotly.PlotData>[] = Object.entries(
      viewData.values,
    ).map(([key, value]: [string, TimeSeriesData]) => {
      return {
        name: key,
        x: value.time,
        y: value.values,
        mode: "lines",
      };
    });

    const yAxesNames = Array.from(
      { length: numRows },
      (_, i) => `y${i === 0 ? "" : i + 1}`,
    ).reverse();

    // Dynamically generate y-axis titles based on plotData names
    plotData = plotData.map((trace, index) => ({
      ...trace,
      yaxis: yAxesNames[index],
    }));
    setPlotData(plotData);
  }, [data, viewData]);

  // The plot fills whatever height is left once the annotations table has taken
  // its share, but never drops below what the stacked traces need to stay legible.
  const { ref: plotAreaRef, height: plotAreaHeight } =
    useElementHeight<HTMLDivElement>();
  const minPlotHeight = Math.max(
    MIN_PLOT_HEIGHT,
    Math.round(
      (plotData.length * MIN_TRACE_HEIGHT) / (1 - RANGESLIDER_FRACTION),
    ),
  );
  const plotHeight = Math.max(plotAreaHeight, minPlotHeight);

  const plotLayout: Partial<Plotly.Layout> = useMemo(() => {
    let maxTime = -Infinity;
    let minTime = Infinity;

    for (const trace of plotData) {
      const xData = trace.x as number[];
      if (xData && xData.length > 0) {
        const traceMax = arrayMax(xData);
        const traceMin = arrayMin(xData);
        if (traceMax > maxTime) maxTime = traceMax;
        if (traceMin < minTime) minTime = traceMin;
      }
    }

    const numRows = plotData.length;
    const domainHeight = 1 / numRows;
    // Dynamically generate y-axis domains based on numRows
    const yAxisDomains = Array.from({ length: numRows }, (_, i) => {
      const start = i * domainHeight;
      const end = (i + 1) * domainHeight;
      return [start, end];
    });

    // Build yaxis layout object dynamically
    const yAxesLayout = yAxisDomains.reduce(
      (acc, domain, idx) => {
        const axisNum = idx === 0 ? "" : idx + 1; // yaxis, yaxis2, yaxis3, ...
        acc[`yaxis${axisNum}`] = {
          domain,
          autorange: true,
          fixedrange: true,
          title: {
            text: plotData[numRows - idx - 1].name || "",
            font: {
              family: "Courier New, monospace",
              size: 12,
              color: "#7f7f7f",
            },
          },
        };
        return acc;
      },
      {} as Record<string, unknown>,
    );

    return applyGlobalStyle(
      {
        uirevision: "true",
        //grid: { rows: 1, columns: 1, pattern: "independent" },
        dragmode: "pan",
        autosize: true,
        height: plotHeight,
        xaxis: {
          minallowed: minTime,
          maxallowed: maxTime,
          range: [minTime, maxTime],
          fixedrange: false,
          autorange: false,
          rangeslider: { visible: true, thickness: 0.1 },
          title: {
            text: "Time [s]",
            font: {
              family: "Courier New, monospace",
              size: 12,
              color: "#7f7f7f",
            },
          },
        },
        ...yAxesLayout,
      },
      isDarkMode,
    );
  }, [plotData, isDarkMode, plotHeight]);

  if (!viewData) {
    return null;
  }

  return (
    <View width="100%" height="100%" minHeight={0}>
      <TimeSeriesProvider>
        <div className="flex h-full min-h-0 flex-row justify-between">
          <div className="flex min-h-0 min-w-0 flex-1 flex-col gap-4">
            <div
              ref={plotAreaRef}
              className="flex-1"
              style={{ minHeight: minPlotHeight }}
            >
              <BaseTimeSeriesPlot
                plotId="TimesSeriesView"
                plotConfig={{ data: plotData, layout: plotLayout }}
              >
                <TimeRegion />
                <TimePoint />
              </BaseTimeSeriesPlot>
            </div>
            <AnnotationsTable />
          </div>
          <AnnotationToolbar />
        </div>
      </TimeSeriesProvider>
    </View>
  );
};
