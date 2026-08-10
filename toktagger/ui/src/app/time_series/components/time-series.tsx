"use client";
import { MultiVariateTimeSeriesData, TimeSeriesData } from "@/types";
import { BaseTimeSeriesPlot } from "@/app/components/plots/base-plot";
import { TimeSeriesProvider } from "@/app/contexts/TimeSeriesContext";
import { TimeRegion } from "@/app/components/tools/timeRegion";
import "react-contexify/ReactContexify.css";

import { arrayMax, arrayMin } from "@/app/utils";
import { useEffect, useMemo, useState } from "react";
import { TimePoint } from "@/app/components/tools/timePoint";
import { useSample } from "@/app/contexts/SampleContext";
import { Flex, View } from "@adobe/react-spectrum";
import { AnnotationsTable } from "@/app/components/ui/annotationsTable";
import { AnnotationToolbar } from "@/app/components/tools/annotationToolbar";

export const TimeSeriesView = () => {
  const { data } = useSample();

  const [plotData, setPlotData] = useState<Partial<Plotly.PlotData>[]>([]);

  const viewData = data as MultiVariateTimeSeriesData | null;

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

    return {
      uirevision: "true",
      //grid: { rows: 1, columns: 1, pattern: "independent" },
      dragmode: "pan",
      autosize: true,
      height: window.innerHeight * 0.9,
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
    };
  }, [plotData]);

  if (!viewData) {
    return null;
  }

  return (
    <View width="100%">
      <Flex justifyContent="center" alignItems="center">
        <TimeSeriesProvider>
          <Flex direction="row" flex justifyContent="space-between">
            <Flex direction="column" flex gap="size-200">
              <BaseTimeSeriesPlot
                plotId="TimesSeriesView"
                plotConfig={{ data: plotData, layout: plotLayout }}
              >
                <TimeRegion />
                <TimePoint />
              </BaseTimeSeriesPlot>
              <AnnotationsTable />
            </Flex>
            <AnnotationToolbar />
          </Flex>
        </TimeSeriesProvider>
      </Flex>
    </View>
  );
};
