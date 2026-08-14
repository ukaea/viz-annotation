"use client";

import { PlotProps, Profile2DData, Profile2DViewParams } from "@/types";
import {
  applyGlobalStyle,
  arrayMax,
  arrayMin,
  sumOverFirstAxis,
} from "@/app/utils";
import { BaseTimeSeriesPlot } from "@/app/components/plots/base-plot";
import { TimeSeriesProvider } from "@/app/contexts/TimeSeriesContext";
import { TimeRegion } from "@/app/components/tools/timeRegion";
import { TimePoint } from "@/app/components/tools/timePoint";
import { BoundingBox } from "@/app/components/tools/boundingBox";
import { Polygon } from "@/app/components/tools/polygon";
import { AnnotationToolbar } from "@/app/components/tools/annotationToolbar";
import { AnnotationsTable } from "@/app/components/ui/annotationsTable";
import { useSample } from "@/app/contexts/SampleContext";
import { useElementHeight } from "@/app/hooks/useElementHeight";
import { useEffect, useMemo, useState } from "react";
import { View } from "@adobe/react-spectrum";
import * as d3 from "d3";

// The subplot annotations with real y values should be restricted to.
const HEATMAP_SUBPLOT = "xy2";

// Floor for the plot, also used before the container has been measured. Must
// match the min-h on the container below. The heatmap takes the upper 80% and
// the integrated trace the lower 20%, so this leaves them ~380px and ~95px -
// below that the plot keeps its size and the page scrolls instead.
const MIN_PLOT_HEIGHT = 480;

// Plotly supports a shared color axis but @types/plotly.js does not declare it.
type ColorAxis = {
  cmin: number;
  cmax: number;
  colorscale: [number, string][];
  colorbar: Partial<Plotly.ColorBar>;
};

type ProfileValues = (number | null)[][];

const colorMapInterpolators: Record<string, (value: number) => string> = {
  Viridis: d3.interpolateViridis,
  Plasma: d3.interpolatePlasma,
  Inferno: d3.interpolateInferno,
  Magma: d3.interpolateMagma,
  Cividis: d3.interpolateCividis,
};

// Leaves the lowest values fully transparent so the background shows through.
const buildColorScale = (
  interpFunc: (value: number) => string,
  smallPrecisionFactor: number,
): [number, string][] => [
  [0, "rgba(0, 0, 0, 0)"],
  [smallPrecisionFactor * 1.001, interpFunc(0)],
  ...([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1].map((stop) => [
    stop,
    interpFunc(stop),
  ]) as [number, string][]),
];

const createLinearScalePlot = (
  data: Profile2DData,
  plotProps: PlotProps,
  interpFunc: (value: number) => string,
): { values: ProfileValues; colorAxis: ColorAxis } => {
  const numDigits = plotProps.numSignificantDigits || 4;
  const smallPrecisionFactor = Math.pow(10, -1 * numDigits);
  const values = data.values as ProfileValues;

  return {
    values,
    colorAxis: {
      cmin: Math.max(smallPrecisionFactor, arrayMin(values.flat())),
      cmax: Math.max(smallPrecisionFactor, arrayMax(values.flat())),
      colorscale: buildColorScale(interpFunc, smallPrecisionFactor),
      colorbar: { ticks: "outside", tickfont: { size: 10 } },
    },
  };
};

// Ticks at each power of ten across the range, with unlabelled minor ticks between.
const generateLogTicks = (min: number, max: number) => {
  const formatTickLabel = (value: number) => value.toExponential(0);
  const minPower = Math.floor(Math.log10(min));
  const maxPower = Math.ceil(Math.log10(max));

  const tickvals: number[] = [];
  const ticktext: string[] = [];

  for (let power = minPower; power <= maxPower; power++) {
    const baseValue = Math.pow(10, power);

    if (baseValue >= min && baseValue <= max) {
      tickvals.push(Math.log10(baseValue));
      ticktext.push(formatTickLabel(baseValue));
    }

    for (let multiplier = 2; multiplier <= 9; multiplier++) {
      const subValue = baseValue * multiplier;
      if (subValue >= min && subValue <= max) {
        tickvals.push(Math.log10(subValue));
        ticktext.push("");
      }
    }
  }

  // Always label the ends of the range, even when they are not round numbers.
  if (!tickvals.some((val) => Math.abs(val - Math.log10(min)) < 1e-10)) {
    tickvals.unshift(Math.log10(min));
    ticktext.unshift(formatTickLabel(min));
  }
  if (!tickvals.some((val) => Math.abs(val - Math.log10(max)) < 1e-10)) {
    tickvals.push(Math.log10(max));
    ticktext.push(formatTickLabel(max));
  }

  return { tickvals, ticktext };
};

const createLogScalePlot = (
  data: Profile2DData,
  plotProps: PlotProps,
  interpFunc: (value: number) => string,
): { values: ProfileValues; colorAxis: ColorAxis } => {
  const numDigits = plotProps.numSignificantDigits || 4;
  const smallPrecisionFactor = Math.pow(10, -1 * numDigits);
  const rawValues = data.values as ProfileValues;

  const cmin = Math.max(smallPrecisionFactor, arrayMin(rawValues.flat()));
  const cmax = Math.max(smallPrecisionFactor, arrayMax(rawValues.flat()));

  const values: ProfileValues = rawValues.map((row) =>
    row.map((x) =>
      x !== null ? Math.log10(Math.max(x, smallPrecisionFactor)) : null,
    ),
  );

  const { tickvals, ticktext } = generateLogTicks(cmin, cmax);

  return {
    values,
    colorAxis: {
      cmin: arrayMin(values.flat()),
      cmax: arrayMax(values.flat()),
      colorscale: buildColorScale(interpFunc, smallPrecisionFactor),
      colorbar: {
        ticks: "outside",
        tickmode: "array",
        tickvals,
        ticktext,
        tickfont: { size: 10 },
      },
    },
  };
};

const buildPlotData = (
  data: Profile2DData,
  values: ProfileValues,
  thresholdActive?: boolean,
): Partial<Plotly.PlotData>[] => {
  const heatmap = {
    type: "heatmap",
    x: data.time,
    y: data.dim_1,
    z: values,
    // Hover reports the underlying value, which differs from z on a log scale.
    customdata: data.values,
    hovertemplate:
      "time: %{x:.4g}<br>dim 1: %{y:.4g}<br>value: %{customdata:.4e}<extra></extra>",
    coloraxis: "coloraxis",
    // Fade the heatmap while thresholding so the generated polygons stand out.
    opacity: thresholdActive ? 0.4 : 1,
    yaxis: "y2",
  } as Partial<Plotly.PlotData>;

  const integrated: Partial<Plotly.PlotData> = {
    name: "Integrated values",
    mode: "lines",
    x: data.time,
    y: sumOverFirstAxis(values),
  };

  return [integrated, heatmap];
};

const buildPlotLayout = (
  colorAxis: ColorAxis,
  isDarkMode: boolean,
  height: number,
): Partial<Plotly.Layout> =>
  applyGlobalStyle(
    {
      autosize: true,
      height,
      xaxis: {
        title: { text: "" },
        domain: [0, 1],
        linewidth: 1,
        zerolinewidth: 1,
        showgrid: false,
      },
      // The heatmap occupies the upper 80% of the plot...
      yaxis2: {
        title: { text: "" },
        domain: [0.2, 1],
        linewidth: 1,
        zerolinewidth: 1,
        showgrid: false,
        fixedrange: true,
        anchor: "x",
      },
      // ...with the integrated trace sharing its time axis below.
      yaxis: {
        title: { text: "Integrated<br>Values" },
        domain: [0, 0.2],
        linewidth: 1,
        zerolinewidth: 1,
        showgrid: false,
        anchor: "x",
      },
      showlegend: false,
      // Left drag pans, matching the time series view.
      dragmode: "pan",
      // Preserve zoom/pan across re-renders.
      uirevision: "true",
      ...{ coloraxis: colorAxis },
    } as Partial<Plotly.Layout>,
    isDarkMode,
  );

// Draw/erase shape buttons are omitted since the D3 tools handle drawing instead.
const PLOT_CONFIG: Partial<Plotly.Config> = {
  modeBarButtons: [
    ["zoom2d", "select2d", "pan2d", "autoScale2d", "resetScale2d", "toImage"],
  ],
  displaylogo: false,
  displayModeBar: true,
  scrollZoom: true,
  responsive: true,
};

export const Profile2dView = () => {
  const { data, plotProps, viewParams } = useSample();

  const viewData = data as Profile2DData | null;
  const profileViewParams = viewParams as Profile2DViewParams | null;
  const logScale = profileViewParams?.log_scale ?? false;

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

  // Memoised so an annotation drag doesn't tear down and rebuild the plot mid-draw.
  const plot = useMemo(() => {
    if (!viewData || !plotProps) return null;
    const interpFunc =
      colorMapInterpolators[plotProps.colorMap ?? ""] ?? d3.interpolateCividis;
    const createPlotFunc = logScale
      ? createLogScalePlot
      : createLinearScalePlot;
    return createPlotFunc(viewData, plotProps, interpFunc);
  }, [viewData, plotProps, logScale]);

  const plotData = useMemo<Partial<Plotly.PlotData>[]>(() => {
    if (!viewData || !plot) return [];
    return buildPlotData(viewData, plot.values, plotProps?.thresholdActive);
  }, [viewData, plot, plotProps?.thresholdActive]);

  // The plot fills whatever height is left once the annotations table has taken
  // its share, down to a floor that keeps it usable on a short window.
  const { ref: plotAreaRef, height: plotAreaHeight } =
    useElementHeight<HTMLDivElement>();
  const plotHeight = Math.max(plotAreaHeight, MIN_PLOT_HEIGHT);

  const plotLayout = useMemo(
    () => (plot ? buildPlotLayout(plot.colorAxis, isDarkMode, plotHeight) : {}),
    [plot, isDarkMode, plotHeight],
  );

  if (!viewData || !profileViewParams || !plot) {
    return null;
  }

  return (
    <View width="100%" height="100%" minHeight={0}>
      <TimeSeriesProvider signalName={profileViewParams.signal_name}>
        <div className="flex h-full min-h-0 flex-row justify-between">
          <div className="flex min-h-0 min-w-0 flex-1 flex-col gap-4">
            <div ref={plotAreaRef} className="min-h-[480px] flex-1">
              <BaseTimeSeriesPlot
                plotId="Profile2DView"
                ariaLabel="profile-2d"
                plotConfig={{
                  data: plotData,
                  config: PLOT_CONFIG,
                  layout: plotLayout,
                }}
                rescaleOnZoom={false}
                muteHoverWhileDrawing
              >
                <TimeRegion />
                <TimePoint />
                <BoundingBox subplot={HEATMAP_SUBPLOT} />
                <Polygon subplot={HEATMAP_SUBPLOT} />
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
