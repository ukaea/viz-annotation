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
import { useEffect, useMemo, useState } from "react";
import { Flex, View } from "@adobe/react-spectrum";
import * as d3 from "d3";

// The heatmap is drawn against yaxis2, so annotations carrying real y values belong to
// this subplot only - the integrated values subplot below it has an unrelated y scale.
const HEATMAP_SUBPLOT = "xy2";

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

// A colour scale that leaves the lowest values fully transparent, so the plot
// background (and anything drawn beneath the heatmap) shows through.
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
): Partial<Plotly.Layout> =>
  applyGlobalStyle(
    {
      autosize: true,
      height: window.innerHeight * 0.9,
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
      // Left drag pans and the wheel zooms, matching the time series view. yaxis2 is
      // fixedrange, so the wheel zooms time only.
      dragmode: "pan",
      // Preserve zoom/pan across re-renders.
      uirevision: "true",
      ...{ coloraxis: colorAxis },
    } as Partial<Plotly.Layout>,
    isDarkMode,
  );

// Bounding boxes and polygons are drawn by the D3 annotation tools rather than
// Plotly's built in shape editing, so the draw/erase shape buttons are omitted.
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

  // applyGlobalStyle reads the OS/browser dark mode preference, but that read only
  // happens when plotLayout is recomputed. Without tracking the preference as state,
  // switching themes while the page is open would not restyle the plot until
  // something else (e.g. a reload) forced plotLayout to rebuild.
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

  // BaseTimeSeriesPlot re-initialises Plotly whenever the data or layout it is given
  // change identity, which also rebuilds the D3 overlay the annotation tools draw on.
  // Drawing an annotation updates the sample state and re-renders this component, so
  // without memoising these the plot would be torn down mid-drag and the annotation
  // lost. The time series view memoises for the same reason.
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

  const plotLayout = useMemo(
    () => (plot ? buildPlotLayout(plot.colorAxis, isDarkMode) : {}),
    [plot, isDarkMode],
  );

  if (!viewData || !profileViewParams || !plot) {
    return null;
  }

  return (
    <View width="100%">
      <Flex justifyContent="center" alignItems="center">
        <TimeSeriesProvider signalName={profileViewParams.signal_name}>
          <Flex direction="row" flex justifyContent="space-between">
            <Flex direction="column" flex gap="size-200">
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
              <AnnotationsTable />
            </Flex>
            <AnnotationToolbar />
          </Flex>
        </TimeSeriesProvider>
      </Flex>
    </View>
  );
};
