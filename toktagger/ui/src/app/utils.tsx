import {
  Annotation,
  TimeRegionSchema,
  TimePointSchema,
  TimePoint,
  TimeRegion,
  TimeSeriesAnnotation,
  TimeSeriesAnnotationType,
  BoundingBox,
  BoundingBoxSchema,
  Polygon,
  PolygonSchema,
  TimeSeriesAnnotationPoint,
  Sample,
  TimeSeriesFileDataSchema,
  ShotDataSchema,
} from "@/types";
import { v4 as uuidv4 } from "uuid";
import z from "zod/v4";

const colorPalette = [
  "#FF5733",
  "#33FF57",
  "#3357FF",
  "#FF33A8",
  "#A833FF",
  "#33FFF6",
  "#FFC733",
  "#8DFF33",
  "#FF3380",
  "#33A8FF",
  "#FF8D33",
  "#3380FF",
  "#33FFAA",
  "#FFAA33",
  "#AA33FF",
  "#FF3333",
];

export function randomColor(index: number): string {
  const color = colorPalette[index % colorPalette.length];
  return color;
}

export const linspace = (start: number, end: number, num: number) => {
  const step = (end - start) / (num - 1);
  const arr = [];
  for (let i = 0; i < num; i++) {
    arr.push(start + step * i);
  }
  return arr;
};

// Utility function to find the maximum value in an array
// Handles very large arrays efficiently
// Nulls are skipped: the API serialises NaN samples in a profile as null.
export function arrayMax(arr: (number | null)[]): number {
  let traceMax = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    const value = arr[i];
    if (value !== null && value > traceMax) traceMax = value;
  }
  return traceMax;
}

// Utility function to find the minimum value in an array
// Handles very large arrays efficiently
// Nulls are skipped: the API serialises NaN samples in a profile as null.
export function arrayMin(arr: (number | null)[]): number {
  let traceMin = Infinity;
  for (let i = 0; i < arr.length; i++) {
    const value = arr[i];
    if (value !== null && value < traceMin) traceMin = value;
  }
  return traceMin;
}

export function convertRawAnnotationsToTimeSeries(
  annotation: Annotation,
): TimeSeriesAnnotation | null {
  if (TimeRegionSchema.safeParse(annotation).success) {
    const timeRegion = TimeRegionSchema.parse(annotation);
    return {
      id: uuidv4(),
      created_by: timeRegion.created_by,
      label: timeRegion.label,
      signal_name: timeRegion.signal_name,
      type: TimeSeriesAnnotationType.TIME_REGION,
      points: [
        { x: timeRegion.time_min, y: 0 },
        { x: timeRegion.time_max, y: 0 },
      ],
      selected: false,
    };
  }

  if (TimePointSchema.safeParse(annotation).success) {
    const timePoint = TimePointSchema.parse(annotation);
    return {
      id: uuidv4(),
      created_by: timePoint.created_by,
      label: timePoint.label,
      signal_name: timePoint.signal_name,
      type: TimeSeriesAnnotationType.TIME_POINT,
      points: [{ x: timePoint.time, y: 0 }],
      selected: false,
    };
  }

  if (BoundingBoxSchema.safeParse(annotation).success) {
    const boundingBox = BoundingBoxSchema.parse(annotation);
    return {
      id: uuidv4(),
      created_by: boundingBox.created_by,
      label: boundingBox.label,
      signal_name: boundingBox.signal_name,
      type: TimeSeriesAnnotationType.BOUNDING_BOX,
      points: [
        { x: boundingBox.x_min, y: boundingBox.y_min + boundingBox.height },
        {
          x: boundingBox.x_min + boundingBox.width,
          y: boundingBox.y_min,
        },
      ],
      selected: false,
    };
  }

  if (PolygonSchema.safeParse(annotation).success) {
    const polygon = PolygonSchema.parse(annotation);
    return {
      id: uuidv4(),
      created_by: polygon.created_by,
      label: polygon.label,
      signal_name: polygon.signal_name,
      type: TimeSeriesAnnotationType.POLYGON,
      points: polygon.segmentation[0].reduce<TimeSeriesAnnotationPoint[]>(
        (accumulator, _, i, arr) => {
          if (i % 2 === 0) {
            accumulator.push({
              x: arr[i],
              y: arr[i + 1],
            });
          }
          return accumulator;
        },
        [],
      ),
      selected: false,
    };
  }

  console.warn(
    `The following annotation could not be parsed into a time series annotation:\n ${annotation}`,
  );
  return null;
}

export function convertTimeSeriesToRawAnnotations(
  annotation: TimeSeriesAnnotation,
): Annotation | null {
  if (annotation.type === TimeSeriesAnnotationType.TIME_POINT) {
    const timePoint: TimePoint = {
      project_id: null,
      sample_id: null,
      validated: false,
      uncertainty: 1,
      created_by: annotation.created_by,
      signal_name: annotation.signal_name ?? null,
      type: "time_point",
      time: annotation.points[0].x,
      label: annotation.label,
    };
    return timePoint;
  }

  if (annotation.type === TimeSeriesAnnotationType.TIME_REGION) {
    const timePoint: TimeRegion = {
      project_id: null,
      sample_id: null,
      validated: false,
      uncertainty: 1,
      created_by: annotation.created_by,
      signal_name: annotation.signal_name ?? null,
      type: "time_region",
      time_min: annotation.points[0].x,
      time_max: annotation.points[1].x,
      label: annotation.label,
    };
    return timePoint;
  }

  if (annotation.type === TimeSeriesAnnotationType.BOUNDING_BOX) {
    const boundingBox: BoundingBox = {
      project_id: null,
      sample_id: null,
      validated: false,
      uncertainty: 1,
      created_by: annotation.created_by,
      signal_name: annotation.signal_name ?? null,
      type: "bounding_box",
      x_min: Math.min(annotation.points[0].x, annotation.points[1].x),
      y_min: Math.min(annotation.points[0].y, annotation.points[1].y),
      height: Math.abs(annotation.points[0].y - annotation.points[1].y),
      width: Math.abs(annotation.points[0].x - annotation.points[1].x),
      label: annotation.label,
    };
    return boundingBox;
  }

  if (annotation.type === TimeSeriesAnnotationType.POLYGON) {
    const polygon: Polygon = {
      project_id: null,
      sample_id: null,
      validated: false,
      uncertainty: 1,
      created_by: annotation.created_by,
      signal_name: annotation.signal_name ?? null,
      type: "polygon",
      segmentation: [
        annotation.points.flatMap(({ x, y }) => {
          return [x, y];
        }),
      ],
      label: annotation.label,
    };
    return polygon;
  }

  console.warn(
    `The following annotation could not be parsed into a raw annotation:\n ${annotation}`,
  );
  return null;
}

// Collapse a 2D profile down its first axis, giving one value per column.
// Nulls are skipped: the API serialises NaN samples in a profile as null.
export function sumOverFirstAxis(arr: (number | null)[][]): number[] {
  if (arr.length === 0) return [];

  const numCols = arr[0].length;
  const sums = new Array(numCols).fill(0);

  for (const row of arr) {
    for (let j = 0; j < numCols; j++) {
      const value = row[j];
      if (value !== null) {
        sums[j] += value;
      }
    }
  }

  return sums;
}

// Plotly supports layout.coloraxis, but @types/plotly.js does not declare it.
type LayoutWithColorAxis = Partial<Plotly.Layout> & {
  coloraxis?: { colorbar?: Partial<Plotly.ColorBar> };
};

// Matches every axis key Plotly supports on a layout: xaxis, xaxis2, ..., yaxis,
// yaxis2, ... . Views with more than two axes (e.g. one y-axis per time series
// signal) need every one of them styled, not just the first couple.
const AXIS_KEY_PATTERN = /^[xy]axis\d*$/;

// Apply the shared dark mode styling to a Plotly layout. Takes the current
// preference as a parameter rather than reading it itself, so callers can react
// to the preference changing rather than it only being read once on mount.
export const applyGlobalStyle = (
  layout: Partial<Plotly.Layout>,
  isDarkMode: boolean,
) => {
  if (!isDarkMode) return layout;

  const foreground = "rgb(255, 255, 255)";
  const axes = layout as unknown as Record<
    string,
    Partial<Plotly.LayoutAxis> | undefined
  >;

  for (const key of Object.keys(layout)) {
    if (!AXIS_KEY_PATTERN.test(key)) continue;
    const axis = axes[key];
    if (!axis) continue;

    // Merge rather than replace, so an axis-specific font family/size survives.
    if (axis.title) {
      axis.title.font = { ...axis.title.font, color: foreground };
    }
    axis.linecolor = foreground;
    axis.zerolinecolor = foreground;
    axis.tickcolor = foreground;
    axis.tickfont = { ...axis.tickfont, color: foreground };
    // Plotly's default gridcolor is a light grey meant to be subtle against a
    // white background. Left alone it reads as a bold near-white line against
    // the dark, transparent one set below.
    axis.gridcolor = "rgba(255, 255, 255, 0.15)";
  }

  const colorbar = (layout as LayoutWithColorAxis).coloraxis?.colorbar;
  if (colorbar) {
    colorbar.tickcolor = foreground;
    colorbar.tickfont = { color: foreground };
    colorbar.outlinecolor = foreground;
  }

  // Let the surrounding page background show through the plot.
  layout.paper_bgcolor = "rgba(0, 0, 0, 0)";
  layout.plot_bgcolor = "rgba(0, 0, 0, 0)";

  return layout;
};

export function getSignalNames(sample: Sample | null): string[] {
  const sampleDataType = z.union([TimeSeriesFileDataSchema, ShotDataSchema]);

  if (!sample || !sampleDataType.safeParse(sample.data).success) {
    return [];
  }

  return sampleDataType.parse(sample.data).signal_names;
}

export function shallowEqual(
  a: Record<string, unknown>,
  b: Record<string, unknown>,
) {
  if (a === b) return true;

  const keysA = Object.keys(a);
  const keysB = Object.keys(b);

  if (keysA.length !== keysB.length) return false;

  for (const key of keysA) {
    if (a[key] !== b[key]) return false;
  }

  return true;
}
