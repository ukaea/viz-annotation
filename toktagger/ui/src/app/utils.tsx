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
} from "@/types";
import { v4 as uuidv4 } from "uuid";

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
export function arrayMax(arr: number[]): number {
  let traceMax = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] > traceMax) traceMax = arr[i];
  }
  return traceMax;
}

// Utility function to find the minimum value in an array
// Handles very large arrays efficiently
export function arrayMin(arr: number[]): number {
  let traceMin = Infinity;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] < traceMin) traceMin = arr[i];
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
      type: TimeSeriesAnnotationType.BOUNDING_BOX,
      points: [
        { x: boundingBox.x_min, y: boundingBox.y_min },
        {
          x: boundingBox.x_min + boundingBox.width,
          y: boundingBox.y_min + boundingBox.height,
        },
      ],
      selected: false,
    };
  } else {
    console.log(BoundingBoxSchema.safeParse(annotation).error?.message);
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
      type: "bounding_box",
      x_min: Math.min(annotation.points[0].x, annotation.points[1].x),
      y_min: Math.min(annotation.points[0].y, annotation.points[1].y),
      height: Math.abs(annotation.points[0].y - annotation.points[1].y),
      width: Math.abs(annotation.points[0].x - annotation.points[1].x),
      label: annotation.label,
    };
    return boundingBox;
  }

  console.warn(
    `The following annotation could not be parsed into a raw annotation:\n ${annotation}`,
  );
  return null;
}
