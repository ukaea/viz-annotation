import { PlotlyHTMLElement } from "plotly.js";
import { z } from "zod/v4";

export const BaseAnnotationSchema = z.object({
  project_id: z.string().nullable().default(null),
  sample_id: z.string().nullable().default(null),
  shot_id: z.number().optional(),
  timestamp: z.string().optional(),
  validated: z.boolean().nullable().default(null),
  uncertainty: z.number().nullable().default(1),
  created_by: z.string().default("manual"),
  label: z.string(),
  type: z.string(),
});

export type BaseAnnotation = z.infer<typeof BaseAnnotationSchema>;

export const TimeRegionSchema = BaseAnnotationSchema.extend({
  type: z.literal("time_region"),
  time_min: z.number(),
  time_max: z.number(),
});
export type TimeRegion = z.infer<typeof TimeRegionSchema>;

export const TimePointSchema = BaseAnnotationSchema.extend({
  type: z.literal("time_point"),
  time: z.number(),
});
export type TimePoint = z.infer<typeof TimePointSchema>;

export const ClassLabelSchema = BaseAnnotationSchema.extend({
  type: z.literal("class_label"),
});
export type ClassLabel = z.infer<typeof ClassLabelSchema>;

export const BoundingBoxSchema = BaseAnnotationSchema.extend({
  type: z.literal("bounding_box"),
  height: z.number(),
  width: z.number(),
  x_min: z.number(),
  y_min: z.number(),
});

export type BoundingBox = z.infer<typeof BoundingBoxSchema>;

export const VideoBoundingBoxSchema = BaseAnnotationSchema.extend({
  type: z.literal("video_bounding_box"),
  frame: z.number().int(),
  track_id: z.string(), // force string
  height: z.number().int(),
  width: z.number().int(),
  x_min: z.number().int(),
  y_min: z.number().int(),
});

export type VideoBoundingBox = z.infer<typeof VideoBoundingBoxSchema>;

const PolygonCoordinatesSchema = z
  .array(z.number())
  .min(6)
  .refine((coordinates) => coordinates.length % 2 === 0, {
    message: "A polygon must contain an even number of coordinates",
  });

const VideoPolygonCoordinatesSchema = z
  .array(z.number().int())
  .min(6)
  .refine((coordinates) => coordinates.length % 2 === 0, {
    message: "A polygon must contain an even number of coordinates",
  });

export const PolygonSchema = BaseAnnotationSchema.extend({
  type: z.literal("polygon"),
  segmentation: z.array(PolygonCoordinatesSchema).length(1),
});

export type Polygon = z.infer<typeof PolygonSchema>;

export const VideoPolygonSchema = BaseAnnotationSchema.extend({
  type: z.literal("video_polygon"),
  frame: z.number().int(),
  track_id: z.string(),
  segmentation: z.array(VideoPolygonCoordinatesSchema).length(1),
});

export type VideoPolygon = z.infer<typeof VideoPolygonSchema>;

export const VideoPointSchema = BaseAnnotationSchema.extend({
  type: z.literal("video_point"),
  frame: z.number().int(),
  track_id: z.string(),
  x: z.number().int(),
  y: z.number().int(),
});

export type VideoPoint = z.infer<typeof VideoPointSchema>;

export const AnnotationSchema = z.union([
  TimePointSchema,
  TimeRegionSchema,
  ClassLabelSchema,
  BoundingBoxSchema,
  PolygonSchema,
  VideoBoundingBoxSchema,
  VideoPolygonSchema,
  VideoPointSchema,
]);
export type Annotation = z.infer<typeof AnnotationSchema>;

export type NavAdapter = {
  getAnnotations: () => Annotation[];
  clear: () => void;
  afterSave?: () => void;
};

export const AnnotationsSchema = z.array(AnnotationSchema);
export type Annotations = z.infer<typeof AnnotationsSchema>;

export const TimeSeriesDataSchema = z.object({
  time: z.array(z.number()),
  values: z.array(z.number()),
});
export type TimeSeriesData = z.infer<typeof TimeSeriesDataSchema>;

export const MultiVariateTimeSeriesDataSchema = z.object({
  values: z.record(z.string(), TimeSeriesDataSchema),
});
export type MultiVariateTimeSeriesData = z.infer<
  typeof MultiVariateTimeSeriesDataSchema
>;

export const SpectrogramDataSchema = z.object({
  time: z.array(z.number()),
  frequency: z.array(z.number()),
  amplitude: z.array(z.array(z.number())),
  threshold_mask: z.array(z.array(z.number())).optional(),
});
export type SpectrogramData = z.infer<typeof SpectrogramDataSchema>;
export const ImageDataSchema = z.object({
  frame: z.number(),
  values: z.string(), // base64 PNG
});
export type ImageData = z.infer<typeof ImageDataSchema>;

export const DataSchema = z.union([
  TimeSeriesDataSchema,
  MultiVariateTimeSeriesDataSchema,
  SpectrogramDataSchema,
  ImageDataSchema,
]);
export type Data = z.infer<typeof DataSchema>;

export const CompositeDataSchema = z.object({
  values: z.record(z.string(), DataSchema),
});
export type CompositeData = z.infer<typeof CompositeDataSchema>;

export const CategorySchema = z.object({
  name: z.string(),
  color: z.string(),
});
export type Category = z.infer<typeof CategorySchema>;

export const BaseDisplayAnnotationSchema = z.object({
  created_by: z.string().default("manual"),
  selected: z.boolean().default(false),
  category: CategorySchema,
});

export type BaseDisplayAnnotation = z.infer<typeof BaseDisplayAnnotationSchema>;

export const ZoneSchema = BaseDisplayAnnotationSchema.extend({
  x0: z.number(),
  x1: z.number(),
});
export type Zone = z.infer<typeof ZoneSchema>;

export const VSpanSchema = BaseDisplayAnnotationSchema.extend({
  x: z.number(),
});
export type VSpan = z.infer<typeof VSpanSchema>;

export const SpectrogramMaskSchema = z.object({
  values: z.array(z.array(z.number())),
});
export type SpectrogramMask = z.infer<typeof SpectrogramMaskSchema>;

export const DisplayAnnotationSchema = z.union([
  ZoneSchema,
  VSpanSchema,
  SpectrogramMaskSchema,
]);
export type DisplayAnnotation = z.infer<typeof DisplayAnnotationSchema>;

export enum TaskType {
  TimeSeries = "time-series",
  // Spectrogram = "spectrogram",
  Video = "video",
}

export const TaskSchema = z.enum([
  TaskType.TimeSeries,
  // TaskType.Spectrogram,
  TaskType.Video,
]);

export const ProjectSchema = z.object({
  _id: z.string().nullable(),
  name: z.string(),
  task: TaskSchema,
  query_strategy: z.string(),
  data_loader: z.string(),
  timestamp: z.string().optional(),
  time_min: z.number().nullable().optional(),
  time_max: z.number().nullable().optional(),
  min_time_step: z.number().nullable().optional(),
  model_types: z.array(z.string()),
  shot_labels: z.array(z.string()).default([]),
  time_region_labels: z.array(z.string()).default([]),
  time_point_labels: z.array(z.string()).default([]),
  bounding_box_labels: z.array(z.string()).default([]),
  polygon_labels: z.array(z.string()).default([]),
  video_bounding_box_labels: z.array(z.string()).default([]),
});
export type Project = z.infer<typeof ProjectSchema>;

export const FileDataSchema = z.object({
  file_name: z.string(),
  type: z.string(),
  protocol: z.string(),
});
export type FileData = z.infer<typeof FileDataSchema>;

export const TimeSeriesFileDataSchema = FileDataSchema.extend({
  signal_names: z.array(z.string()),
});
export type TimeSeriesFileData = z.infer<typeof TimeSeriesFileDataSchema>;

export const ImageArrayFileDataSchema = FileDataSchema.extend({
  signal_name: z.string().optional(),
});
export type ImageArrayFileData = z.infer<typeof ImageArrayFileDataSchema>;

export const ShotDataSchema = z.object({
  protocol: z.string(),
  signal_names: z.array(z.string()),
});
export type ShotData = z.infer<typeof ShotDataSchema>;

export const SampleDataSchema = z.union([
  TimeSeriesFileDataSchema,
  ImageArrayFileDataSchema,
  FileDataSchema,
  ShotDataSchema,
]);
export type SampleData = z.infer<typeof SampleDataSchema>;

export const SampleSchema = z.object({
  _id: z.string().optional(),
  timestamp: z.string(),
  project_id: z.string().optional(),
  shot_id: z.number(),
  data: SampleDataSchema,
  validated_annotations: z.boolean(),
});
export type Sample = z.infer<typeof SampleSchema>;

export const SampleUpdateSchema = z.object({
  validated_annotations: z.boolean(),
});
export type SampleUpdate = z.infer<typeof SampleUpdateSchema>;

export const ModelSchema = z.object({
  _id: z.string(),
  timestamp: z.string(),
  project_id: z.string(),
  type: z.string(),
  version: z.int(),
  training_status: z.string(),
  progress: z.number(),
  score: z.number(),
  task_id: z.string(),
});

export type Model = z.infer<typeof ModelSchema>;

export const LocalLoadFormSchema = z.object({
  weights_path: z.string().min(1),
});
export type LocalLoadForm = z.infer<typeof LocalLoadFormSchema>;

export const GitlabLoadFormSchema = z.object({
  model_name: z.string().min(1),
  weights_path: z.string().min(1),
  model_version: z.string().nullish(),
  gitlab_project_id: z.number().min(1),
});
export type GitlabLoadForm = z.infer<typeof GitlabLoadFormSchema>;

export const HuggingfaceLoadFormSchema = z.object({
  model_name: z.string().min(1),
  weights_path: z.string().min(1),
  model_version: z.string().nullish(),
  huggingface_userspace: z.string().min(1),
});
export type HuggingfaceLoadForm = z.infer<typeof HuggingfaceLoadFormSchema>;

export const DataParamsSchema = z.object({
  name: z.string(),
  // Only used for video/image loader params.
  frame: z.number().nullable().optional(),
});
export type DataParams = z.infer<typeof DataParamsSchema>;
export const ImageDataParamsSchema = DataParamsSchema.extend({
  frame: z.number().nullable(),
});
export type ImageDataParams = z.infer<typeof ImageDataParamsSchema>;
export const SamplesSummarySchema = z.object({
  total: z.number(),
  shot_min: z.number().optional(),
  shot_max: z.number().optional(),
  data: SampleDataSchema,
});
export type SamplesSummary = z.infer<typeof SamplesSummarySchema>;

export const ViewParamsSchema = z.object({
  name: z.string(),
});
export type ViewParams = z.infer<typeof ViewParamsSchema>;

export const SpectrogramViewParamsSchema = ViewParamsSchema.extend({
  nperseg: z.number().optional(),
  time_min: z.number().optional(),
  time_max: z.number().optional(),
  frequency_min: z.number().optional(),
  frequency_max: z.number().optional(),
  amplitude_min: z.number().optional(),
  amplitude_max: z.number().optional(),
  threshold_value: z.number().optional(),
});
export type SpectrogramViewParams = z.infer<typeof SpectrogramViewParamsSchema>;

export const HealthInfoSchema = z.object({
  name: z.string(),
  version: z.string(),
  db_connected: z.boolean(),
  models_enabled: z.boolean(),
  gpu_available: z.boolean(),
});
export type HealthInfo = z.infer<typeof HealthInfoSchema>;

export type ToolingProps = {
  plotId?: string;
  plotReady?: boolean;
  forceUpdate?: number;
  onUpdate?: CallableFunction;
  selectedXRange?: [number, number];
};

export enum ToolingTypes {
  ZONE,
  VSPAN,
}

export enum TimeSeriesAnnotationType {
  TIME_POINT = "TIME POINT",
  TIME_REGION = "TIME REGION",
  BOUNDING_BOX = "BOUNDING BOX",
  POLYGON = "POLYGON",
}

export type TimeSeriesToolDefinition = {
  type: TimeSeriesAnnotationType;
  label: string;
};

export type TimeSeriesCategory = {
  label: string;
  color: string;
  type: TimeSeriesAnnotationType;
};

export type TimeSeriesAnnotationPoint = {
  x: number;
  y: number;
};

export type TimeSeriesAnnotation = {
  id: string;
  created_by: string;
  label: string;
  type: TimeSeriesAnnotationType;
  points: TimeSeriesAnnotationPoint[];
  selected: boolean;
};

export type ToolingCallbacks = {
  start: (
    x: number,
    y: number,
    label: string,
    axisSize: { x: number; y: number },
  ) => void;
  move: (x: number, y: number) => void;
  end: (x: number, y: number) => void;
  hover?: (x: number, y: number) => void;
  // Discards any in-progress annotation and resets tool-local state - called when a draw
  // is abandoned (tool switched mid-draw, Escape pressed) rather than completed normally
  cancel?: () => void;
};

export type PlotProps = {
  colorMap?: string;
  numSignificantDigits?: number;
  thresholdActive?: boolean;
};

type PlotlyAxisTransforms = {
  p2d: (pixels: number) => number;
  d2p: (value: number) => number;
  _tmax: number;
  _tmin: number;
  range: [number, number];
};
export interface ExtendedPlotlyHTMLElement extends PlotlyHTMLElement {
  _fullLayout: Record<string, PlotlyAxisTransforms>;
}

export interface SelectionRange {
  x: {
    low: number;
    high: number;
  };
  y: {
    low: number;
    high: number;
  };
}
