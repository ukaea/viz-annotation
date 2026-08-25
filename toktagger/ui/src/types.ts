import { Config, PlotlyHTMLElement } from "plotly.js";
import { z } from "zod/v4";

export const BaseAnnotationSchema = z.object({
  project_id: z.string().nullable().default(null),
  sample_id: z.string().nullable().default(null),
  shot_id: z.number().optional(),
  timestamp: z.string().optional(),
  validated: z.boolean().nullable().default(null),
  uncertainty: z.number().nullable().default(1),
  created_by: z.string().default("manual"),
  signal_name: z.string().nullable().default(null),
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

// A label tied to one frame and one tracked instance. The base schema for other video annotation types.
export const VideoFrameSchema = BaseAnnotationSchema.extend({
  type: z.literal("video_frame_label"),
  frame: z.number().int(),
  track_id: z.string(), // force string
});

export type VideoFrame = z.infer<typeof VideoFrameSchema>;

export const VideoBoundingBoxSchema = VideoFrameSchema.extend({
  type: z.literal("video_bounding_box"),
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

export const VideoPolygonSchema = VideoFrameSchema.extend({
  type: z.literal("video_polygon"),
  segmentation: z.array(VideoPolygonCoordinatesSchema).length(1),
});

export type VideoPolygon = z.infer<typeof VideoPolygonSchema>;

export const VideoPointSchema = VideoFrameSchema.extend({
  type: z.literal("video_point"),
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
  VideoFrameSchema,
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

export const Profile2DDataSchema = z.object({
  time: z.array(z.number()),
  dim_1: z.array(z.number()),
  // Nullable: the API serialises NaN samples in a profile as null.
  values: z.array(z.array(z.number().nullable())),
});
export type Profile2DData = z.infer<typeof Profile2DDataSchema>;
export const ImageDataSchema = z.object({
  frame: z.number(),
  values: z.string(), // base64 PNG
});
export type ImageData = z.infer<typeof ImageDataSchema>;

export const DataSchema = z.union([
  TimeSeriesDataSchema,
  MultiVariateTimeSeriesDataSchema,
  Profile2DDataSchema,
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

export const DisplayAnnotationSchema = z.union([ZoneSchema, VSpanSchema]);
export type DisplayAnnotation = z.infer<typeof DisplayAnnotationSchema>;

export enum TaskType {
  TimeSeries = "time-series",
  Profile2D = "profile-2d",
  Video = "video",
}

export const TaskSchema = z.enum([
  TaskType.TimeSeries,
  TaskType.Profile2D,
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
  status: z.string(),
  progress: z.number(),
  score: z.number(),
  task_id: z.string(),
});

export type Model = z.infer<typeof ModelSchema>;

export const LocalLoadFormSchema = z.object({
  weights_path: z.string().nonempty(),
});
export type LocalLoadForm = z.infer<typeof LocalLoadFormSchema>;

export const GitlabLoadFormSchema = z.object({
  model_name: z.string().nonempty(),
  weights_path: z.string().nonempty(),
  model_version: z.string().nullish(),
  gitlab_project_id: z.number().min(1),
});
export type GitlabLoadForm = z.infer<typeof GitlabLoadFormSchema>;

export const HuggingfaceLoadFormSchema = z.object({
  model_name: z.string().nonempty(),
  weights_path: z.string().nonempty(),
  model_version: z.string().nullish(),
  huggingface_userspace: z.string().nonempty(),
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
  name: z.literal("identity"),
});
export type ViewParams = z.infer<typeof ViewParamsSchema>;

export const Profile2DViewParamsSchema = ViewParamsSchema.extend({
  name: z.literal("profile_2d"),
  signal_name: z.string(),
  log_scale: z.boolean().default(false),
  time_min: z.number().optional(),
  time_max: z.number().optional(),
  dim_1_min: z.number().optional(),
  dim_1_max: z.number().optional(),
  values_min: z.number().optional(),
  values_max: z.number().optional(),
});
export type Profile2DViewParams = z.infer<typeof Profile2DViewParamsSchema>;

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
  // Restricts rendering to a single subplot (e.g. "xy2"); all subplots by default.
  subplot?: string;
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
  // Binds the annotation to a specific signal; null for single-signal views.
  signal_name?: string | null;
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
  hover?: (x: number, y: number, axisSize: { x: number; y: number }) => void;
  // Called when a draw is abandoned (tool switched or Escape pressed) instead of finished
  cancel?: () => void;
  // Alternative gesture for finishing an in-progress shape (e.g. double-click to close a polygon)
  doubleClick?: (x: number, y: number) => void;
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
  _context: { doubleClick: Config["doubleClick"] };
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
