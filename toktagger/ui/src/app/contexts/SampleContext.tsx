"use client";
import React, {
  createContext,
  useCallback,
  useContext,
  useState,
  useEffect,
  ReactNode,
  useRef,
} from "react";
import { ToastQueue } from "@adobe/react-spectrum";
import { z } from "zod/v4";
import {
  Project,
  Sample,
  Data,
  Annotation,
  ViewParams,
  ViewParamsSchema,
  PlotProps,
  Profile2DViewParams,
  Profile2DViewParamsSchema,
  MultiVariateTimeSeriesData,
  Profile2DData,
  MultiVariateTimeSeriesDataSchema,
  Profile2DDataSchema,
  ImageData,
  ImageDataSchema,
  TaskType,
  DataParams,
} from "@/types";
import { ApiError, BACKEND_API_URL, apiFetch, ensureOk } from "@/app/core";
import { getSignalNames } from "@/app/utils";

const viewParamsKey = (projectId: string) => `view-params-${projectId}`;
const colorMapKey = (projectId: string) => `color-map-${projectId}`;

// Reads persisted view params, discarding anything that fails schema validation.
function readSavedViewParams(
  projectId: string,
): ViewParams | Profile2DViewParams {
  const fallback: ViewParams = { name: "identity" };
  if (!projectId) return fallback;

  const saved = sessionStorage.getItem(viewParamsKey(projectId));
  if (!saved) return fallback;

  try {
    const parsed: unknown = JSON.parse(saved);
    const result = z
      .union([ViewParamsSchema, Profile2DViewParamsSchema])
      .safeParse(parsed);
    if (result.success) return result.data;
  } catch {
    // Malformed JSON - fall through and discard.
  }

  sessionStorage.removeItem(viewParamsKey(projectId));
  return fallback;
}

function readSavedColorMap(projectId: string): string | null {
  if (!projectId) return null;
  return sessionStorage.getItem(colorMapKey(projectId));
}

interface SampleContextType {
  project: Project | null;
  sample: Sample | null;
  data: Data | null;
  annotations: Annotation[];
  // The annotations as the server last returned them, so a save can tell which of
  // them the user removed locally.
  serverAnnotations: Annotation[];
  dataParams: DataParams;
  viewParams: ViewParams | Profile2DViewParams;
  plotProps: PlotProps;
  annotationLabels: { id: number; name: string }[];
  videoFrameBounds: { min: number | null; max: number | null };
  isLoading: boolean;
  isValidated: boolean | null;
  error: string | null;
  // HTTP status behind `error`, when it came from the API. Lets the page tell
  // "no access" (403) apart from a genuine failure.
  errorStatus: number | null;
  setAnnotations: React.Dispatch<React.SetStateAction<Annotation[]>>;
  // Replaces the working set with a freshly fetched one, so `serverAnnotations`
  // stays the baseline a save diffs against.
  syncAnnotationsFromServer: (annotations: Annotation[]) => void;
  setDataParams: React.Dispatch<React.SetStateAction<DataParams>>;
  setViewParams: React.Dispatch<
    React.SetStateAction<ViewParams | Profile2DViewParams>
  >;
  setPlotProps: (props: PlotProps) => void;
  setIsValidated: (validated: boolean) => void;
}

const SampleContext = createContext<SampleContextType | undefined>(undefined);

interface SampleProviderProps {
  projectId: string;
  sampleId: string;
  children: ReactNode;
}

async function getData<T>(url: string, signal?: AbortSignal): Promise<T> {
  const response = await ensureOk(await apiFetch(url, { signal }));
  const payload = await response.json();
  return payload as T;
}

async function getSample(
  projectId: string,
  sampleId: string,
  signal?: AbortSignal,
): Promise<Sample> {
  return await getData<Sample>(
    `${BACKEND_API_URL}/projects/${projectId}/samples/${sampleId}`,
    signal,
  );
}

async function getProject(
  projectId: string,
  signal?: AbortSignal,
): Promise<Project> {
  return await getData<Project>(
    `${BACKEND_API_URL}/projects/${projectId}`,
    signal,
  );
}

async function getAnnotations(
  projectId: string,
  sampleId: string,
  signal?: AbortSignal,
): Promise<Annotation[]> {
  return await getData<Annotation[]>(
    `${BACKEND_API_URL}/projects/${projectId}/samples/${sampleId}/annotations`,
    signal,
  );
}

async function parseData(
  data: Data,
  task: TaskType,
): Promise<MultiVariateTimeSeriesData | Profile2DData | ImageData | undefined> {
  if (task == TaskType.TimeSeries) {
    const result = MultiVariateTimeSeriesDataSchema.safeParse(data);
    if (!result.success) {
      throw new Error("Invalid data for time series view");
    }
    return result.data;
  } else if (task == TaskType.Profile2D) {
    // The server selects the signal, so the response is a single profile.
    const result = Profile2DDataSchema.safeParse(data);
    if (!result.success) {
      throw new Error("Invalid data for profile 2D view");
    }
    return result.data;
  } else if (task == TaskType.Video) {
    const result = ImageDataSchema.safeParse(data);
    if (!result.success) {
      throw new Error("Invalid data for video view");
    }
    return result.data;
  }

  return undefined;
}

export function SampleProvider({
  projectId,
  sampleId,
  children,
}: SampleProviderProps) {
  const [project, setProject] = useState<Project | null>(null);
  const [sample, setSample] = useState<Sample | null>(null);
  const [data, setData] = useState<Data | null>(null);
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [serverAnnotations, setServerAnnotations] = useState<Annotation[]>([]);

  const [viewParams, setViewParams] = useState<
    ViewParams | Profile2DViewParams
  >(() => readSavedViewParams(projectId));

  const [dataParams, setDataParams] = useState<DataParams>({
    name: "identity",
  });
  const [prevSampleId, setPrevSampleId] = useState(sampleId);

  const [plotProps, setPlotProps] = useState<PlotProps>(() => ({
    colorMap: readSavedColorMap(projectId) ?? "Cividis",
  }));

  // Persist view params so they survive a refresh or navigating between samples.
  useEffect(() => {
    if (!projectId) return;
    sessionStorage.setItem(
      viewParamsKey(projectId),
      JSON.stringify(viewParams),
    );
  }, [viewParams, projectId]);

  // Persist only the colour map - thresholdActive has its own source of truth.
  useEffect(() => {
    if (!projectId || !plotProps.colorMap) return;
    sessionStorage.setItem(colorMapKey(projectId), plotProps.colorMap);
  }, [plotProps.colorMap, projectId]);

  const [isLoading, setIsLoading] = useState<boolean>(true);

  const [isValidated, setIsValidated] = useState<boolean | null>(null);

  const [error, setError] = useState<string | null>(null);
  const [errorStatus, setErrorStatus] = useState<number | null>(null);
  const [videoFrameBounds, setVideoFrameBounds] = useState<{
    min: number | null;
    max: number | null;
  }>({ min: null, max: null });

  // Video: remember the last successfully loaded frame so missing frames become navigation bounds.
  const lastGoodVideoFrameRef = useRef<number | null>(null);
  // Video: track which sample has already had its first-frame bootstrap request.
  const bootstrappedVideoSampleIdRef = useRef<string | null>(null);
  // Annotations are sample-level working state. Do not reload them on every video frame fetch.
  const loadedAnnotationsSampleKeyRef = useRef<string | null>(null);
  // Only the latest data request may update SampleContext state.
  const dataRequestIdRef = useRef(0);

  if (prevSampleId !== sampleId) {
    setPrevSampleId(sampleId);
    setDataParams({ name: "identity" });
  }

  useEffect(() => {
    setVideoFrameBounds({ min: null, max: null });
    lastGoodVideoFrameRef.current = null;
  }, [sampleId]);

  function extractDetail(payload: unknown): string {
    if (!payload) return "Unknown error";
    if (typeof payload === "string") return payload;
    if (typeof payload === "object") {
      const d = (payload as { detail?: unknown }).detail;
      if (typeof d === "string" && d.trim()) return d;
      if (Array.isArray(d)) {
        const first = d.find((x) => typeof x === "string" && x.trim());
        if (typeof first === "string") return first;
      }
    }
    try {
      return JSON.stringify(payload);
    } catch {
      return "Unknown error";
    }
  }

  function isMissingFrameError(status: number, detail: string): boolean {
    // Treat 404 + common "missing frame" phrasing as "navigation boundary" rather than fatal.
    if (status === 404) return true;
    const msg = (detail || "").toLowerCase();
    return (
      msg.includes("could not find image") ||
      msg.includes("file not found") ||
      msg.includes("no such file") ||
      msg.includes("frame index")
    );
  }

  // Consolidated data fetching - fetch everything together
  useEffect(() => {
    const requestId = ++dataRequestIdRef.current;
    const controller = new AbortController();
    const isCurrentRequest = () =>
      dataRequestIdRef.current === requestId && !controller.signal.aborted;

    const refreshData = async () => {
      setIsLoading(true);
      setError(null);
      setErrorStatus(null);

      try {
        // Fetch project, sample, and annotations in parallel
        const [projectData, sampleData, dbAnnotations] = await Promise.all([
          getProject(projectId, controller.signal),
          getSample(projectId, sampleId, controller.signal),
          getAnnotations(projectId, sampleId, controller.signal),
        ]);

        if (!isCurrentRequest()) return;

        setProject(projectData);
        setSample(sampleData);
        const sampleKey = `${projectId}:${sampleId}`;
        if (loadedAnnotationsSampleKeyRef.current !== sampleKey) {
          setAnnotations(dbAnnotations);
          setServerAnnotations(dbAnnotations);
          loadedAnnotationsSampleKeyRef.current = sampleKey;
        }
        setIsValidated(sampleData.validated_annotations);

        let params = viewParams;
        // Default to the sample's first signal until one is explicitly selected.
        if (
          projectData.task === TaskType.Profile2D &&
          params.name !== "profile_2d"
        ) {
          params = {
            name: "profile_2d",
            signal_name: getSignalNames(sampleData)[0],
          } as Profile2DViewParams;
        }

        // ------------------------------------------------------------
        // video projects must request image data parameters.
        // Backend ImageDataLoader requires params.name === "image".
        // frame: null means "backend picks first frame automatically".
        // ------------------------------------------------------------
        let effectiveDataParams: DataParams = dataParams;

        if (projectData.task === TaskType.Video) {
          const isFirstRequestForSample =
            bootstrappedVideoSampleIdRef.current !== sampleId;
          effectiveDataParams = {
            ...dataParams,
            name: "image",
            frame: isFirstRequestForSample ? null : (dataParams.frame ?? null),
          };
        }

        const response = await apiFetch(
          `${BACKEND_API_URL}/projects/${projectId}/samples/${sampleId}/data`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ params: effectiveDataParams, view: params }),
            signal: controller.signal,
          },
        );

        if (!isCurrentRequest()) return;

        if (!response.ok) {
          let payload: unknown = null;
          try {
            payload = await response.json();
          } catch {
            // ignore; payload stays null
          }

          if (!isCurrentRequest()) return;

          const detail = extractDetail(payload);

          // Video-only: treat missing frame as "boundary" and stay on last good frame.
          if (projectData.task === TaskType.Video) {
            const requestedFrame = effectiveDataParams.frame;

            const lastGood = lastGoodVideoFrameRef.current;

            if (
              typeof requestedFrame === "number" &&
              typeof lastGood === "number" &&
              requestedFrame !== lastGood &&
              isMissingFrameError(response.status, detail)
            ) {
              ToastQueue.negative(`Frame ${requestedFrame} not found.`, {
                timeout: 2500,
              });

              setVideoFrameBounds((prev) => {
                // Only tighten bounds for adjacent navigation attempts.
                // Large jump probes (e.g. 0 -> 5000) should not clamp next/prev.
                if (Math.abs(requestedFrame - lastGood) !== 1) {
                  return prev;
                }

                if (requestedFrame < lastGood) {
                  return { ...prev, min: lastGood };
                }
                return { ...prev, max: lastGood };
              });

              // Roll back params; do NOT set error and do NOT clear data.
              setDataParams((prev) => ({
                ...prev,
                name: "image",
                frame: lastGood,
              }));

              return;
            }
          }

          setError(detail);
          setErrorStatus(response.status);
          setData(null);
          return;
        }

        const fetchedData: Data = await response.json();

        if (!isCurrentRequest()) return;

        const viewData = await parseData(fetchedData, projectData.task);
        if (!isCurrentRequest()) return;
        if (!viewData) {
          setError("Data could not read the data for the selected view");
          return;
        }

        // Video: remember last good frame so we can roll back on missing-frame errors.
        if (projectData.task === TaskType.Video) {
          const frame = (viewData as { frame?: unknown }).frame;
          if (typeof frame === "number" && Number.isFinite(frame)) {
            const requestedFrame = effectiveDataParams.frame;

            // Never display a response for a different explicitly requested frame.
            if (
              typeof requestedFrame === "number" &&
              frame !== requestedFrame
            ) {
              return;
            }

            setData(viewData);
            bootstrappedVideoSampleIdRef.current = sampleId;
            lastGoodVideoFrameRef.current = frame;

            // Only the initial frame:null request needs the backend response to
            // establish the requested frame. Explicit requests already carry it.
            if (requestedFrame === null || requestedFrame === undefined) {
              setDataParams((prev) => {
                if (prev.name === "image" && prev.frame === frame) {
                  return prev;
                }
                return {
                  name: "image",
                  frame,
                };
              });
            }

            setVideoFrameBounds((prev) => ({
              ...prev,
              min: prev.min === null ? frame : Math.min(prev.min, frame),
            }));
          }
        } else {
          setData(viewData);
        }
      } catch (err) {
        if (
          controller.signal.aborted ||
          (err instanceof Error && err.name === "AbortError") ||
          !isCurrentRequest()
        ) {
          return;
        }
        if (err instanceof ApiError && err.status === 403) {
          // Reported as a refusal rather than as a missing project, so a user who
          // has lost access - or was never given it - can act on the message.
          setError(err.message || "You are not a member of this project.");
          setErrorStatus(403);
        } else if (err instanceof ApiError && err.status === 404) {
          setError("Project not found.");
          setErrorStatus(404);
        } else {
          setError(err instanceof Error ? err.message : "An error occurred");
          setErrorStatus(err instanceof ApiError ? err.status : null);
        }
      } finally {
        if (isCurrentRequest()) {
          setIsLoading(false);
        }
      }
    };

    refreshData();

    return () => {
      controller.abort();
    };
  }, [projectId, sampleId, dataParams, viewParams, plotProps]);

  const syncAnnotationsFromServer = useCallback((fetched: Annotation[]) => {
    setAnnotations(fetched);
    setServerAnnotations(fetched);
  }, []);

  const annotationLabels =
    project?.task === TaskType.Video
      ? (project.video_bounding_box_labels || []).map((name, i) => ({
          id: i + 1,
          name,
        }))
      : [];

  const value: SampleContextType = {
    project,
    sample,
    data,
    annotations,
    serverAnnotations,
    dataParams,
    viewParams,
    plotProps,
    annotationLabels,
    videoFrameBounds,
    isLoading,
    isValidated,
    error,
    errorStatus,
    setAnnotations,
    syncAnnotationsFromServer,
    setPlotProps,
    setViewParams,
    setDataParams,
    setIsValidated,
  };

  return (
    <SampleContext.Provider value={value}>{children}</SampleContext.Provider>
  );
}

export function useSample() {
  const context = useContext(SampleContext);
  if (context === undefined) {
    throw new Error("useSample must be used within a SampleProvider");
  }
  return context;
}
