"use client";
import type { SortDescriptor } from "@react-types/shared";
import type {
  Project,
  Sample,
  SamplesSummary,
  SampleUpdate,
  Annotation,
  DataParams,
} from "@/types";
import { RJSFSchema } from "@rjsf/utils";

export let BACKEND_API_URL = "http://localhost:8002";
if (import.meta.env.VITE_DATA_API_URL) {
  BACKEND_API_URL = import.meta.env.VITE_DATA_API_URL;
}

export function apiFetch(
  url: string,
  options: RequestInit = {},
): Promise<Response> {
  const token = localStorage.getItem("tt_access_token");
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...((options.headers as Record<string, string>) ?? {}),
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
  return fetch(url, { ...options, headers });
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

// Fetch helpers below cast the response body directly, so a non-2xx response
// (e.g. 403 for a project you're not a member of) would otherwise be silently
// cast as if it were valid data. Call this before reading the body so callers
// get a real error to catch instead of garbage state.
export async function ensureOk(response: Response): Promise<Response> {
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      if (body && typeof body.detail === "string") {
        detail = body.detail;
      }
    } catch {
      // Body wasn't JSON — fall back to statusText.
    }
    throw new ApiError(response.status, detail);
  }
  return response;
}

export const getURL = async (url: string) => {
  const response = await apiFetch(url);
  const payload = await response.json();
  return payload;
};

export async function getSamplesSummary(
  project_id: string,
): Promise<SamplesSummary> {
  const url = `${BACKEND_API_URL}/projects/${project_id}/samples/summary`;
  const response = await apiFetch(url);
  const data = await response.json();
  const summary = data as SamplesSummary;
  return summary;
}

export const getSamples = async (
  sortDescriptor: SortDescriptor,
  project_id: string,
  page: number,
  samplesPerPage: number,
  shotId: string,
): Promise<Sample[]> => {
  const params = new URLSearchParams();
  params.append("sort_by", sortDescriptor.column.toString());
  params.append("sort_direction", sortDescriptor.direction);
  params.append("start", ((page - 1) * samplesPerPage).toString());
  params.append("count", samplesPerPage.toString());

  if (shotId !== "") {
    params.append("shot_id", shotId);
  }

  const url = `${BACKEND_API_URL}/projects/${project_id}/samples?${params.toString()}`;
  const response = await ensureOk(await apiFetch(url));
  const data = await response.json();
  const samples = data as Sample[];
  return samples;
};

export const getSample = async (
  project_id: string,
  sample_id: string,
): Promise<Sample> => {
  const response = await apiFetch(
    `${BACKEND_API_URL}/projects/${project_id}/samples/${sample_id}`,
  );
  const data = await response.json();
  const sample = data as Sample;
  return sample;
};

export const getNextSample = async (
  project_id: string,
  visited_sample_ids: string[],
  sortDescriptor: SortDescriptor | null,
): Promise<Sample | null> => {
  const params = new URLSearchParams();
  if (sortDescriptor) {
    params.append("sort_by", sortDescriptor.column.toString());
    params.append("sort_direction", sortDescriptor.direction);
  }
  const NEXT_URL =
    `${BACKEND_API_URL}/projects/${project_id}/samples/next` +
    (params.toString() ? `?${params.toString()}` : "");
  const sampleResult = await apiFetch(NEXT_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(visited_sample_ids),
  });

  if (sampleResult.status === 204) {
    return null; // No next sample available
  } else if (!sampleResult.ok) {
    throw new Error(
      `Failed to fetch next sample: ${sampleResult.status} ${sampleResult.statusText}`,
    );
  }
  const data = await sampleResult.json();
  const sample = data as Sample;
  return sample;
};

export const getProjects = async (
  sortDescriptor: SortDescriptor,
  page: number,
  projectsPerPage: number,
  name: string,
): Promise<Project[]> => {
  const params = new URLSearchParams();
  params.append("sort_by", sortDescriptor.column.toString());
  params.append("sort_direction", sortDescriptor.direction);
  params.append("start", ((page - 1) * projectsPerPage).toString());
  params.append("count", projectsPerPage.toString());
  if (name !== "") {
    params.append("name", name);
  }

  const response = await apiFetch(
    `${BACKEND_API_URL}/projects?${params.toString()}`,
  );
  const data = await response.json();
  const projects = data as Project[];
  return projects;
};

export const getProject = async (project_id: string): Promise<Project> => {
  const response = await ensureOk(
    await apiFetch(`${BACKEND_API_URL}/projects/${project_id}`),
  );
  const data = await response.json();
  const project = data as Project;
  return project;
};

export const deleteProject = async (project_id: string) => {
  const response = await apiFetch(`${BACKEND_API_URL}/projects/${project_id}`, {
    method: "DELETE",
    headers: {
      "Content-Type": "application/json",
    },
  });
  if (!response.ok && response.status !== 404) {
    throw new Error(`Failed to delete project: ${response.statusText}`);
  }
};

export async function getShotSample(project_id: string, shot_id: string) {
  const NEXT_URL = `${BACKEND_API_URL}/projects/${project_id}/samples?shot_id=${shot_id}`;
  const sampleResult = await apiFetch(NEXT_URL);
  const sampleArray = await sampleResult.json();
  let sample = null;
  if (sampleArray.length > 0) {
    sample = sampleArray[0];
  }
  return sample;
}

export async function getAnnotationsForSample(
  project_id: string,
  sample_id: string,
): Promise<Annotation[]> {
  const ANNOTATIONS_URL = `${BACKEND_API_URL}/projects/${project_id}/samples/${sample_id}/annotations`;
  const response = await apiFetch(ANNOTATIONS_URL);
  if (!response.ok) {
    throw new Error(`Failed to fetch annotations: ${response.statusText}`);
  }
  const annotations = await response.json();
  return annotations;
}

export async function getAnnotations(
  project_id: string,
): Promise<Annotation[]> {
  const ANNOTATIONS_URL = `${BACKEND_API_URL}/projects/${project_id}/annotations`;
  const response = await apiFetch(ANNOTATIONS_URL);
  if (!response.ok) {
    throw new Error(`Failed to fetch annotations: ${response.statusText}`);
  }
  const annotations = await response.json();
  return annotations;
}

export async function saveSampleAnnotations(
  project_id: string,
  sample_id: string,
  annotations: Annotation[],
  saveOnNavigate: boolean = true,
) {
  if (!saveOnNavigate) {
    return;
  }
  // Backend sets created_by from the JWT; just mark as validated
  const updatedAnnotations = annotations.map((annotation: Annotation) => ({
    ...annotation,
    validated: true,
  }));

  const ANNOTATIONS_URL = `${BACKEND_API_URL}/projects/${project_id}/samples/${sample_id}/annotations?validated=True`;
  const response = await apiFetch(ANNOTATIONS_URL, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(updatedAnnotations),
  });
  if (!response.ok) {
    throw new ApiError(
      response.status,
      `Failed to save annotations: ${response.statusText}`,
    );
  }
}

export async function saveAnnotations(
  project_id: string,
  annotations: Annotation[],
) {
  const ANNOTATIONS_URL = `${BACKEND_API_URL}/projects/${project_id}/annotations`;
  const response = await apiFetch(ANNOTATIONS_URL, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(annotations),
  });
  if (!response.ok) {
    throw new Error(`Failed to save annotations: ${response.statusText}`);
  }
}

export function importJSONFile(
  project_id: string,
  shot_id: number | null,
  file: File,
  callback?: () => void,
  errorCallback?: () => void,
): void {
  const reader = new FileReader();
  reader.onload = async (e) => {
    try {
      const parsed = JSON.parse(e.target?.result as string);
      const annotations = parsed as Annotation[];

      // If shot_id is provided, set it for all annotations that don't have it
      if (shot_id !== null) {
        annotations.map((annotation: Annotation) => {
          if (!annotation.shot_id) {
            annotation.shot_id = shot_id;
          }
          return annotation;
        });
      }

      await saveAnnotations(project_id, annotations);
      callback?.();
    } catch {
      errorCallback?.();
    }
  };
  reader.readAsText(file);
}

export function saveJSONToFile(data: object, filename: string) {
  const jsonStr = JSON.stringify(data, null, 2); // pretty print with 2 spaces
  const blob = new Blob([jsonStr], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url); // Clean up
}

export const exportAnnotations = async (project: Project, sample?: Sample) => {
  if (sample) {
    // Export annotations for the current sample only
    const annotations = await getAnnotationsForSample(project._id, sample._id);
    saveJSONToFile(
      annotations,
      `${project.name}_${sample.shot_id}_annotations.json`,
    );
  } else {
    // Export annotations for all samples in the project
    const annotations = await getAnnotations(project._id);
    saveJSONToFile(annotations, `${project.name}_all_annotations.json`);
  }
};

export const deleteSample = async (project_id: string, sample_id: string) => {
  await apiFetch(
    `${BACKEND_API_URL}/projects/${project_id}/samples/${sample_id}`,
    {
      method: "DELETE",
    },
  );
};

export const updateSample = async (
  project_id: string,
  sample_id: string,
  updates: SampleUpdate,
) => {
  const response = await apiFetch(
    `${BACKEND_API_URL}/projects/${project_id}/samples`,
    {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify([
        {
          _id: sample_id,
          updates: updates,
        },
      ]),
    },
  );
  const data = await response.json();
  const sample = data as Sample;
  return sample;
};

export const deleteSamples = async (project_id: string) => {
  await apiFetch(`${BACKEND_API_URL}/projects/${project_id}/samples`, {
    method: "DELETE",
  });
};

export const startTraining = async (
  project_id: string,
  selected_model: string,
  useGPU: boolean,
  params: Record<string, unknown>,
): Promise<Response> => {
  const response = await apiFetch(
    `${BACKEND_API_URL}/projects/${project_id}/models/${selected_model}/train?use_gpu=${useGPU}`,
    {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ params: params }),
    },
  );
  return response;
};

export const stopTraining = async (
  project_id: string,
  selected_model: string,
  version: number,
): Promise<Response> => {
  const response = await apiFetch(
    `${BACKEND_API_URL}/projects/${project_id}/models/${selected_model}/train?version=${version}`,
    {
      method: "DELETE",
      headers: {
        "Content-Type": "application/json",
      },
    },
  );
  return response;
};

export const startPredictions = async (
  project_id: string,
  selected_model: string,
  version: number,
  num_predictions: number,
  use_gpu: boolean,
  params: Record<string, unknown>,
): Promise<Response> => {
  const response = await apiFetch(
    `${BACKEND_API_URL}/projects/${project_id}/models/${selected_model}/predict?version=${version}&num_predictions=${num_predictions}&use_gpu=${use_gpu}`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ params: params }),
    },
  );
  return response;
};

export const startSamplePredictions = async (
  project_id: string,
  sample_id: string,
  selected_model: string,
  use_gpu: boolean,
  params: Record<string, unknown>,
  data_params: DataParams,
): Promise<Response> => {
  const response = await apiFetch(
    `${BACKEND_API_URL}/projects/${project_id}/samples/${sample_id}/models/${selected_model}/predict?use_gpu=${use_gpu}`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ params: params, data_params: data_params }),
    },
  );
  return response;
};

export const getSamplePredictions = async (
  project_id: string,
  sample_id: string,
  selected_model: string,
  task_id: string,
): Promise<Response> => {
  const response = await apiFetch(
    `${BACKEND_API_URL}/projects/${project_id}/samples/${sample_id}/models/${selected_model}/predict/${task_id}`,
    {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    },
  );
  return response;
};

export const getModels = async (project_id: string): Promise<Response> => {
  const response = await apiFetch(
    `${BACKEND_API_URL}/projects/${project_id}/models`,
  );
  return response;
};

export const getModelSchema = async (
  modelName: string,
  schemaType: string,
): Promise<RJSFSchema | null> => {
  const response = await apiFetch(
    `${BACKEND_API_URL}/meta/models/${modelName}/${schemaType}`,
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch model schema!`);
  }
  const data = await response.json();
  if (!data) {
    return null;
  }
  const schema: RJSFSchema = data as RJSFSchema;
  return schema;
};

export const getModelTrainSchema = async (
  modelName: string,
): Promise<RJSFSchema | null> => {
  return getModelSchema(modelName, "train");
};

export const getModelPredictSchema = async (
  modelName: string,
): Promise<RJSFSchema | null> => {
  return getModelSchema(modelName, "predict");
};

export const getModelTypes = async (task: string): Promise<Response> => {
  const response = await apiFetch(
    `${BACKEND_API_URL}/meta/models?task=${task}`,
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch model types!`);
  }
  return response;
};

export const getModelLoadTypes = async (): Promise<Response> => {
  const response = await apiFetch(`${BACKEND_API_URL}/meta/models/load`);
  if (!response.ok) {
    throw new Error(`Failed to fetch model types!`);
  }
  return response;
};

export const startLoadModelWeights = async (
  project_id: string,
  selected_model: string,
  load_method: string,
  weights_path: string,
): Promise<Response> => {
  const response = await apiFetch(
    `${BACKEND_API_URL}/projects/${project_id}/models/${selected_model}/load?method=${load_method}&weights_path=${weights_path}`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    },
  );
  return response;
};

export const getLoadModelStatus = async (
  project_id: string,
  selected_model: string,
  task_id: string,
): Promise<Response> => {
  const response = await apiFetch(
    `${BACKEND_API_URL}/projects/${project_id}/models/${selected_model}/load/${task_id}`,
    {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    },
  );
  return response;
};
