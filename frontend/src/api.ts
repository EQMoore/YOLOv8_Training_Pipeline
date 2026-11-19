// Thin client for the FastAPI service in ../api/main.py.

const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "/api";

export const ARTIFACTS = [
  "final_model.pt",
  "final_model.onnx",
  "final_model.quant.onnx",
] as const;
export type Artifact = (typeof ARTIFACTS)[number];

// Mirrors _NAME_RE on the server: names become GCS object paths.
export const NAME_RE = /^[A-Za-z0-9._-]{1,64}$/;

export const TERMINAL_STATES = new Set([
  "PIPELINE_STATE_SUCCEEDED",
  "PIPELINE_STATE_FAILED",
  "PIPELINE_STATE_CANCELLED",
]);

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function errorFrom(res: Response): Promise<ApiError> {
  let detail = res.statusText;
  try {
    const body = await res.json();
    if (typeof body?.detail === "string") detail = body.detail;
  } catch {
    /* non-JSON error body */
  }
  if (res.status === 401 || res.status === 403) {
    detail = "The API rejected your token. Check it in the token field above.";
  }
  return new ApiError(res.status, detail);
}

async function request(path: string, token: string, init: RequestInit = {}) {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { ...init.headers, Authorization: `Bearer ${token}` },
    });
  } catch {
    throw new ApiError(0, "Can't reach the API. Is the server running?");
  }
  if (!res.ok) throw await errorFrom(res);
  return res;
}

export interface ModelEntry {
  name: string;
  artifacts: Artifact[];
  hasDataset: boolean;
}

export interface DatasetEntry {
  name: string;
  path: string;
  hasModel: boolean;
}

export interface BucketListing {
  models: ModelEntry[];
  datasets: DatasetEntry[];
}

// /get_models returns every blob under "{user}/": dataset zips live at
// "{user}/{model}.zip" and trained artifacts at "{user}/{model}/{artifact}".
export function parseListing(blobs: string[]): BucketListing {
  const artifacts = new Map<string, Artifact[]>();
  const zips = new Map<string, string>();
  for (const blob of blobs) {
    const parts = blob.split("/");
    if (parts.length === 2 && parts[1].endsWith(".zip")) {
      zips.set(parts[1].slice(0, -4), blob);
    } else if (parts.length === 3 && (ARTIFACTS as readonly string[]).includes(parts[2])) {
      const list = artifacts.get(parts[1]) ?? [];
      list.push(parts[2] as Artifact);
      artifacts.set(parts[1], list);
    }
  }
  const models = [...artifacts.entries()]
    .map(([name, list]) => ({
      name,
      artifacts: ARTIFACTS.filter((a) => list.includes(a)),
      hasDataset: zips.has(name),
    }))
    .sort((a, b) => a.name.localeCompare(b.name));
  const datasets = [...zips.entries()]
    .map(([name, path]) => ({ name, path, hasModel: artifacts.has(name) }))
    .sort((a, b) => a.name.localeCompare(b.name));
  return { models, datasets };
}

export async function listBucket(token: string): Promise<BucketListing> {
  const res = await request("/get_models", token);
  return parseListing(await res.json());
}

export interface JobStatus {
  job_id: string;
  state: string;
  error?: string;
}

export async function getJobStatus(token: string, jobId: string): Promise<JobStatus> {
  const res = await request(`/job_status?job_id=${encodeURIComponent(jobId)}`, token);
  return res.json();
}

export async function downloadArtifact(token: string, model: string, artifact: Artifact) {
  const qs = new URLSearchParams({ model, artifact });
  const res = await request(`/download_model_file?${qs}`, token);
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${model}.${artifact.replace(/^final_model\./, "")}`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export interface TrainParams {
  model: string;
  epochs: number;
  batch: number;
  arch: string;
}

export interface TrainResponse {
  job_id: string;
  user_id: string;
  model: string;
  status: string;
}

// XHR rather than fetch so the dataset upload can report progress.
export function submitTraining(
  token: string,
  dataset: File,
  params: TrainParams,
  onProgress: (fraction: number) => void,
): Promise<TrainResponse> {
  const form = new FormData();
  form.append("dataset", dataset);
  form.append("model", params.model);
  form.append("epochs", String(params.epochs));
  form.append("batch", String(params.batch));
  form.append("arch", params.arch);

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/train_yolo`);
    xhr.setRequestHeader("Authorization", `Bearer ${token}`);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => {
      let body: unknown = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        /* non-JSON body */
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(body as TrainResponse);
        return;
      }
      let detail =
        (body as { detail?: unknown } | null)?.detail ?? xhr.statusText ?? "Request failed";
      if (typeof detail !== "string") detail = "The API rejected the request.";
      if (xhr.status === 401 || xhr.status === 403) {
        detail = "The API rejected your token. Check it in the token field above.";
      } else if (xhr.status === 409) {
        detail = `A model named "${params.model}" already exists. Pick another name.`;
      }
      reject(new ApiError(xhr.status, detail as string));
    };
    xhr.onerror = () => reject(new ApiError(0, "Can't reach the API. Is the server running?"));
    xhr.send(form);
  });
}

export function formatState(state: string | undefined): string {
  if (!state) return "Checking…";
  const s = state.replace(/^PIPELINE_STATE_/, "").replace(/_/g, " ").toLowerCase();
  return s.charAt(0).toUpperCase() + s.slice(1);
}
