import { useEffect, useRef, useState, type Dispatch, type FormEvent, type SetStateAction } from "react";
import {
  ApiError,
  NAME_RE,
  TERMINAL_STATES,
  formatState,
  getJobStatus,
  submitTraining,
} from "../api";
import type { StagedDataset, TabId } from "../App";
import type { TrackedJob } from "../storage";
import { formatBytes } from "../zip";

const ARCHS = [
  { id: "yolov8n", label: "YOLOv8n — nano, fastest" },
  { id: "yolov8s", label: "YOLOv8s — small" },
  { id: "yolov8m", label: "YOLOv8m — medium" },
  { id: "yolov8l", label: "YOLOv8l — large" },
  { id: "yolov8x", label: "YOLOv8x — extra large, most accurate" },
];

const POLL_MS = 15000;

interface Props {
  token: string;
  staged: StagedDataset | null;
  clearStaged: () => void;
  jobs: TrackedJob[];
  setJobs: Dispatch<SetStateAction<TrackedJob[]>>;
  onJobFinished: () => void;
  goTo: (tab: TabId) => void;
}

function stateTone(state?: string) {
  if (!state) return "pending";
  if (state.endsWith("SUCCEEDED")) return "ok";
  if (state.endsWith("FAILED") || state.endsWith("CANCELLED")) return "bad";
  return "running";
}

export default function TrainingTab({
  token,
  staged,
  clearStaged,
  jobs,
  setJobs,
  onJobFinished,
  goTo,
}: Props) {
  const [model, setModel] = useState(() =>
    staged ? staged.file.name.replace(/\.zip$/i, "").replace(/[^A-Za-z0-9._-]/g, "-").slice(0, 64) : "",
  );
  const [arch, setArch] = useState("yolov8n");
  const [epochs, setEpochs] = useState(10);
  const [batch, setBatch] = useState(16);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [justSubmitted, setJustSubmitted] = useState<string | null>(null);

  const nameValid = NAME_RE.test(model);
  const canSubmit =
    !!token && !!staged && nameValid && epochs >= 1 && batch >= 1 && progress === null;

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!canSubmit || !staged) return;
    setError(null);
    setJustSubmitted(null);
    setProgress(0);
    try {
      const res = await submitTraining(token, staged.file, { model, arch, epochs, batch }, setProgress);
      setJobs((prev) => [
        { job_id: res.job_id, model, arch, epochs, batch, submittedAt: new Date().toISOString() },
        ...prev,
      ]);
      setJustSubmitted(model);
      clearStaged();
      setModel("");
    } catch (err) {
      setError((err as ApiError).message);
    } finally {
      setProgress(null);
    }
  };

  // Poll every job that hasn't reached a terminal state.
  const jobsRef = useRef(jobs);
  jobsRef.current = jobs;
  useEffect(() => {
    if (!token) return;
    let cancelled = false;
    const poll = async () => {
      const pending = jobsRef.current.filter((j) => !j.state || !TERMINAL_STATES.has(j.state));
      let finished = false;
      for (const job of pending) {
        try {
          const s = await getJobStatus(token, job.job_id);
          if (cancelled) return;
          if (TERMINAL_STATES.has(s.state)) finished = true;
          setJobs((prev) =>
            prev.map((j) => (j.job_id === job.job_id ? { ...j, state: s.state, error: s.error } : j)),
          );
        } catch {
          /* transient; try again next tick */
        }
      }
      if (finished && !cancelled) onJobFinished();
    };
    poll();
    const id = setInterval(poll, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [token, jobs.length, setJobs, onJobFinished]);

  const forget = (jobId: string) => setJobs((prev) => prev.filter((j) => j.job_id !== jobId));

  return (
    <section>
      <div className="section-head">
        <div>
          <h2>Training</h2>
          <p className="lede">
            Fine-tune a YOLOv8 detector on your staged dataset. When it finishes, the model is
            exported to PyTorch, ONNX and int8 ONNX.
          </p>
        </div>
      </div>

      <form className="train-form" onSubmit={submit}>
        <div className="field wide">
          <span className="field-label">Dataset</span>
          {staged ? (
            <div className="dataset-pill">
              <span className="strong">{staged.file.name}</span>
              <span className="muted">{formatBytes(staged.file.size)}</span>
              <button type="button" className="link" onClick={() => goTo("upload")}>
                Change
              </button>
            </div>
          ) : (
            <div className="dataset-pill empty-pill">
              <span className="muted">No dataset staged.</span>
              <button type="button" className="link" onClick={() => goTo("upload")}>
                Upload a dataset
              </button>
            </div>
          )}
        </div>

        <label className="field wide">
          <span className="field-label">Model name</span>
          <input
            value={model}
            onChange={(e) => setModel(e.target.value)}
            placeholder="car-detector"
            maxLength={64}
            aria-invalid={model !== "" && !nameValid}
            aria-describedby="model-hint"
          />
          <span id="model-hint" className={`field-hint${model && !nameValid ? " error-text" : ""}`}>
            Letters, numbers, dots, dashes and underscores. Must be unused.
          </span>
        </label>

        <label className="field">
          <span className="field-label">Base model</span>
          <select value={arch} onChange={(e) => setArch(e.target.value)}>
            {ARCHS.map((a) => (
              <option key={a.id} value={a.id}>
                {a.label}
              </option>
            ))}
          </select>
        </label>

        <label className="field">
          <span className="field-label">Epochs</span>
          <input
            type="number"
            min={1}
            max={1000}
            value={epochs}
            onChange={(e) => setEpochs(Number(e.target.value))}
          />
        </label>

        <label className="field">
          <span className="field-label">Batch size</span>
          <input
            type="number"
            min={1}
            max={256}
            value={batch}
            onChange={(e) => setBatch(Number(e.target.value))}
          />
        </label>

        <div className="actions wide">
          <button className="btn" type="submit" disabled={!canSubmit}>
            {progress === null
              ? "Start training"
              : progress < 1
                ? `Uploading ${Math.round(progress * 100)}%`
                : "Submitting job…"}
          </button>
          {progress !== null && (
            <div className="progress" aria-hidden="true">
              <div style={{ width: `${Math.round(progress * 100)}%` }} />
            </div>
          )}
        </div>

        {error && <p className="alert error wide">{error}</p>}
        {justSubmitted && (
          <p className="alert ok wide">
            Training started for {justSubmitted}. Its status updates below.
          </p>
        )}
      </form>

      <div className="section-head sub">
        <h2>Jobs</h2>
        <p className="lede small">Jobs started from this browser. Status refreshes every 15 seconds.</p>
      </div>

      {jobs.length === 0 ? (
        <p className="muted">No jobs yet.</p>
      ) : (
        <ul className="rows">
          {jobs.map((j) => {
            const tone = stateTone(j.state);
            const done = j.state && TERMINAL_STATES.has(j.state);
            return (
              <li key={j.job_id} className={`row job ${tone}`}>
                <div className="row-main">
                  <h3 className="row-title">{j.model}</h3>
                  <p className="row-meta">
                    {j.arch}, {j.epochs} epochs, batch {j.batch} · started{" "}
                    {new Date(j.submittedAt).toLocaleString()}
                  </p>
                  <p className="row-meta mono">{j.job_id}</p>
                  {j.error && <p className="alert error">{j.error}</p>}
                </div>
                <div className="job-side">
                  <span className={`state ${tone}`}>{formatState(j.state)}</span>
                  {tone === "ok" && (
                    <button className="link" onClick={() => goTo("models")}>
                      View model
                    </button>
                  )}
                  {done && (
                    <button className="link muted" onClick={() => forget(j.job_id)}>
                      Remove
                    </button>
                  )}
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
