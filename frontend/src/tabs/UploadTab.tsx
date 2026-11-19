import { useRef, useState } from "react";
import type { StagedDataset, TabId } from "../App";
import { checkDataset, formatBytes } from "../zip";

interface Props {
  staged: StagedDataset | null;
  setStaged: (s: StagedDataset | null) => void;
  goTo: (tab: TabId) => void;
}

export default function UploadTab({ staged, setStaged, goTo }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [checking, setChecking] = useState(false);

  const accept = async (file: File | undefined) => {
    if (!file) return;
    setChecking(true);
    try {
      setStaged({ file, check: await checkDataset(file) });
    } finally {
      setChecking(false);
    }
  };

  const check = staged?.check;
  const blocked = !!check && check.problems.length > 0;

  return (
    <section>
      <div className="section-head">
        <div>
          <h2>Upload</h2>
          <p className="lede">
            Add a dataset ZIP in Ultralytics YOLO format: an <code>images/</code> folder, a{" "}
            <code>labels/</code> folder, and a <code>data.yaml</code>. It's checked here first,
            then uploaded when you start training.
          </p>
        </div>
      </div>

      <div
        className={`dropzone${dragging ? " dragging" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          accept(e.dataTransfer.files[0]);
        }}
      >
        <p className="dropzone-title">
          {checking ? "Checking ZIP…" : "Drop a dataset ZIP here"}
        </p>
        <button className="btn" onClick={() => inputRef.current?.click()} disabled={checking}>
          Choose file
        </button>
        <input
          ref={inputRef}
          type="file"
          accept=".zip,application/zip"
          hidden
          onChange={(e) => {
            accept(e.target.files?.[0]);
            e.target.value = "";
          }}
        />
      </div>

      {staged && check && (
        <div className={`staged detection${blocked ? " bad" : ""}`}>
          <div className="staged-head">
            <div>
              <h3 className="row-title">{staged.file.name}</h3>
              <p className="row-meta">{formatBytes(staged.file.size)}</p>
            </div>
            <button className="btn ghost small" onClick={() => setStaged(null)}>
              Remove
            </button>
          </div>

          {check.readable && (
            <dl className="stats">
              <div>
                <dt>data.yaml</dt>
                <dd className="mono">{check.dataYaml ?? "missing"}</dd>
              </div>
              <div>
                <dt>Images</dt>
                <dd>{check.images.toLocaleString()}</dd>
              </div>
              <div>
                <dt>Label files</dt>
                <dd>{check.labels.toLocaleString()}</dd>
              </div>
              <div>
                <dt>Files in ZIP</dt>
                <dd>{check.files.toLocaleString()}</dd>
              </div>
            </dl>
          )}

          {check.problems.map((p) => (
            <p key={p} className="alert error">
              {p}
            </p>
          ))}
          {check.warnings.map((w) => (
            <p key={w} className="alert warn">
              {w}
            </p>
          ))}

          <div className="actions">
            <button className="btn" disabled={blocked} onClick={() => goTo("training")}>
              Continue to training
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
