import { useState } from "react";
import { Artifact, downloadArtifact, ApiError } from "../api";
import type { ListingState, TabId } from "../App";

const ARTIFACT_INFO: Record<Artifact, { label: string; hint: string }> = {
  "final_model.pt": { label: "PyTorch", hint: ".pt — fine-tune further or run with Ultralytics" },
  "final_model.onnx": { label: "ONNX", hint: ".onnx — portable full-precision export" },
  "final_model.quant.onnx": { label: "ONNX int8", hint: ".quant.onnx — smallest, for CPU and edge" },
};

interface Props {
  token: string;
  listing: ListingState;
  goTo: (tab: TabId) => void;
}

export default function ModelsTab({ token, listing, goTo }: Props) {
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const download = async (model: string, artifact: Artifact) => {
    const key = `${model}/${artifact}`;
    setBusy(key);
    setError(null);
    try {
      await downloadArtifact(token, model, artifact);
    } catch (e) {
      setError(`Download of ${model} failed: ${(e as ApiError).message}`);
    } finally {
      setBusy(null);
    }
  };

  const models = listing.data?.models ?? [];

  return (
    <section>
      <div className="section-head">
        <div>
          <h2>Models</h2>
          <p className="lede">Trained detectors in your storage, ready to download.</p>
        </div>
        <button className="btn ghost" onClick={listing.refresh} disabled={!token || listing.loading}>
          {listing.loading ? "Refreshing…" : "Refresh"}
        </button>
      </div>

      {listing.error && <p className="alert error">{listing.error}</p>}
      {error && <p className="alert error">{error}</p>}

      {token && listing.data && models.length === 0 && (
        <div className="empty">
          <p>No trained models yet. Upload a dataset to train your first one.</p>
          <button className="btn" onClick={() => goTo("upload")}>
            Upload a dataset
          </button>
        </div>
      )}

      {models.length > 0 && (
        <ul className="rows">
          {models.map((m) => (
            <li key={m.name} className="row detection">
              <div className="row-main">
                <h3 className="row-title">{m.name}</h3>
                <p className="row-meta">
                  {m.artifacts.length} of 3 formats exported
                  {m.hasDataset ? "" : " · dataset removed"}
                </p>
              </div>
              <div className="artifact-buttons">
                {m.artifacts.map((a) => (
                  <button
                    key={a}
                    className="btn small"
                    title={ARTIFACT_INFO[a].hint}
                    disabled={busy !== null}
                    onClick={() => download(m.name, a)}
                  >
                    {busy === `${m.name}/${a}` ? "Downloading…" : `Download ${ARTIFACT_INFO[a].label}`}
                  </button>
                ))}
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
