import { useCallback, useEffect, useRef, useState, type KeyboardEvent } from "react";
import { ApiError, BucketListing, listBucket } from "./api";
import { TrackedJob, usePersistentState } from "./storage";
import { DatasetCheck } from "./zip";
import ModelsTab from "./tabs/ModelsTab";
import DatasetsTab from "./tabs/DatasetsTab";
import UploadTab from "./tabs/UploadTab";
import TrainingTab from "./tabs/TrainingTab";

export type TabId = "models" | "datasets" | "upload" | "training";

const TABS: { id: TabId; label: string }[] = [
  { id: "models", label: "Models" },
  { id: "datasets", label: "Datasets" },
  { id: "upload", label: "Upload" },
  { id: "training", label: "Training" },
];

export interface StagedDataset {
  file: File;
  check: DatasetCheck;
}

export interface ListingState {
  data: BucketListing | null;
  loading: boolean;
  error: string | null;
  refresh: () => void;
}

export default function App() {
  const [token, setToken] = usePersistentState("yolo.token", "");
  const [tab, setTab] = usePersistentState<TabId>("yolo.tab", "models");
  const [jobs, setJobs] = usePersistentState<TrackedJob[]>("yolo.jobs", []);
  const [staged, setStaged] = useState<StagedDataset | null>(null);

  const [listing, setListing] = useState<BucketListing | null>(null);
  const [listingLoading, setListingLoading] = useState(false);
  const [listingError, setListingError] = useState<string | null>(null);
  const requestId = useRef(0);

  const refresh = useCallback(() => {
    if (!token) {
      setListing(null);
      setListingError(null);
      return;
    }
    const id = ++requestId.current;
    setListingLoading(true);
    setListingError(null);
    listBucket(token)
      .then((data) => id === requestId.current && setListing(data))
      .catch((e: ApiError) => {
        if (id !== requestId.current) return;
        setListing(null);
        setListingError(e.message);
      })
      .finally(() => id === requestId.current && setListingLoading(false));
  }, [token]);

  useEffect(refresh, [refresh]);

  const listingState: ListingState = {
    data: listing,
    loading: listingLoading,
    error: listingError,
    refresh,
  };

  const tabRefs = useRef<Record<string, HTMLButtonElement | null>>({});
  const onTabKey = (e: KeyboardEvent, index: number) => {
    const delta = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
    if (!delta) return;
    e.preventDefault();
    const next = TABS[(index + delta + TABS.length) % TABS.length];
    setTab(next.id);
    tabRefs.current[next.id]?.focus();
  };

  const activeJobs = jobs.filter((j) => !j.state || !j.state.match(/SUCCEEDED|FAILED|CANCELLED/)).length;

  return (
    <div className="shell">
      <header className="masthead">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <h1>YOLO Trainer</h1>
        </div>
        <label className="token">
          <span>API token</span>
          <input
            type="password"
            value={token}
            placeholder="tok_…"
            autoComplete="off"
            spellCheck={false}
            onChange={(e) => setToken(e.target.value.trim())}
          />
        </label>
      </header>

      <nav className="tabs" role="tablist" aria-label="Sections">
        {TABS.map((t, i) => (
          <button
            key={t.id}
            ref={(el) => (tabRefs.current[t.id] = el)}
            role="tab"
            id={`tab-${t.id}`}
            aria-selected={tab === t.id}
            aria-controls={`panel-${t.id}`}
            tabIndex={tab === t.id ? 0 : -1}
            className="tab"
            onClick={() => setTab(t.id)}
            onKeyDown={(e) => onTabKey(e, i)}
          >
            {t.label}
            {t.id === "upload" && staged && <span className="tab-badge">1 staged</span>}
            {t.id === "training" && activeJobs > 0 && (
              <span className="tab-badge">{activeJobs} running</span>
            )}
          </button>
        ))}
      </nav>

      <main
        className="panel"
        role="tabpanel"
        id={`panel-${tab}`}
        aria-labelledby={`tab-${tab}`}
      >
        {!token && (
          <p className="notice">
            Paste your API token above to load your models and start training. Tokens are issued
            by whoever runs this service.
          </p>
        )}
        {tab === "models" && <ModelsTab token={token} listing={listingState} goTo={setTab} />}
        {tab === "datasets" && <DatasetsTab listing={listingState} goTo={setTab} />}
        {tab === "upload" && <UploadTab staged={staged} setStaged={setStaged} goTo={setTab} />}
        {tab === "training" && (
          <TrainingTab
            token={token}
            staged={staged}
            clearStaged={() => setStaged(null)}
            jobs={jobs}
            setJobs={setJobs}
            onJobFinished={refresh}
            goTo={setTab}
          />
        )}
      </main>
    </div>
  );
}
