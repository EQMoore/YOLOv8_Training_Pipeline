// Browser-local persistence. The API has no "list my jobs" endpoint, so jobs
// submitted from this browser are remembered here and polled by id.

import { useEffect, useState } from "react";

export interface TrackedJob {
  job_id: string;
  model: string;
  arch: string;
  epochs: number;
  batch: number;
  submittedAt: string;
  state?: string;
  error?: string;
}

function read<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function write(key: string, value: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage unavailable (private mode, quota) */
  }
}

export function usePersistentState<T>(key: string, initial: T) {
  const [value, setValue] = useState<T>(() => read(key, initial));
  useEffect(() => write(key, value), [key, value]);
  return [value, setValue] as const;
}
