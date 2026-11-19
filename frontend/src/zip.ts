// Reads the file list from a ZIP's central directory without unpacking it,
// so a dataset can be checked in the browser before it is uploaded.

const EOCD_SIG = 0x06054b50;
const CDIR_SIG = 0x02014b50;

async function readBytes(file: File, start: number, end: number): Promise<DataView> {
  const buf = await file.slice(start, end).arrayBuffer();
  return new DataView(buf);
}

export async function listZipEntries(file: File): Promise<string[]> {
  // End-of-central-directory record: 22 bytes plus an optional comment (<64 KiB).
  const tailStart = Math.max(0, file.size - (22 + 0xffff));
  const tail = await readBytes(file, tailStart, file.size);
  let eocd = -1;
  for (let i = tail.byteLength - 22; i >= 0; i--) {
    if (tail.getUint32(i, true) === EOCD_SIG) {
      eocd = i;
      break;
    }
  }
  if (eocd < 0) throw new Error("not a zip");

  const count = tail.getUint16(eocd + 10, true);
  const size = tail.getUint32(eocd + 12, true);
  const offset = tail.getUint32(eocd + 16, true);
  if (offset === 0xffffffff || count === 0xffff) throw new Error("zip64 not supported");

  const dir = await readBytes(file, offset, offset + size);
  const decoder = new TextDecoder();
  const names: string[] = [];
  let p = 0;
  for (let i = 0; i < count && p + 46 <= dir.byteLength; i++) {
    if (dir.getUint32(p, true) !== CDIR_SIG) throw new Error("corrupt central directory");
    const nameLen = dir.getUint16(p + 28, true);
    const extraLen = dir.getUint16(p + 30, true);
    const commentLen = dir.getUint16(p + 32, true);
    const nameBytes = new Uint8Array(dir.buffer, dir.byteOffset + p + 46, nameLen);
    names.push(decoder.decode(nameBytes));
    p += 46 + nameLen + extraLen + commentLen;
  }
  return names;
}

// Matches trainer_image/train.py, which searches the extracted tree for these.
const DATA_YAML_NAMES = new Set(["data.yaml", "data.yml", "dataset.yaml"]);
const IMAGE_EXT = /\.(jpe?g|png|bmp|webp|tiff?)$/i;

export interface DatasetCheck {
  // False when the zip couldn't be read; the trainer will still check it.
  readable: boolean;
  dataYaml: string | null;
  images: number;
  labels: number;
  files: number;
  problems: string[];
  warnings: string[];
}

export async function checkDataset(file: File): Promise<DatasetCheck> {
  const result: DatasetCheck = {
    readable: true,
    dataYaml: null,
    images: 0,
    labels: 0,
    files: 0,
    problems: [],
    warnings: [],
  };

  if (!file.name.toLowerCase().endsWith(".zip")) {
    result.problems.push("Choose a .zip file.");
    return result;
  }

  let entries: string[];
  try {
    entries = await listZipEntries(file);
  } catch {
    result.readable = false;
    result.warnings.push(
      "Couldn't read this ZIP's contents in the browser. It can still be uploaded, and the trainer will check it.",
    );
    return result;
  }

  for (const entry of entries) {
    if (entry.endsWith("/")) continue;
    result.files++;
    const lower = entry.toLowerCase();
    const base = lower.split("/").pop() ?? "";
    if (DATA_YAML_NAMES.has(base) && !result.dataYaml) result.dataYaml = entry;
    if (IMAGE_EXT.test(lower) && /(^|\/)images\//.test(lower)) result.images++;
    if (lower.endsWith(".txt") && /(^|\/)labels\//.test(lower)) result.labels++;
  }

  if (!result.dataYaml) {
    result.problems.push(
      "No data.yaml found. Training needs an Ultralytics data.yaml (or data.yml / dataset.yaml) somewhere in the ZIP.",
    );
  }
  if (result.images === 0) {
    result.warnings.push("No images found inside an images/ folder.");
  }
  if (result.labels === 0) {
    result.warnings.push("No label .txt files found inside a labels/ folder.");
  }
  return result;
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  const units = ["KB", "MB", "GB"];
  let v = n / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v < 10 ? 1 : 0)} ${units[i]}`;
}
