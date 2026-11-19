# Frontend

React + TypeScript (Vite) UI for the training API in `../api`.

| Tab | What it does |
| --- | --- |
| **Models** | Lists trained models and downloads `.pt`, `.onnx` or int8 `.quant.onnx` files. |
| **Datasets** | Lists the dataset ZIPs stored for your account and whether each one produced a model. |
| **Upload** | Checks a dataset ZIP in the browser (`data.yaml`, `images/`, `labels/`) and stages it for training. |
| **Training** | Sets the model name, base architecture, epochs and batch size, uploads the staged ZIP and starts training, then tracks job status. |

## Run it

```bash
# terminal 1: the API (see ../README.md for the required env vars)
cd api && uvicorn main:app --reload

# terminal 2: the frontend
cd frontend
npm install
npm run dev          # http://localhost:5173
```

Paste an API token from `API_TOKENS` into the token field. It's kept in the
browser's localStorage.

The dev server proxies `/api/*` to `http://localhost:8000` because the API has
no CORS middleware. To point the proxy somewhere else, set `API_PROXY_TARGET`.
For a deployed build, set `VITE_API_BASE` to the API's URL at build time
(`VITE_API_BASE=https://api.example.com npm run build`) and either serve both
from the same origin or add CORS to the API.

## Limitations from the current API

- **Jobs are tracked per browser.** The API has no "list my jobs" endpoint, so
  the Training tab remembers job IDs it submitted in localStorage and polls
  `/job_status` every 15 seconds. Jobs started elsewhere don't show up there,
  but their models still appear in Models.
- **Datasets can't be reused.** `/train_yolo` takes the ZIP and the model name
  together, so each training run uploads its dataset again.
- **Downloads are buffered.** Model files are fetched with the auth header and
  saved through a blob URL, so a large artifact is held in memory first.
