import type { ListingState, TabId } from "../App";

interface Props {
  listing: ListingState;
  goTo: (tab: TabId) => void;
}

export default function DatasetsTab({ listing, goTo }: Props) {
  const datasets = listing.data?.datasets ?? [];

  return (
    <section>
      <div className="section-head">
        <div>
          <h2>Datasets</h2>
          <p className="lede">
            Every training run stores its dataset ZIP under the model's name. To train on new
            data, upload it as a new dataset.
          </p>
        </div>
        <button className="btn ghost" onClick={listing.refresh} disabled={listing.loading}>
          {listing.loading ? "Refreshing…" : "Refresh"}
        </button>
      </div>

      {listing.error && <p className="alert error">{listing.error}</p>}

      {listing.data && datasets.length === 0 && (
        <div className="empty">
          <p>No datasets uploaded yet.</p>
          <button className="btn" onClick={() => goTo("upload")}>
            Upload a dataset
          </button>
        </div>
      )}

      {datasets.length > 0 && (
        <table className="table">
          <thead>
            <tr>
              <th scope="col">Name</th>
              <th scope="col">Stored at</th>
              <th scope="col">Model</th>
            </tr>
          </thead>
          <tbody>
            {datasets.map((d) => (
              <tr key={d.path}>
                <td className="strong">{d.name}</td>
                <td className="mono">{d.path}</td>
                <td>
                  {d.hasModel ? (
                    <button className="link" onClick={() => goTo("models")}>
                      Trained
                    </button>
                  ) : (
                    <button className="link muted" onClick={() => goTo("training")}>
                      Training or failed
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
