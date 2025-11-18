export default function DataTable({ data }) {
  if (!data || data.length === 0) {
    return (
      <div className="w-full flex justify-center py-20 text-[var(--text-muted)]">
        No records found.
      </div>
    );
  }

  const headers = Object.keys(data[0]);

  return (
    <div
      className="
        overflow-x-auto rounded-xl shadow
        bg-[var(--card-bg)]
        border border-[var(--border)]
      "
    >
      <table className="w-full text-sm table-fixed">
        <thead className="bg-[var(--tab)]">
          <tr className="border-b border-[var(--border)]">
            {headers.map((h) => (
              <th
                key={h}
                className="
                  px-4 py-3 text-left text-sm font-semibold
                  text-[var(--text-main)]
                "
              >
                {h.replace(/_/g, " ").toUpperCase()}
              </th>
            ))}
          </tr>
        </thead>

        <tbody>
          {data.map((row, i) => (
            <tr key={i} className="border-b border-[var(--border)]">
              {headers.map((h) => {
                let classes =
                  "px-4 py-3 align-top whitespace-normal break-words";

                if (h === "feedback") classes += " min-w-[350px] max-w-[600px]";
                else if (h === "end_client") classes += " min-w-[120px]";
                else if (h === "sender") classes += " min-w-[150px]";
                else classes += " min-w-[150px]";

                return (
                  <td key={h} className={classes}>
                    {row[h] ?? ""}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
