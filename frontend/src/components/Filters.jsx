import { useState } from "react";

export default function Filters({ filters, setFilters, filterOptions }) {
  const [dateRanges, setDateRanges] = useState({});

  const handleSelect = (key, value) => {
    setFilters((prev) => ({ ...prev, [key]: value ? [value] : [] }));
  };

  const handleDateChange = (key, index, newValue) => {
    const updated = [...(dateRanges[key] || ["", ""])];
    updated[index] = newValue;
    setDateRanges((prev) => ({ ...prev, [key]: updated }));

    if (updated[0] && updated[1]) {
      setFilters((prev) => ({ ...prev, [key]: updated }));
    } else {
      setFilters((prev) => ({ ...prev, [key]: [] }));
    }
  };

  return (
    <div
      className="
        w-full p-6 rounded-xl shadow
        bg-[var(--card-bg)]
        border border-[var(--border)]
        text-[var(--text-main)]
      "
    >
      <h2 className="font-semibold text-lg mb-6">Filters</h2>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        {Object.keys(filterOptions).map((key) => {
          const label = key.replace(/_/g, " ").toUpperCase();

          const isDate = key === "deadline" || key === "email_datetime_est";

          if (isDate) {
            return (
              <div key={key} className="flex flex-col gap-2">
                <label className="text-sm font-medium text-[var(--text-muted)]">
                  {label}
                </label>

                <div className="flex gap-3">
                  <input
                    type="date"
                    className="
                      w-full px-3 py-2 rounded-lg
                      bg-[var(--input-bg)] text-[var(--text-main)]
                      border border-[var(--border)]
                      focus:ring-2 focus:ring-indigo-500
                    "
                    onChange={(e) => handleDateChange(key, 0, e.target.value)}
                  />

                  <input
                    type="date"
                    className="
                      w-full px-3 py-2 rounded-lg
                      bg-[var(--input-bg)] text-[var(--text-main)]
                      border border-[var(--border)]
                      focus:ring-2 focus:ring-indigo-500
                    "
                    onChange={(e) => handleDateChange(key, 1, e.target.value)}
                  />
                </div>
              </div>
            );
          }

          return (
            <div key={key} className="flex flex-col gap-2">
              <label className="text-sm font-medium text-[var(--text-muted)]">
                {label}
              </label>

              <select
                className="
                  w-full px-3 py-2 rounded-lg
                  bg-[var(--input-bg)] text-[var(--text-main)]
                  border border-[var(--border)]
                  focus:ring-2 focus:ring-indigo-500
                "
                onChange={(e) => handleSelect(key, e.target.value)}
              >
                <option value="">Choose</option>

                {filterOptions[key].map((opt) => (
                  <option key={opt} value={String(opt)}>
                    {String(opt)}
                  </option>
                ))}
              </select>
            </div>
          );
        })}
      </div>
    </div>
  );
}
