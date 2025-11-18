import axios from "axios";
import { useEffect, useState } from "react";
import Filters from "../components/Filters.jsx";
import DataTable from "../components/DataTable.jsx";

export default function Assessments() {
  const [data, setData] = useState([]);
  const [filters, setFilters] = useState({});
  const [filterOptions, setFilterOptions] = useState({});
  const [loading, setLoading] = useState(true);   // ⬅️ NEW

  useEffect(() => {
    const fetchAssessments = async () => {
      try {
        const res = await axios.get("http://localhost:8000/assessments");
        const rows = res.data || [];
        setData(rows);

        const options = {};
        rows.forEach((row) => {
          Object.keys(row).forEach((k) => {
            if (!options[k]) options[k] = new Set();
            options[k].add(row[k]);
          });
        });
        Object.keys(options).forEach((k) => {
          options[k] = Array.from(options[k]);
        });
        setFilterOptions(options);
      } catch (err) {
        console.error("Error fetching assessments:", err);
      } finally {
        setLoading(false);      // ⬅️ stop loading
      }
    };

    fetchAssessments();
  }, []);

  const applyFilters = async () => {
    try {
      const res = await axios.post("http://localhost:8000/filter-assessments", {
        filters,
      });
      setData(res.data || []);
    } catch (err) {
      console.error("Error filtering assessments:", err);
    }
  };

  return (
    <div className="w-full h-full flex flex-col">
      <h2 className="text-2xl font-semibold text-center mb-8">
        Assessments
      </h2>

      <div className="w-full mb-8">
        <Filters
          filters={filters}
          setFilters={setFilters}
          filterOptions={filterOptions}
        />
      </div>

      <div className="flex justify-center mb-8">
        <button
          onClick={applyFilters}
          className="px-6 py-2 rounded-xl bg-indigo-600 text-white font-medium text-sm hover:bg-indigo-700 transition shadow active:scale-95"
        >
          Apply Filters
        </button>
      </div>

      {/* SAME loading design as other pages */}
      <div className="flex-1 flex">
        {loading ? (
          <div className="w-full flex-1 flex justify-center items-center">
            <div className="animate-spin h-12 w-12 rounded-full border-4 border-indigo-500 border-t-transparent"></div>
          </div>
        ) : (
          <DataTable data={data} />
        )}
      </div>
    </div>
  );
}
