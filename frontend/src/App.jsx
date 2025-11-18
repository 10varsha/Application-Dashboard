import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import Navbar from "./components/Navbar.jsx";
import Assessments from "./pages/Assessments.jsx";
import AssessmentsResponse from "./pages/AssessmentsResponse.jsx";
import PendingAssessments from "./pages/PendingAssessments.jsx";

export default function App() {
  return (
    <BrowserRouter>
      <Navbar />

      <main
        className="
          w-full flex justify-center pt-28 pb-20
          bg-[var(--bg-main)] text-[var(--text-main)]
          transition-colors duration-300
        "
      >
        <div className="w-full max-w-[1600px] mx-auto px-6">

          {/* TOP NAV TABS */}
          <div className="flex flex-wrap gap-4 mb-12 justify-center">
            {[
              { path: "/", label: "Assessments" },
              { path: "/responses", label: "Assessments Response" },
              { path: "/pending", label: "Pending Assessments" },
            ].map((tab) => (
              <NavLink
                key={tab.path}
                to={tab.path}
                end={tab.path === "/"}
                className={({ isActive }) =>
                  `
                  px-5 py-2 rounded-xl text-sm font-medium transition 
                  border border-[var(--border)]
                  ${
                    isActive
                      ? "bg-[var(--tab-active)] text-[var(--text-main)]"
                      : "bg-[var(--tab)] text-[var(--text-muted)] hover:opacity-75"
                  }
                `
                }
              >
                {tab.label}
              </NavLink>
            ))}
          </div>

          {/* PAGE CONTENT */}
          <Routes>
            <Route path="/" element={<Assessments />} />
            <Route path="/responses" element={<AssessmentsResponse />} />
            <Route path="/pending" element={<PendingAssessments />} />
          </Routes>
        </div>
      </main>
    </BrowserRouter>
  );
}
