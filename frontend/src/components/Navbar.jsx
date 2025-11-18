import ThemeToggle from "./ThemeToggle.jsx";

export default function Navbar() {
  return (
    <nav
      className="
        w-full fixed top-0 left-0 z-50
        border-b
        bg-[var(--bg-main)] text-[var(--text-main)]
        border-[var(--border)]
        shadow-sm
      "
    >
      <div className="w-full max-w-[1900px] mx-auto flex justify-between items-center px-6 py-4">
        <h1 className="text-2xl font-semibold tracking-tight">
          Assessments Dashboard
        </h1>
        <ThemeToggle />
      </div>
    </nav>
  );
}
