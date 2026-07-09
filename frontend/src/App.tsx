import { useState } from "react";
// Import "pages" (components).
import PreciseMode from "./pages/PreciseMode";
import FuzzyMode from "./pages/FuzzyMode";
import MarketMode from "./pages/MarketMode";

type Page = "precise" | "fuzzy" | "market";

export default function App() {
  // React state: which page is currently selected.
  const [page, setPage] = useState<Page>("precise");

  // Helper function that returns a Tailwind class string for tabs.
  // If active === true, we use a dark "selected" style.
  // If active === false, we use a light "unselected" style.
  const tabClass = (active: boolean) =>
    [
      "rounded-full px-4 py-2 text-sm font-medium transition",
      active
        ? "bg-slate-900 text-white shadow-sm"
        : "border border-slate-200 bg-white text-slate-700 hover:bg-slate-50",
    ].join(" ");

  return (
    // The outer page wrapper:
    // - min-h-screen: at least full viewport height
    // - bg-slate-50: light gray background
    // - text-slate-900: default text color
    <div className="min-h-screen bg-slate-50 text-slate-900">
      {/* Top Nav / Header */}
      {/* 
        sticky top-0: the header stays at the top when scrolling
        z-10: keep it above other content
        bg-white/80 backdrop-blur: "glass" effect (semi-transparent + blur)
      */}
      <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/80 backdrop-blur">
        {/* Centered container with fixed max width */}
        <div className="mx-auto flex w-full max-w-5xl items-center justify-between px-6 py-4">
          {/* Left side: logo + title */}
          <div className="flex items-center gap-3">
            {/* Simple logo block */}
            <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-slate-900">
              <svg
                width="22"
                height="22"
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                className="text-white"
              >
                <path
                  d="M12 3V6"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                />
                <rect
                  x="5"
                  y="7"
                  width="14"
                  height="10"
                  rx="3"
                  stroke="currentColor"
                  strokeWidth="1.8"
                />
                <circle cx="9.5" cy="12" r="1" fill="currentColor" />
                <circle cx="14.5" cy="12" r="1" fill="currentColor" />
                <path
                  d="M9 15.5H15"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                />
              </svg>
            </div>

            {/* Title + subtitle */}
            <div>
              <div className="text-sm font-semibold tracking-tight">
                Property Value Prediction System
              </div>
              <div className="text-xs text-slate-500">
                Web Application
              </div>
            </div>
          </div>

          {/* Right side: tab buttons — both address-driven (same model). Chat is
              the default; Search is the form view. */}
          <nav className="flex items-center gap-2">
            <button className={tabClass(page === "precise")} onClick={() => setPage("precise")}>
              Chat
            </button>
            <button className={tabClass(page === "fuzzy")} onClick={() => setPage("fuzzy")}>
              Search
            </button>
            <button className={tabClass(page === "market")} onClick={() => setPage("market")}>
              Market price
            </button>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="mx-auto w-full max-w-6xl px-6 py-8">
        {page === "precise" && <PreciseMode />}
        {page === "fuzzy" && <FuzzyMode />}
        {page === "market" && <MarketMode />}
      </main>
    </div>
  );
}