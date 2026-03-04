import { useState } from "react";
// Import the three "pages" (components).
import AgentPage from "./pages/AgentPage";
import HumanPage from "./pages/HumanPage";
import AnalysisPage from "./pages/AnalysisPage";

// A TypeScript union type: page can only be one of these three strings.
// This prevents typos like "agnet" from compiling.
type Page = "agent" | "human" | "analysis";

export default function App() {
  // React state: which page is currently selected.
  // Default is "agent" --> AgentPage is shown first.
  const [page, setPage] = useState<Page>("agent");

  // Helper function that returns a Tailwind class string for tabs.
  // If active === true, we use a dark "selected" style.
  // If active === false, we use a light "unselected" style.
  const tabClass = (active: boolean) =>
    [
      // Base style for ALL tabs (shared styles)
      "rounded-full px-4 py-2 text-sm font-medium transition",
      // Conditional style depending on whether the tab is active
      active
        ? "bg-slate-900 text-white shadow-sm"
        : "border border-slate-200 bg-white text-slate-700 hover:bg-slate-50",
    ].join(" "); // join array into a single string

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
            <div className="h-8 w-8 rounded-xl bg-slate-900" />

            {/* Title + subtitle */}
            <div>
              <div className="text-sm font-semibold tracking-tight">
                CMPT733 Final Project
              </div>
              <div className="text-xs text-slate-500">
                Housing Price Demo
              </div>
            </div>
          </div>

          {/* Right side: tab buttons */}
          <nav className="flex items-center gap-2">
            {/* 
              On click -> setPage("agent")
              tabClass(page === "agent") decides the active style
            */}
            <button className={tabClass(page === "agent")} onClick={() => setPage("agent")}>
              Agent
            </button>
            <button className={tabClass(page === "human")} onClick={() => setPage("human")}>
              Human
            </button>
            <button className={tabClass(page === "analysis")} onClick={() => setPage("analysis")}>
              Analysis
            </button>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="mx-auto w-full max-w-5xl px-6 py-8">
        {page === "agent" && <AgentPage />}
        {page === "human" && <HumanPage />}
        {page === "analysis" && <AnalysisPage />}
      </main>
    </div>
  );
}