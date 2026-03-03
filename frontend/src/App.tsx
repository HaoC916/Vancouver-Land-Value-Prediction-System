import { useState } from "react";
import AgentPage from "./pages/AgentPage";
import HumanPage from "./pages/HumanPage";
import AnalysisPage from "./pages/AnalysisPage";

type Page = "agent" | "human" | "analysis";

export default function App() {
  const [page, setPage] = useState<Page>("agent");

  const tabClass = (active: boolean) =>
    [
      "rounded-full px-4 py-2 text-sm font-medium transition",
      active
        ? "bg-slate-900 text-white shadow-sm"
        : "border border-slate-200 bg-white text-slate-700 hover:bg-slate-50",
    ].join(" ");

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      {/* Top Nav */}
      <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/80 backdrop-blur">
        <div className="mx-auto flex w-full max-w-5xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="h-8 w-8 rounded-xl bg-slate-900" />
            <div>
              <div className="text-sm font-semibold tracking-tight">
                CMPT733 Final Project
              </div>
              <div className="text-xs text-slate-500">
                Housing Price Demo
              </div>
            </div>
          </div>

          <nav className="flex items-center gap-2">
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