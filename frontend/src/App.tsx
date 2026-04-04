import { useState } from "react";
// Import the three "pages" (components).
import ChatEstimate from "./pages/ChatEstimate";
import ManualEstimate from "./pages/ManualEstimate";

// A TypeScript union type: page can only be one of these three strings.
// This prevents typos like "agnet" from compiling.
type Page = "chat" | "manual";

export default function App() {
  // React state: which page is currently selected.
  const [page, setPage] = useState<Page>("chat");

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
            <div className="h-8 w-8 rounded-xl bg-slate-900" />

            {/* Title + subtitle */}
            <div>
              <div className="text-sm font-semibold tracking-tight">
                Vancouver Land Value Estimator
              </div>
              <div className="text-xs text-slate-500">
                Web Application
              </div>
            </div>
          </div>

          {/* Right side: tab buttons */}
          <nav className="flex items-center gap-2">
            {/* 
              On click -> setPage("agent")
              tabClass(page === "agent") decides the active style
            */}
            <button className={tabClass(page === "chat")} onClick={() => setPage("chat")}>
              Chat Estimate
            </button>
            <button className={tabClass(page === "manual")} onClick={() => setPage("manual")}>
              Manual Estimate
            </button>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="mx-auto w-full max-w-6xl px-6 py-8">
        {page === "chat" && <ChatEstimate />}
        {page === "manual" && <ManualEstimate />}
      </main>
    </div>
  );
}