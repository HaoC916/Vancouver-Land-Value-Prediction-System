import { useState } from "react";
import AgentPage from "./pages/AgentPage";
import HumanPage from "./pages/HumanPage";
import AnalysisPage from "./pages/AnalysisPage";

export default function App() {
  const [page, setPage] = useState<"agent" | "human" | "analysis">("agent");

  return (
    <div style={{ minHeight: "100vh", fontFamily: "system-ui, sans-serif", background: "#fff" }}>
      {/* Top Nav */}
      <div style={{ position: "sticky", top: 0, zIndex: 10, background: "white", borderBottom: "1px solid #eee" }}>
        <div
          style={{
            maxWidth: 1100,
            margin: "0 auto",
            padding: "16px 24px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            boxSizing: "border-box",
          }}
        >
          <div style={{ fontWeight: 800 }}>CMPT733 Final Project</div>

          <div style={{ display: "flex", gap: 8 }}>
            <button onClick={() => setPage("agent")}>Agent</button>
            <button onClick={() => setPage("human")}>Human</button>
            <button onClick={() => setPage("analysis")}>Analysis</button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div style={{ maxWidth: 1100, margin: "0 auto", padding: 24, boxSizing: "border-box" }}>
        {page === "agent" && <AgentPage />}
        {page === "human" && <HumanPage />}
        {page === "analysis" && <AnalysisPage />}
      </div>
    </div>
  );
}