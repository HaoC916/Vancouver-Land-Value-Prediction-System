import { useEffect, useRef, useState } from "react";

type Msg = { role: "user" | "agent"; text: string };

export default function AgentPage() {
  const [data, setData] = useState<any>(null);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Msg[]>([]);
  const [showDebug, setShowDebug] = useState(false);

  // Keep one typing timer at a time
  const typingTimerRef = useRef<number | null>(null);

  // Auto-scroll to bottom when messages change
  const bottomRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    fetch("/data/agent/agent_run_sample.json")
      .then((res) => res.json())
      .then((json) => setData(json))
      .catch((err) => console.error("Failed to load JSON:", err));
  }, []);

  function typeReply(fullText: string) {
    // Stop previous typing
    if (typingTimerRef.current !== null) {
      window.clearInterval(typingTimerRef.current);
      typingTimerRef.current = null;
    }

    let agentIndex = -1;
    setMessages((prev) => {
      agentIndex = prev.length;
      return [...prev, { role: "agent", text: "" }];
    });

    let i = 0;
    typingTimerRef.current = window.setInterval(() => {
      i++;

      setMessages((prev) => {
        if (agentIndex < 0 || agentIndex >= prev.length) return prev;
        const next = [...prev];
        next[agentIndex] = { ...next[agentIndex], text: fullText.slice(0, i) };
        return next;
      });

      if (i >= fullText.length && typingTimerRef.current !== null) {
        window.clearInterval(typingTimerRef.current);
        typingTimerRef.current = null;
      }
    }, 12);
  }

  function handleSend() {
    const raw = input;
    const cmd = raw.trim().toLowerCase();
    if (!cmd) return;

    // push user message
    setMessages((prev) => [...prev, { role: "user", text: raw }]);
    setInput("");

    if (data === null) {
      typeReply("Data not loaded yet.");
      return;
    }

    let reply = "Unknown command. Try: show steps | show step 5 | show summary";

    if (cmd === "show steps") {
      reply = data.steps.map((s: any) => `Step ${s.step}:\n${s.response}`).join("\n\n");
    } else if (cmd.startsWith("show step ")) {
      const stepNum = Number(cmd.replace("show step ", ""));
      const found = data.steps.find((s: any) => s.step === stepNum);
      reply = found ? found.response : `No such step: ${stepNum}`;
    } else if (cmd === "show summary") {
      reply = data.response ?? "(no summary in json)";
    }

    typeReply(reply);
  }

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">Agent Demo</h1>
        <p className="mt-1 text-sm text-slate-500">
          Simulated chat UI for agent outputs
        </p>
      </div>

      {/* Controls */}
      <div className="flex items-center justify-between">
        <button
          onClick={() => setShowDebug((v) => !v)}
          className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 shadow-sm hover:bg-slate-50"
        >
          {showDebug ? "Hide Debug" : "Show Debug"}
        </button>
        <div className="text-xs text-slate-500">
          Data: <span className="font-medium">{data ? "Loaded" : "Loading..."}</span>
        </div>
      </div>

      {/* Debug */}
      {showDebug && (
        <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="mb-2 text-sm font-semibold text-slate-800">Loaded JSON (raw)</div>
          <pre className="max-h-72 overflow-auto rounded-xl bg-slate-50 p-3 text-xs text-slate-700">
            {data === null ? "Loading..." : JSON.stringify(data, null, 2)}
          </pre>
        </div>
      )}

      {/* Chat card */}
      <div className="rounded-2xl border border-slate-200 bg-white shadow-sm">
        <div className="border-b border-slate-200 px-4 py-3">
          <div className="text-sm font-semibold">Simulated Chat</div>
          <div className="text-xs text-slate-500">Try: show steps | show step 5 | show summary</div>
        </div>

        {/* Messages */}
        <div className="h-[420px] w-full min-w-0 overflow-y-auto overflow-x-hidden bg-slate-50 px-4 py-4">
          <div className="flex flex-col gap-3">
            {messages.map((m, idx) => (
              <div
                key={idx}
                className={[
                  "min-w-0 max-w-[78%] whitespace-pre-wrap break-words [overflow-wrap:anywhere]",
                  "rounded-2xl border px-4 py-3 text-sm leading-relaxed shadow-sm",
                  m.role === "user"
                    ? "ml-auto border-blue-200 bg-blue-50 text-slate-900"
                    : "mr-auto border-slate-200 bg-white text-slate-900",
                ].join(" ")}
              >
                {m.text}
              </div>
            ))}
            <div ref={bottomRef} />
          </div>
        </div>

        {/* Input */}
        <div className="flex items-center gap-2 border-t border-slate-200 p-3">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder='Try: "show summary"'
            className="h-11 w-full min-w-0 flex-1 rounded-xl border border-slate-200 bg-white px-3 text-sm outline-none focus:border-slate-300 focus:ring-2 focus:ring-slate-200"
            onKeyDown={(e) => {
              if (e.key === "Enter") handleSend();
            }}
          />
          <button
            onClick={handleSend}
            className="h-11 shrink-0 rounded-xl bg-slate-900 px-4 text-sm font-semibold text-white hover:bg-slate-800"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}