import { useEffect, useRef, useState } from "react";

// Each message is either from the user or from the agent.
// text is the content shown in the chat bubble.
type Msg = { role: "user" | "agent"; text: string };

// Command is a parsed version of what the user typed.
// It helps keep "parsing" separate from "reply building".
type Command =
  | { kind: "show_steps" }
  | { kind: "show_step"; stepNum: number }
  | { kind: "show_summary" }
  | { kind: "unknown"; raw: string };

export default function AgentPage() {
  // data: JSON loaded from /data/agent/agent_run_sample.json
  // null means "not loaded yet".
  const [data, setData] = useState<any>(null);
  // input: current text inside the input box
  const [input, setInput] = useState("");
  // messages: array of chat messages displayed in the chat history
  const [messages, setMessages] = useState<Msg[]>([]);
  // showDebug: show the raw JSON debug panel
  const [showDebug, setShowDebug] = useState(false);

  // Keep one typing timer at a time
  const typingTimerRef = useRef<number | null>(null);

  // Auto-scroll to bottom when messages change
  const bottomRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Load JSON once when the component mounts (empty dependency array []).
  useEffect(() => {
    fetch("/data/agent/agent_run_sample.json")
      .then((res) => res.json())
      .then((json) => setData(json))
      .catch((err) => console.error("Failed to load JSON:", err));
  }, []);

  // -----------------------------
  // (1) Parse command
  // -----------------------------
  function parseCommand(rawInput: string): Command {
    // Called when user clicks "Send" or presses Enter
    // (We keep parsing separate so handleSend stays small.)

    const cmd = rawInput.trim().toLowerCase();
    if (!cmd) return { kind: "unknown", raw: "" };

    if (cmd === "show steps") {
      return { kind: "show_steps" };
    }

    if (cmd === "show summary") {
      return { kind: "show_summary" };
    }

    if (cmd.startsWith("show step ")) {
      // Parse step number from command
      const stepNum = Number(cmd.replace("show step ", "").trim());
      if (Number.isFinite(stepNum)) {
        return { kind: "show_step", stepNum };
      }
    }

    return { kind: "unknown", raw: cmd };
  }

  // -----------------------------
  // (2) Build reply text
  // -----------------------------
  function buildReply(command: Command): string {
    // Default reply
    let reply = "Unknown command. Try: show steps | show step 5 | show summary";

    if (data === null) {
      return "Data not loaded yet.";
    }

    // 3) Decide reply based on command
    if (command.kind === "show_steps") {
      // Show all steps; join into a big string
      reply = data.steps.map((s: any) => `Step ${s.step}:\n${s.response}`).join("\n\n");
    } else if (command.kind === "show_step") {
      const found = data.steps.find((s: any) => s.step === command.stepNum);
      reply = found ? found.response : `No such step: ${command.stepNum}`;
    } else if (command.kind === "show_summary") {
      // Use summary field if it exists
      reply = data.response ?? "(no summary in json)";
    }

    return reply;
  }

  // -----------------------------
  // (3) Animate output (typewriter)
  // -----------------------------
  function animateOutput(fullText: string) {
    // Typewriter effect: reveal the agent reply character by character.

    // 1) Stop any previous typing animation
    if (typingTimerRef.current !== null) {
      window.clearInterval(typingTimerRef.current);
      typingTimerRef.current = null;
    }

    // 2) Add a new "empty" agent message first.
    //    We'll fill its text gradually in the interval callback.
    let agentIndex = -1;
    setMessages((prev) => {
      agentIndex = prev.length;
      return [...prev, { role: "agent", text: "" }];
    });

    // 3) Start the interval: update the agent message one character at a time
    let i = 0;
    typingTimerRef.current = window.setInterval(() => {
      i++;

      // Update messages state using functional setState
      setMessages((prev) => {
        // Safety check: if index is invalid, do nothing
        if (agentIndex < 0 || agentIndex >= prev.length) return prev;
        // Copy the array (React state must be immutable)
        const next = [...prev];
        // Replace the agent message with updated text
        next[agentIndex] = { ...next[agentIndex], text: fullText.slice(0, i) };
        return next;
      });

      // Stop typing when finished
      if (i >= fullText.length && typingTimerRef.current !== null) {
        window.clearInterval(typingTimerRef.current);
        typingTimerRef.current = null;
      }
    }, 12); // interval speed (ms). Smaller = faster typing.
  }

  // -----------------------------
  // Orchestrator: push user msg -> parse -> build -> animate
  // -----------------------------
  function handleSend() {
    const raw = input;
    const cmd = raw.trim().toLowerCase();
    if (!cmd) return;

    // 1) Push user's message into chat
    setMessages((prev) => [...prev, { role: "user", text: raw }]);

    // 2) Clear input box
    setInput("");

    // parse command
    const command = parseCommand(raw);

    // build reply
    const reply = buildReply(command);

    // 4) Display agent reply using typewriter effect
    animateOutput(reply);
  }

  return (
    // space-y-6: vertical spacing between sections
    <div className="space-y-6">
      {/* Page header section */}
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">Agent Demo</h1>
        <p className="mt-1 text-sm text-slate-500">Simulated chat UI for agent outputs</p>
      </div>

      {/* Controls row: debug toggle + load status */}
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

      {/* Debug panel: only render when showDebug is true */}
      {showDebug && (
        <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="mb-2 text-sm font-semibold text-slate-800">Loaded JSON (raw)</div>
          {/* pre: keeps formatting, scrollable */}
          <pre className="max-h-72 overflow-auto rounded-xl bg-slate-50 p-3 text-xs text-slate-700">
            {data === null ? "Loading..." : JSON.stringify(data, null, 2)}
          </pre>
        </div>
      )}

      {/* Chat card container */}
      <div className="rounded-2xl border border-slate-200 bg-white shadow-sm">
        {/* Chat card header */}
        <div className="border-b border-slate-200 px-4 py-3">
          <div className="text-sm font-semibold">Simulated Chat</div>
          <div className="text-xs text-slate-500">Try: show steps | show step 5 | show summary</div>
        </div>

        {/* Messages area (scrollable) */}
        <div className="h-[420px] w-full min-w-0 overflow-y-auto overflow-x-hidden bg-slate-50 px-4 py-4">
          <div className="flex flex-col gap-3">
            {/* Render each message bubble */}
            {messages.map((m, idx) => (
              <div
                key={idx}
                className={[
                  // Layout constraints:
                  // - max width is 78% of chat area
                  // - preserve line breaks and wrap long words
                  "min-w-0 max-w-[78%] whitespace-pre-wrap [overflow-wrap:anywhere]",

                  // Base bubble style
                  "rounded-2xl border px-4 py-3 text-sm leading-relaxed shadow-sm",

                  // Different styles depending on role:
                  // user -> right side (ml-auto)
                  // agent -> left side (mr-auto)
                  m.role === "user"
                    ? "ml-auto border-blue-200 bg-blue-50 text-slate-900"
                    : "mr-auto border-slate-200 bg-white text-slate-900",
                ].join(" ")}
              >
                {m.text}
              </div>
            ))}
            {/* Invisible element for auto-scroll target */}
            <div ref={bottomRef} />
          </div>
        </div>

        {/* Input area */}
        <div className="flex items-center gap-2 border-t border-slate-200 p-3">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder='Try: "show summary"'
            className="h-11 w-full min-w-0 flex-1 rounded-xl border border-slate-200 bg-white px-3 text-sm outline-none focus:border-slate-300 focus:ring-2 focus:ring-slate-200"
            onKeyDown={(e) => {
              // Press Enter to send
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