import { useEffect, useState } from "react";

type Msg = { role: "user" | "agent"; text: string };

export default function AgentPage() {
  // initial state for the data text, which will be updated when the data is loaded
  const [data, setData] = useState<any>(null);
  // state for the user input in the chat box
  const [input, setInput] = useState<string>("");
  // state for the chat messages
  const [messages, setMessages] = useState<{ role: "user" | "agent"; text: string }[]>([]);
  // debug switch: show/hide raw JSON and steps list
  const [showDebug, setShowDebug] = useState<boolean>(false);
  useEffect(() => {
    // fetch data from the data agent folder
    fetch("/data/agent/agent_run_sample.json")
      .then((res) => res.json())
      .then((json) => setData(json))
      .catch((err) => console.error("Failed to load JSON:", err));
  }, []); // empty dependency array means this effect runs once on mount

   // Typewriter effect: show agent reply letter by letter
  function typeReply(fullText: string) {

    // 1) Add an empty agent message first, then update it gradually
    let agentIndex = -1;

    setMessages((prev) => {
      agentIndex = prev.length;
      return [...prev, { role: "agent" as const, text: "" }];
    });

    // 2) Append characters over time
    let i = 0;
    const timer = setInterval(() => {
      i++;

      setMessages((prev) => {
        // safety check
        if (agentIndex < 0 || agentIndex >= prev.length) return prev;

        const next = [...prev];
        const current = next[agentIndex];

        next[agentIndex] = { ...current, text: fullText.slice(0, i) };
        return next;
      });

      if (i >= fullText.length) {
        clearInterval(timer);
      }
    }, 15); // change to 25/40 to slow down
  }

  function handleSend() {
    const cmd = input.trim().toLowerCase();
    if (!cmd) return;

    // Add the user's command to the messages
    const newMessages = [...messages, { role: "user" as const, text: input }];
    if (data === null) {
      typeReply("Data not loaded yet.");
      setInput("");
      return;
    }

    let reply = "Unknown command. Try: show steps | show step 5 | show summary";

    if (cmd === "show steps") {
      reply = data.steps.map((s: any) => `Step ${s.step}: ${s.response}`).join("\n\n");
    } else if (cmd.startsWith("show step ")) {
      const numStr = cmd.replace("show step ", "");
      const stepNum = Number(numStr);
      const found = data.steps.find((s: any) => s.step === stepNum);
      reply = found ? found.response : `No such step: ${stepNum}`;
    } else if (cmd === "show summary") {
      reply = data.response ?? "(no summary in json)";
    }
    typeReply(reply);
    setInput("");
  }

  return (
    <div style={{ 
      padding: 24, 
      fontFamily: "system-ui, sans-serif",
      maxWidth: 1100,
      margin: "0 auto", // center the page
      boxSizing: "border-box",
    }}>
      <h1>CMPT733 Final Project — UI</h1>
      <p>If you can see this, React is working.</p>
      {/* Debug toggle button */}
      <button
        onClick={() => setShowDebug((v) => !v)}
        style={{
          padding: "8px 12px",
          borderRadius: 10,
          border: "1px solid #ddd",
          cursor: "pointer",
          marginBottom: 16,
        }}
      >
        {showDebug ? "Hide Debug" : "Show Debug"}
      </button>

      {showDebug && (
        <>  
        <h2>Loaded JSON (raw)</h2>
        <pre style={{ background: "#f2f2f2", padding: 12, borderRadius: 8 }}>
          {data === null ? "Loading..." : JSON.stringify(data, null, 2)}
        </pre>
        </>
      )}

      {/*Simulated Chat UI*/}
      <h2 style={{ marginTop: 30 }}>Simulated Chat</h2>
      <div style={{ 
        border: "1px solid #eee", 
        borderRadius: 12, 
        padding: 12, 
        background: "white",
        width: "100%",
        maxWidth: "100%",
        overflow: "hidden",
        boxSizing: "border-box",
      }}>

        {/* Chat history */}
        <div 
          style={{ 
            display: "flex", 
            flexDirection: "column", 
            gap: 10, 
            marginBottom: 12,
            // -- fixed height + scroll --
            height: 360,
            overflowY: "auto",
            padding: 8,
            borderRadius: 12,
            background: "#fafafa",
            border: "1px solid #eee",
            width: "100%",
            boxSizing: "border-box",    
            minWidth: 0,
            overflowX: "hidden",   // hide horizontal overflow
          }}
        > 
          {/* Chat bubbles */}
          {messages.map((m, idx) => (
            <div 
              key={idx}
              style={{
                alignSelf: m.role === "user" ? "flex-end" : "flex-start",
                background: m.role === "user" ? "#dbeafe" : "white",
                display: "block",
                maxWidth: "78%",
                padding: "10px 12px",
                borderRadius: 14,
                whiteSpace: "pre-wrap",
                lineHeight: 1.55,
                border: "1px solid #eee",
                boxShadow: "0 1px 2px rgba(0,0,0,0.04)",
                wordBreak: "break-word",   // FORCE WORD BREAK
                overflowWrap: "anywhere",  // HANDLE LONG TEXT
                minWidth: 0,
              }}
            >
              {m.text}
            </div>
          ))}
        </div>

        {/* Input row */}
        <div style={{ display: "flex", gap: 8 }}>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder='Try: "show step 5"'
            style={{ flex: 1, padding: 10, borderRadius: 10, border: "1px solid #ddd" }}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                handleSend();
              }}}
          />
          <button
            onClick={handleSend}
            style={{ padding: "10px 14px", borderRadius: 10, border: "1px solid #ddd", cursor: "pointer" }}
          >
            Send
          </button>
        </div>
      </div>

      {/* ------ When data is loaded, show the UI chat messages ------ */}
      {showDebug && data !== null && (
        <div style={{ marginTop: 20 }}>
          <h2>Chat View (Steps)</h2>

          <p style = {{ color: "#666"}}>
            thread_id: <code>{data.thread_id}</code>
          </p>

          {data.steps.map((s: any) => (
            <div 
              key={s.step} 
              style={{ 
                marginBottom: 12, 
                padding: 12, 
                border: "1px solid #eee", 
                background: "#fafafa",
                borderRadius: 12
              }}
            >
              <div style={{ fontWeight: 700, marginBottom: 6 }}>
                Agent — Step {s.step}
              </div>

              <div style = {{ whiteSpace: "pre-wrap", lineHeight: 1.6}}>
                {s.response}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}