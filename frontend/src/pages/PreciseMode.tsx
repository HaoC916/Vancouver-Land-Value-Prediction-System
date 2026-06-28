import { useEffect, useRef, useState } from "react";

import { API_BASE } from "../config";

type Msg = {
  role: "user" | "agent";
  text: string;
};

type ResolvedProperty = {
  PID: string | null;
  display_address: string;
  PROPERTY_POSTAL_CODE: string;
  LEGAL_TYPE: string;
  ZONING_DISTRICT: string;
  ZONING_CLASSIFICATION: string;
  NEIGHBOURHOOD_CODE: string;
  YEAR_BUILT: number | null;
  BIG_IMPROVEMENT_YEAR: number | null;
  REPORT_YEAR: number;
  UNIT: string;
};

type ResolveResponse = {
  status: "single" | "need_unit" | "none";
  candidate: ResolvedProperty | null;
  unit_count: number;
};

type PredictResult = {
  point_estimate: number;
  lower_bound: number;
  upper_bound: number;
  error_band: number;
  error_band_source: string;
  used_features: Record<string, unknown>;
};

type ParsedAddress = {
  unit: string;
  streetNumber: string;
  streetName: string;
  postal: string;
  raw: string;
};
type Phase = "address" | "unit" | "result";

function formatCurrency(n: number): string {
  return new Intl.NumberFormat("en-CA", {
    style: "currency",
    currency: "CAD",
    maximumFractionDigits: 0,
  }).format(n);
}

function normalizePostalCode(value: string): string {
  return value.trim().toUpperCase().replace(/[\s-]/g, "");
}

function valueOrDash(value: unknown): string {
  return value === null || value === undefined || value === "" ? "—" : String(value);
}

function parseAddress(text: string): ParsedAddress | null {
  const [addrPart, postalPart] = text.split(",");
  const raw = (addrPart || "").trim();
  let addr = raw;
  let unit = "";

  // Optional "UNIT-BUILDING" prefix, e.g. "2301-1128 Hastings St W" -> unit 2301,
  // building 1128. (The city writes condo addresses this way.)
  const unitMatch = addr.match(/^(\d+[A-Za-z]?)\s*-\s*(\d.*)$/);
  if (unitMatch) {
    unit = unitMatch[1];
    addr = unitMatch[2].trim();
  }

  const match = addr.match(/^(\d+[A-Za-z]?)\s+(.+)$/);
  if (!match) return null;
  return {
    unit,
    streetNumber: match[1],
    streetName: match[2].trim(),
    postal: postalPart ? normalizePostalCode(postalPart) : "",
    raw,
  };
}

const GREETING =
  "Hi! Tell me a Vancouver street address and I'll estimate its property value.\n\n" +
  "For example: 1128 Hastings St W. You can add a postal code too, e.g. 1128 Hastings St W, V6E 4R5.";

export default function PreciseMode() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [phase, setPhase] = useState<Phase>("address");
  const [pendingAddr, setPendingAddr] = useState<ParsedAddress | null>(null);
  const [selected, setSelected] = useState<ResolvedProperty | null>(null);
  const [result, setResult] = useState<PredictResult | null>(null);
  const [backendOk, setBackendOk] = useState<boolean | null>(null);
  const [isBusy, setIsBusy] = useState(false);

  const bottomRef = useRef<HTMLDivElement | null>(null);
  const typingTimerRef = useRef<number | null>(null);
  const pendingAgentMessagesRef = useRef<string[]>([]);
  const isAnimatingRef = useRef(false);
  const hasBootedRef = useRef(false);

  // ---- typewriter: reveal agent messages character by character ----
  function playNextAgentMessage() {
    if (isAnimatingRef.current) return;
    const nextText = pendingAgentMessagesRef.current.shift();
    if (!nextText) return;

    isAnimatingRef.current = true;
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
        next[agentIndex] = { ...next[agentIndex], text: nextText.slice(0, i) };
        return next;
      });
      if (i >= nextText.length && typingTimerRef.current !== null) {
        window.clearInterval(typingTimerRef.current);
        typingTimerRef.current = null;
        isAnimatingRef.current = false;
        playNextAgentMessage();
      }
    }, 12);
  }

  function queueAgentMessage(text: string) {
    pendingAgentMessagesRef.current.push(text);
    playNextAgentMessage();
  }

  function addUserMessage(text: string) {
    setMessages((prev) => [...prev, { role: "user", text }]);
  }

  useEffect(() => {
    if (hasBootedRef.current) return;
    hasBootedRef.current = true;
    queueAgentMessage(GREETING);
    async function loadHealth() {
      try {
        const res = await fetch(`${API_BASE}/health`);
        if (!res.ok) throw new Error("health failed");
        await res.json();
        setBackendOk(true);
      } catch {
        setBackendOk(false);
      }
    }
    loadHealth();
    // Note: we intentionally do NOT clear the typing timer in a cleanup here.
    // Under React StrictMode the mount effect runs twice (mount → cleanup →
    // mount); clearing the timer in cleanup would kill the greeting animation and
    // leave an empty bubble. The interval clears itself when a message finishes.
    // run once on mount: greet + health check
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function resetConversation() {
    pendingAgentMessagesRef.current = [];
    if (typingTimerRef.current !== null) {
      window.clearInterval(typingTimerRef.current);
      typingTimerRef.current = null;
    }
    isAnimatingRef.current = false;
    setMessages([]);
    setPhase("address");
    setPendingAddr(null);
    setSelected(null);
    setResult(null);
    setInput("");
    queueAgentMessage("Okay, starting over. What's the address?");
  }

  async function estimate(candidate: ResolvedProperty) {
    setSelected(candidate);
    setResult(null);
    setPhase("result");
    queueAgentMessage(`Got it — ${candidate.display_address}. Estimating…`);

    setIsBusy(true);
    try {
      const payload = {
        PROPERTY_POSTAL_CODE: candidate.PROPERTY_POSTAL_CODE,
        LEGAL_TYPE: candidate.LEGAL_TYPE,
        ZONING_DISTRICT: candidate.ZONING_DISTRICT,
        ZONING_CLASSIFICATION: candidate.ZONING_CLASSIFICATION,
        NEIGHBOURHOOD_CODE: candidate.NEIGHBOURHOOD_CODE,
        YEAR_BUILT: candidate.YEAR_BUILT,
        BIG_IMPROVEMENT_YEAR: candidate.BIG_IMPROVEMENT_YEAR,
        REPORT_YEAR: candidate.REPORT_YEAR,
        PID: candidate.PID,
      };
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(await res.text());
      const data: PredictResult = await res.json();
      setResult(data);
      queueAgentMessage(
        `Estimated property value: ${formatCurrency(data.point_estimate)} ` +
          `(likely ${formatCurrency(data.lower_bound)} – ${formatCurrency(data.upper_bound)}).\n\n` +
          "That's the total assessed value, land plus building. Want another address? Just type it."
      );
    } catch (e) {
      setPhase("address");
      queueAgentMessage(`Sorry, the estimate failed: ${e instanceof Error ? e.message : "unknown error"}`);
    } finally {
      setIsBusy(false);
    }
  }

  async function resolve(addr: ParsedAddress, unit?: string) {
    setIsBusy(true);
    try {
      const params = new URLSearchParams();
      params.set("street_number", addr.streetNumber);
      params.set("street_name", addr.streetName);
      if (addr.postal) params.set("property_postal_code", addr.postal);
      if (unit) params.set("unit", unit);

      const res = await fetch(`${API_BASE}/resolve_address?${params.toString()}`);
      if (!res.ok) throw new Error(await res.text());
      const data: ResolveResponse = await res.json();

      if (data.status === "single" && data.candidate) {
        setPendingAddr(null);
        await estimate(data.candidate);
        return;
      }
      if (data.status === "need_unit") {
        setPendingAddr(addr);
        setPhase("unit");
        queueAgentMessage(
          `${addr.raw} is a multi-unit building. What's your unit number? (for example 2308)`
        );
        return;
      }
      // none
      if (unit) {
        setPendingAddr(addr);
        setPhase("unit");
        queueAgentMessage(`I couldn't find unit ${unit} at ${addr.raw}. What's the unit number?`);
      } else {
        setPhase("address");
        queueAgentMessage(
          "I couldn't find that address. Check the street number and name, or add a postal code. " +
            "Note this only covers City of Vancouver addresses (Burnaby, Richmond, etc. aren't included)."
        );
      }
    } catch (e) {
      queueAgentMessage(`Lookup failed: ${e instanceof Error ? e.message : "unknown error"}`);
    } finally {
      setIsBusy(false);
    }
  }

  async function handleSend() {
    const raw = input.trim();
    if (!raw || isBusy) return;
    setInput("");
    addUserMessage(raw);

    const cmd = raw.toLowerCase();
    if (cmd === "reset" || cmd === "restart" || cmd === "start over") {
      resetConversation();
      return;
    }
    if (cmd === "back") {
      if (phase === "unit") {
        setPhase("address");
        setPendingAddr(null);
        queueAgentMessage("Sure — what's the address?");
      } else {
        queueAgentMessage("Type a Vancouver street address to get started.");
      }
      return;
    }

    if (phase === "unit" && pendingAddr) {
      // input is a unit number
      await resolve(pendingAddr, raw);
      return;
    }

    // address / result phase: treat as an address
    const parsed = parseAddress(raw);
    if (!parsed) {
      queueAgentMessage("Start with the street number, then the street name — e.g. 1128 Hastings St W.");
      return;
    }
    await resolve(parsed, parsed.unit || undefined);
  }

  const phaseHint =
    phase === "unit"
      ? "Enter your unit number, or type back to change the address."
      : "Type a Vancouver address, or reset to start over.";

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">Chat</h1>
      </div>

      <div className="flex items-center justify-between">
        <div className="text-sm text-slate-600">
          Backend status:{" "}
          <span className="font-medium">
            {backendOk === null ? "Checking..." : backendOk ? "Connected" : "Offline"}
          </span>
        </div>
        <div className="text-xs text-slate-500">API: {API_BASE}</div>
      </div>

      <div className="grid gap-6 lg:grid-cols-[1.25fr_0.95fr]">
        {/* Left: chat */}
        <div className="self-start rounded-2xl border border-slate-200 bg-white shadow-sm">
          <div className="border-b border-slate-200 px-4 py-3">
            <div className="text-sm font-semibold">Chat Window</div>
          </div>

          <div className="h-[520px] overflow-y-auto overflow-x-hidden bg-slate-50 px-4 py-4">
            <div className="flex flex-col gap-3">
              {messages.map((m, idx) => (
                <div
                  key={idx}
                  className={[
                    "min-w-0 max-w-[80%] whitespace-pre-wrap [overflow-wrap:anywhere]",
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

          <div className="border-t border-slate-200 p-3 space-y-2">
            <div className="text-xs text-slate-500">{phaseHint}</div>
            <div className="flex items-center gap-2">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={phase === "unit" ? "Example: 2308" : "Example: 1128 Hastings St W"}
                className="h-11 w-full min-w-0 flex-1 rounded-xl border border-slate-200 bg-white px-3 text-sm outline-none focus:border-slate-300 focus:ring-2 focus:ring-slate-200"
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleSend();
                }}
                disabled={isBusy}
              />
              <button
                onClick={handleSend}
                disabled={isBusy}
                className="h-11 shrink-0 rounded-xl bg-slate-900 px-4 text-sm font-semibold text-white hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
              >
                Send
              </button>
            </div>
          </div>
        </div>

        {/* Right: what we found + result */}
        <div className="space-y-4">
          <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="mb-4">
              <h2 className="text-lg font-semibold">What we found</h2>
            </div>
            <div className="space-y-3 text-sm">
              {[
                ["Address", selected?.display_address],
                ["Postal code", selected?.PROPERTY_POSTAL_CODE],
                ["Property type", selected?.LEGAL_TYPE],
                ["Zoning", selected?.ZONING_DISTRICT],
                ["Neighbourhood", selected?.NEIGHBOURHOOD_CODE],
                ["Year built", selected?.YEAR_BUILT],
                ["Last major improvement", selected?.BIG_IMPROVEMENT_YEAR],
                ["Assessment year", selected?.REPORT_YEAR],
              ].map(([label, value]) => (
                <div
                  key={label as string}
                  className="flex items-start justify-between gap-3 border-b border-slate-100 pb-2"
                >
                  <span className="text-slate-500">{label}</span>
                  <span className="text-right font-medium text-slate-900">
                    {valueOrDash(value)}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="mb-4">
              <h2 className="text-lg font-semibold">Estimated Result</h2>
            </div>
            {isBusy && !result ? (
              <div className="text-sm text-slate-600">Running estimate...</div>
            ) : result ? (
              <div className="space-y-4">
                <div>
                  <div className="text-sm text-slate-500">Estimated property value</div>
                  <div className="mt-1 text-4xl font-semibold tracking-tight text-slate-900">
                    {formatCurrency(result.point_estimate)}
                  </div>
                </div>
                <div>
                  <div className="text-sm font-medium text-slate-800">Likely range</div>
                  <div className="mt-1 text-sm text-slate-700">
                    {formatCurrency(result.lower_bound)} to {formatCurrency(result.upper_bound)}
                  </div>
                </div>
                <p className="text-sm text-slate-600">
                  This is a model estimate of the total assessed property value (land plus
                  building) — not a guaranteed sale price or an official appraisal. The range
                  reflects how much similar properties in this area typically vary (about{" "}
                  {formatCurrency(result.error_band)} either way).
                </p>
                <details className="text-sm text-slate-600">
                  <summary className="cursor-pointer font-medium text-slate-800">
                    Technical details
                  </summary>
                  <pre className="mt-2 max-h-48 overflow-auto rounded-xl bg-slate-50 p-3 text-xs text-slate-700">
                    {JSON.stringify(
                      { error_band_source: result.error_band_source, ...result.used_features },
                      null,
                      2
                    )}
                  </pre>
                </details>
              </div>
            ) : (
              <div className="text-sm text-slate-600">
                No estimate yet. Send a Vancouver address in the chat to get started.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
