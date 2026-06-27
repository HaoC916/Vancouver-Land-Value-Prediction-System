import { useEffect, useRef, useState } from "react";

import { API_BASE } from "../config";

type Msg = {
  role: "user" | "agent";
  text: string;
};

type FuzzyCandidate = {
  candidate_id: number;
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
};

type FuzzyLookupResponse = {
  match_count: number;
  auto_selected: boolean;
  candidates: FuzzyCandidate[];
};

type PredictResult = {
  point_estimate: number;
  lower_bound: number;
  upper_bound: number;
  error_band: number;
  error_band_source: string;
  used_features: Record<string, unknown>;
};

type Phase = "address" | "choose" | "result";

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

const GREETING =
  "Hi! Tell me a Vancouver street address and I'll estimate its property value.\n\n" +
  "For example: 1128 Hastings St W. You can add a postal code too, e.g. 1128 Hastings St W, V6E 4R5.";

export default function PreciseMode() {
  const [messages, setMessages] = useState<Msg[]>([{ role: "agent", text: GREETING }]);
  const [input, setInput] = useState("");
  const [phase, setPhase] = useState<Phase>("address");
  const [candidates, setCandidates] = useState<FuzzyCandidate[]>([]);
  const [selected, setSelected] = useState<FuzzyCandidate | null>(null);
  const [result, setResult] = useState<PredictResult | null>(null);
  const [backendOk, setBackendOk] = useState<boolean | null>(null);
  const [isBusy, setIsBusy] = useState(false);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
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
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, candidates]);

  function addMsg(role: "user" | "agent", text: string) {
    setMessages((prev) => [...prev, { role, text }]);
  }

  function resetConversation() {
    setPhase("address");
    setCandidates([]);
    setSelected(null);
    setResult(null);
    setInput("");
    setMessages([{ role: "agent", text: GREETING }]);
  }

  async function lookupAddress(text: string) {
    // "1050 26TH AVE W, V6H 2A5" -> street number, street name, optional postal
    const [addrPart, postalPart] = text.split(",");
    const addr = (addrPart || "").trim();
    const match = addr.match(/^(\d+[A-Za-z]?)\s+(.+)$/);
    if (!match) {
      addMsg(
        "agent",
        "Start with the street number, then the street name — e.g. 1128 Hastings St W."
      );
      return;
    }
    const streetNumber = match[1];
    const streetName = match[2].trim();
    const postal = postalPart ? normalizePostalCode(postalPart) : "";

    setIsBusy(true);
    try {
      const params = new URLSearchParams();
      params.set("street_number", streetNumber);
      params.set("street_name", streetName);
      if (postal) params.set("property_postal_code", postal);
      params.set("limit", "20");

      const res = await fetch(`${API_BASE}/fuzzy_lookup?${params.toString()}`);
      if (!res.ok) throw new Error(await res.text());
      const data: FuzzyLookupResponse = await res.json();

      if (data.match_count === 0) {
        setCandidates([]);
        setPhase("address");
        addMsg(
          "agent",
          "I couldn't find that address. Check the street number and name, or add a postal code. Note this only covers City of Vancouver addresses (Burnaby, Richmond, etc. aren't included)."
        );
        return;
      }
      if (data.auto_selected && data.candidates.length === 1) {
        setCandidates([]);
        await estimate(data.candidates[0]);
        return;
      }
      setCandidates(data.candidates);
      setPhase("choose");
      addMsg(
        "agent",
        `I found ${data.match_count} properties at this address — which one? Tap a unit below, or reply with its number.`
      );
    } catch (e) {
      addMsg("agent", `Lookup failed: ${e instanceof Error ? e.message : "unknown error"}`);
    } finally {
      setIsBusy(false);
    }
  }

  async function estimate(candidate: FuzzyCandidate) {
    setSelected(candidate);
    setCandidates([]);
    setResult(null);
    addMsg("agent", `Got it — ${candidate.display_address}. Estimating…`);

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
      setPhase("result");
      addMsg(
        "agent",
        `Estimated property value: ${formatCurrency(data.point_estimate)} ` +
          `(likely ${formatCurrency(data.lower_bound)} – ${formatCurrency(data.upper_bound)}).\n\n` +
          "That's the total assessed value, land plus building. Want another address? Just type it."
      );
    } catch (e) {
      setPhase("address");
      addMsg("agent", `Prediction failed: ${e instanceof Error ? e.message : "unknown error"}`);
    } finally {
      setIsBusy(false);
    }
  }

  async function handleSend() {
    const raw = input.trim();
    if (!raw || isBusy) return;
    setInput("");
    addMsg("user", raw);

    const cmd = raw.toLowerCase();
    if (cmd === "reset" || cmd === "restart" || cmd === "start over") {
      resetConversation();
      return;
    }
    if (cmd === "back") {
      if (phase === "choose") {
        setCandidates([]);
        setPhase("address");
        addMsg("agent", "Sure — what's the address?");
      } else {
        addMsg("agent", "Type a Vancouver street address to get started.");
      }
      return;
    }

    if (phase === "choose") {
      const n = Number(raw);
      if (Number.isInteger(n) && n >= 1 && n <= candidates.length) {
        await estimate(candidates[n - 1]);
        return;
      }
      // not a valid number — treat as a fresh address search
      await lookupAddress(raw);
      return;
    }

    // address / result phase: treat the message as an address
    await lookupAddress(raw);
  }

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
        <div className="rounded-2xl border border-slate-200 bg-white shadow-sm">
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

              {phase === "choose" && candidates.length > 0 && (
                <div className="mr-auto flex max-w-[90%] flex-col gap-2">
                  {candidates.map((c, idx) => (
                    <button
                      key={c.candidate_id}
                      onClick={() => estimate(c)}
                      disabled={isBusy}
                      className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-left text-sm shadow-sm hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      <span className="mr-2 text-slate-400">{idx + 1}.</span>
                      <span className="font-medium text-slate-900">{c.display_address}</span>
                      <span className="ml-1 text-slate-500">
                        · {c.LEGAL_TYPE} · built {valueOrDash(c.YEAR_BUILT)}
                      </span>
                    </button>
                  ))}
                </div>
              )}

              <div ref={bottomRef} />
            </div>
          </div>

          <div className="border-t border-slate-200 p-3 space-y-2">
            <div className="text-xs text-slate-500">
              Type an address, or <span className="font-medium">reset</span> to start over.
              When choosing a unit you can also type <span className="font-medium">back</span>.
            </div>
            <div className="flex items-center gap-2">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Example: 1128 Hastings St W"
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
                      {
                        error_band_source: result.error_band_source,
                        ...result.used_features,
                      },
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
