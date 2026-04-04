import { useEffect, useRef, useState } from "react";

/**
 *  Backend FastAPI Address
 */
const API_BASE = "http://127.0.0.1:8000";

/**
 * Chat Message
 * role: user / agent
 * text: message content
 */
type Msg = {
  role: "user" | "agent";
  text: string;
};

/**
 * Rigth Side: The right side showing the currently collected property information
 * We only collect fields that correspond to the backend PredictRequest
 */
type Profile = {
  PROPERTY_POSTAL_CODE: string;
  LEGAL_TYPE: string;
  ZONING_DISTRICT: string;
  ZONING_CLASSIFICATION: string;
  NEIGHBOURHOOD_CODE: string;
  YEAR_BUILT: number | "";
  BIG_IMPROVEMENT_YEAR: number | "";
  REPORT_YEAR: number | "";
};

/**
 * Backend /predict returns data structure
 * This structure corresponds to Python's PredictionResult
 */
type PredictResult = {
  point_estimate: number;
  lower_bound: number;
  upper_bound: number;
  error_band: number;
  error_band_source: string;
  used_features: Record<string, unknown>;
};

/**
 * Chat Input Parsing Result
 * updates: Fields extracted from the user's message
 * notes: Feedback messages for the user
 * wantsEstimate: Whether the user is requesting to start prediction
 * wantsReset: Whether the user is requesting to clear/reset
 */
type ParseResult = {
  updates: Partial<Profile>;
  notes: string[];
  wantsEstimate: boolean;
  wantsReset: boolean;
};

/**
 * Initial empty profile.
 */
const emptyProfile: Profile = {
  PROPERTY_POSTAL_CODE: "",
  LEGAL_TYPE: "",
  ZONING_DISTRICT: "",
  ZONING_CLASSIFICATION: "",
  NEIGHBOURHOOD_CODE: "",
  YEAR_BUILT: "",
  BIG_IMPROVEMENT_YEAR: "",
  REPORT_YEAR: "",
};

/**
 * Backend required fields
 * Note: BIG_IMPROVEMENT_YEAR and REPORT_YEAR are both optional
 */
const REQUIRED_FIELDS: (keyof Profile)[] = [
  "PROPERTY_POSTAL_CODE",
  "LEGAL_TYPE",
  "ZONING_DISTRICT",
  "ZONING_CLASSIFICATION",
  "NEIGHBOURHOOD_CODE",
  "YEAR_BUILT",
];

/**
 * Field labels for display purposes
 * The keys must match the Profile keys
 */
const FIELD_LABELS: Record<keyof Profile, string> = {
  PROPERTY_POSTAL_CODE: "Postal Code",
  LEGAL_TYPE: "Legal Type",
  ZONING_DISTRICT: "Zoning District",
  ZONING_CLASSIFICATION: "Zoning Classification",
  NEIGHBOURHOOD_CODE: "Neighbourhood Code",
  YEAR_BUILT: "Year Built",
  BIG_IMPROVEMENT_YEAR: "Big Improvement Year",
  REPORT_YEAR: "Report Year",
};

/**
 * Initialize a readable currency format for display
 * e.g. 800494 -> $800,494
 */
function formatCurrency(value: number): string {
  return `$${value.toLocaleString()}`;
}

/**
 * check if a field has a value
 * empty string => false
 * number 0 is also a valid value, but we don't use 0 for years. 
 * so we can treat 0 as "no value" as well.
 */
function hasValue(value: unknown): boolean {
  return value !== "" && value !== null && value !== undefined;
}

/**
 * find the missing required fields
 */
function getMissingFields(profile: Profile): string[] {
  return REQUIRED_FIELDS.filter((field) => !hasValue(profile[field])).map(
    (field) => FIELD_LABELS[field]
  );
}

/**
 * A simple "rule parser"
 *
 * The goal is not to do real NLP, but to make the chat page work:
 * Users input some simple sentences, we extract the fields from them
 *
 * Supports things like:
 * - postal code V6B1A1
 * - legal type strata
 * - zoning district R1-1
 * - zoning classification Comprehensive Development
 * - neighbourhood 13
 * - built in 1990
 * - report year 2026
 * - improvement year 2015
 * - estimate
 * - reset
 */
function parseUserText(raw: string): ParseResult {
  const text = raw.trim();

  const result: ParseResult = {
    updates: {},
    notes: [],
    wantsEstimate: false,
    wantsReset: false,
  };

  if (!text) return result;

  // user wants to reset/clear/start over
  if (/\b(reset|clear|start over|restart)\b/i.test(text)) {
    result.wantsReset = true;
    return result;
  }

  // user wants to run estimate/prediction
  if (/\b(estimate|predict|run prediction|run estimate)\b/i.test(text)) {
    result.wantsEstimate = true;
  }

  // 1) catch postal code（Canada postal code format）
  const postalMatch = text.match(/\b([A-Za-z]\d[A-Za-z][ -]?\d[A-Za-z]\d)\b/);
  if (postalMatch) {
    const cleaned = postalMatch[1].replace(/\s|-/g, "").toUpperCase();
    result.updates.PROPERTY_POSTAL_CODE = cleaned;
    result.notes.push(`Set Postal Code = ${cleaned}`);
  }

  // 2) catch legal type
  const legalMatch = text.match(/\b(strata|land|other)\b/i);
  if (legalMatch && /\blegal\b|\btype\b/i.test(text)) {
    const legalType = legalMatch[1].toUpperCase();
    result.updates.LEGAL_TYPE = legalType;
    result.notes.push(`Set Legal Type = ${legalType}`);
  }

  // 3) catch zoning district
  const zoningDistrictMatch = text.match(/zoning district\s+([A-Za-z0-9-]+)/i);
  if (zoningDistrictMatch) {
    const district = zoningDistrictMatch[1].trim();
    result.updates.ZONING_DISTRICT = district;
    result.notes.push(`Set Zoning District = ${district}`);
  }

  // 4) catch zoning classification
  // simple implementation: require users to use the format "zoning classification xxx" 
  const zoningClassMatch = text.match(/zoning classification\s+(.+)/i);
  if (zoningClassMatch) {
    const zoningClass = zoningClassMatch[1].trim();
    result.updates.ZONING_CLASSIFICATION = zoningClass;
    result.notes.push(`Set Zoning Classification = ${zoningClass}`);
  }

  // 5) catch neighbourhood code
  const neighMatch = text.match(/\b(neighbourhood|neighborhood)( code)?\s+(\d+)\b/i);
  if (neighMatch) {
    const neigh = neighMatch[3];
    result.updates.NEIGHBOURHOOD_CODE = neigh;
    result.notes.push(`Set Neighbourhood Code = ${neigh}`);
  }

  // 6) catch year built
  const builtMatch =
    text.match(/\bbuilt in\s+(\d{4})\b/i) ||
    text.match(/\byear built\s+(\d{4})\b/i);
  if (builtMatch) {
    const yearBuilt = Number(builtMatch[1]);
    result.updates.YEAR_BUILT = yearBuilt;
    result.notes.push(`Set Year Built = ${yearBuilt}`);
  }

  // 7) catch report year
  const reportYearMatch = text.match(/\breport year\s+(\d{4})\b/i);
  if (reportYearMatch) {
    const reportYear = Number(reportYearMatch[1]);
    result.updates.REPORT_YEAR = reportYear;
    result.notes.push(`Set Report Year = ${reportYear}`);
  }

  // 8) catch improvement year
  const improvementMatch =
    text.match(/\bbig improvement year\s+(\d{4})\b/i) ||
    text.match(/\bimprovement year\s+(\d{4})\b/i);
  if (improvementMatch) {
    const improvementYear = Number(improvementMatch[1]);
    result.updates.BIG_IMPROVEMENT_YEAR = improvementYear;
    result.notes.push(`Set Big Improvement Year = ${improvementYear}`);
  }

  return result;
}

export default function ChatEstimate() {
  /**
   * chat messages history
   */
  const [messages, setMessages] = useState<Msg[]>([
    {
      role: "agent",
      text:
        "Hi! I can help estimate Vancouver assessed land value.\n\n" +
        "You can tell me fields one by one, for example:\n" +
        "- postal code V6B1A1\n" +
        "- legal type strata\n" +
        "- zoning district R1-1\n" +
        "- zoning classification Comprehensive Development\n" +
        "- neighbourhood 13\n" +
        "- built in 1990\n" +
        "- report year 2026\n\n" +
        "When you're ready, type: estimate",
    },
  ]);

  /**
   * input box content
   */
  const [input, setInput] = useState("");

  /**
   * the current property profile extracted from the conversation
   */
  const [profile, setProfile] = useState<Profile>(emptyProfile);

  /**
   * the latest prediction result from the backend
   */
  const [result, setResult] = useState<PredictResult | null>(null);

  /**
   * backend health status: null = checking, true = ok, false = not ok
   */
  const [backendOk, setBackendOk] = useState<boolean | null>(null);

  /**
   * whether a prediction request is in progress. 
   * Used to disable input and show "loading" state.
   */
  const [isPredicting, setIsPredicting] = useState(false);

  /**
   * printer animation timer
   */
  const typingTimerRef = useRef<number | null>(null);

  /**
   * auto scroll to bottom
   */
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  /**
   * page loading: check backend health
   */
  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then((res) => {
        if (!res.ok) throw new Error("Health check failed");
        return res.json();
      })
      .then(() => setBackendOk(true))
      .catch(() => setBackendOk(false));
  }, []);

  /**
   * printer animation
   */
  function animateOutput(fullText: string) {
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
        next[agentIndex] = {
          ...next[agentIndex],
          text: fullText.slice(0, i),
        };
        return next;
      });

      if (i >= fullText.length && typingTimerRef.current !== null) {
        window.clearInterval(typingTimerRef.current);
        typingTimerRef.current = null;
      }
    }, 10);
  }

  /**
   * call backend prediction API
   */
  async function runPrediction(currentProfile: Profile) {
    setIsPredicting(true);
    setResult(null);

    const payload = {
      PROPERTY_POSTAL_CODE: currentProfile.PROPERTY_POSTAL_CODE,
      LEGAL_TYPE: currentProfile.LEGAL_TYPE,
      ZONING_DISTRICT: currentProfile.ZONING_DISTRICT,
      ZONING_CLASSIFICATION: currentProfile.ZONING_CLASSIFICATION,
      NEIGHBOURHOOD_CODE: currentProfile.NEIGHBOURHOOD_CODE,
      YEAR_BUILT: Number(currentProfile.YEAR_BUILT),
      BIG_IMPROVEMENT_YEAR: hasValue(currentProfile.BIG_IMPROVEMENT_YEAR)
        ? Number(currentProfile.BIG_IMPROVEMENT_YEAR)
        : null,
      REPORT_YEAR: hasValue(currentProfile.REPORT_YEAR)
        ? Number(currentProfile.REPORT_YEAR)
        : null,
    };

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(errorText || "Prediction failed");
      }

      const data: PredictResult = await res.json();
      setResult(data);

      animateOutput(
        `Done. I estimated the assessed land value at ${formatCurrency(
          data.point_estimate
        )}.\n\n` +
          `Estimated range: ${formatCurrency(data.lower_bound)} to ${formatCurrency(
            data.upper_bound
          )}.\n\n` +
          `Error band source: ${data.error_band_source}.`
      );
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Unknown prediction error";
      animateOutput(`Prediction failed: ${message}`);
    } finally {
      setIsPredicting(false);
    }
  }

  /**
   * Clicking Send or pressing Enter after typing a message
   */
  async function handleSend() {
    const raw = input.trim();
    if (!raw) return;

    // 1. put the user message into the chat history
    setMessages((prev) => [...prev, { role: "user", text: raw }]);

    // 2. clear the input box
    setInput("");

    // 3. parse user input
    const parsed = parseUserText(raw);

    // 4. if user wants to reset
    if (parsed.wantsReset) {
      setProfile(emptyProfile);
      setResult(null);
      animateOutput("Cleared all current property inputs. You can start again.");
      return;
    }

    // 5. use the newly extracted fields to update the profile
    const nextProfile: Profile = {
      ...profile,
      ...parsed.updates,
    };
    setProfile(nextProfile);

    // 6. if user wants to estimate, check if all fields are provided
    if (parsed.wantsEstimate) {
      const missing = getMissingFields(nextProfile);

      if (missing.length > 0) {
        animateOutput(
          "I’m not ready to estimate yet. I still need:\n- " +
            missing.join("\n- ")
        );
        return;
      }

      await runPrediction(nextProfile);
      return;
    }

    // 7. if user just wants to update fields, not predict, give a chat feedback
    const missing = getMissingFields(nextProfile);

    if (parsed.notes.length > 0) {
      const reply =
        "Updated current property profile:\n- " +
        parsed.notes.join("\n- ") +
        (missing.length === 0
          ? "\n\nAll required fields are ready. Type `estimate` when you want me to run prediction."
          : "\n\nStill needed:\n- " + missing.join("\n- "));
      animateOutput(reply);
    } else {
      animateOutput(
        "I couldn't detect a supported field from that message.\n\n" +
          "Try messages like:\n" +
          "- postal code V6B1A1\n" +
          "- legal type strata\n" +
          "- zoning district R1-1\n" +
          "- zoning classification Comprehensive Development\n" +
          "- neighbourhood 13\n" +
          "- built in 1990\n" +
          "- report year 2026\n" +
          "- estimate"
      );
    }
  }

  return (
    <div className="space-y-6">
      {/* page title */}
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">Chat Estimate</h1>
        <p className="mt-1 text-sm text-slate-500">
          Conversational UI for land value estimation. The prediction logic still
          runs in Python FastAPI.
        </p>
      </div>

      {/* backend status */}
      <div className="flex items-center justify-between">
        <div className="text-sm text-slate-600">
          Backend status:{" "}
          <span className="font-medium">
            {backendOk === null
              ? "Checking..."
              : backendOk
              ? "Connected"
              : "Offline"}
          </span>
        </div>

        <div className="text-xs text-slate-500">
          API: {API_BASE}
        </div>
      </div>

      {/* two column layout: left chat, right status */}
      <div className="grid gap-6 lg:grid-cols-[1.25fr_0.95fr]">
        {/* left chat card */}
        <div className="rounded-2xl border border-slate-200 bg-white shadow-sm">
          {/* chat header */}
          <div className="border-b border-slate-200 px-4 py-3">
            <div className="text-sm font-semibold">Conversation</div>
            <div className="text-xs text-slate-500">
              Tell me the property details step by step, then type: estimate
            </div>
          </div>

          {/* message area */}
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

          {/* input area */}
          <div className="border-t border-slate-200 p-3">
            <div className="mb-2 text-xs text-slate-500">
              Example: <span className="font-medium">postal code V6B1A1</span>
            </div>

            <div className="flex items-center gap-2">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder='Type something like: "built in 1990"'
                className="h-11 w-full min-w-0 flex-1 rounded-xl border border-slate-200 bg-white px-3 text-sm outline-none focus:border-slate-300 focus:ring-2 focus:ring-slate-200"
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    handleSend();
                  }
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

        {/* right info card area */}
        <div className="space-y-4">
          {/* current field card */}
          <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="mb-4">
              <h2 className="text-lg font-semibold">Current Property Profile</h2>
              <p className="mt-1 text-sm text-slate-500">
                Fields extracted from the conversation.
              </p>
            </div>

            <div className="space-y-3 text-sm">
              {(Object.keys(profile) as (keyof Profile)[]).map((key) => (
                <div key={key} className="flex items-start justify-between gap-3 border-b border-slate-100 pb-2">
                  <span className="text-slate-500">{FIELD_LABELS[key]}</span>
                  <span className="text-right font-medium text-slate-900">
                    {hasValue(profile[key]) ? String(profile[key]) : "—"}
                  </span>
                </div>
              ))}
            </div>

            <div className="mt-4 rounded-xl bg-slate-50 p-3 text-sm text-slate-700">
              <div className="font-medium">Missing required fields</div>
              <div className="mt-1">
                {getMissingFields(profile).length === 0
                  ? "None. Ready to estimate."
                  : getMissingFields(profile).join(", ")}
              </div>
            </div>
          </div>

          {/* estimated result card */}
          <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="mb-4">
              <h2 className="text-lg font-semibold">Estimated Result</h2>
              <p className="mt-1 text-sm text-slate-500">
                Returned by the Python prediction API.
              </p>
            </div>

            {isPredicting ? (
              <div className="text-sm text-slate-600">Running prediction...</div>
            ) : result ? (
              <div className="space-y-4">
                <div>
                  <div className="text-sm text-slate-500">
                    Point Estimate (Assessed Land Value)
                  </div>
                  <div className="mt-1 text-4xl font-semibold tracking-tight text-slate-900">
                    {formatCurrency(result.point_estimate)}
                  </div>
                </div>

                <div>
                  <div className="text-sm font-medium text-slate-800">
                    Estimated Range
                  </div>
                  <div className="mt-1 text-sm text-slate-700">
                    {formatCurrency(result.lower_bound)} to{" "}
                    {formatCurrency(result.upper_bound)}
                  </div>
                </div>

                <div className="rounded-xl bg-slate-50 p-3 text-sm text-slate-700">
                  <div>
                    Error band: <span className="font-medium">{formatCurrency(result.error_band)}</span>
                  </div>
                  <div className="mt-1">
                    Source: <span className="font-medium">{result.error_band_source}</span>
                  </div>
                </div>

                <div>
                  <div className="mb-2 text-sm font-medium text-slate-800">
                    Derived / Lookup Details
                  </div>
                  <pre className="max-h-48 overflow-auto rounded-xl bg-slate-50 p-3 text-xs text-slate-700">
                    {JSON.stringify(result.used_features, null, 2)}
                  </pre>
                </div>
              </div>
            ) : (
              <div className="text-sm text-slate-600">
                No prediction yet. Fill the required fields in chat, then type{" "}
                <span className="font-medium">estimate</span>.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}