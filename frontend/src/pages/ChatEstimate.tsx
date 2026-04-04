import { useEffect, useMemo, useRef, useState } from "react";

/**
 * ------------------------------------------------------------
 * 1. Backend API base URL
 * ------------------------------------------------------------
 * React calls FastAPI here.
 */
const API_BASE = "http://127.0.0.1:8000";

/**
 * ------------------------------------------------------------
 * 2. Basic Types
 * ------------------------------------------------------------
 */

/**
 * One chat message bubble
 */
type Msg = {
  role: "user" | "agent";
  text: string;
};

/**
 * Current collected property profile
 * These fields should match the backend PredictRequest.
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
 * Predict result returned by backend /predict
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
 * Health response returned by backend /health
 */
type HealthResponse = {
  ok: boolean;
  default_report_year: number;
  min_report_year: number;
  max_report_year: number;
};

/**
 * Options response returned by backend /options
 * In v3, options are filtered using context.
 */
type OptionsResponse = {
  LEGAL_TYPE: string[];
  ZONING_DISTRICT: string[];
  ZONING_CLASSIFICATION: string[];
  NEIGHBOURHOOD_CODE: string[];
  context_row_count: number;
  default_report_year: number;
  min_report_year: number;
  max_report_year: number;
};

/**
 * A single guided step in chat
 */
type Step = {
  field: keyof Profile;
  label: string;
  kind: "text" | "number" | "option";
  required: boolean;
  placeholder: string;
  prompt: string;
  helpText: string;
};

/**
 * ------------------------------------------------------------
 * 3. Static Data
 * ------------------------------------------------------------
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

const REQUIRED_FIELDS: (keyof Profile)[] = [
  "PROPERTY_POSTAL_CODE",
  "LEGAL_TYPE",
  "ZONING_DISTRICT",
  "ZONING_CLASSIFICATION",
  "NEIGHBOURHOOD_CODE",
  "YEAR_BUILT",
];

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

const STEPS: Step[] = [
  {
    field: "PROPERTY_POSTAL_CODE",
    label: "Postal Code",
    kind: "text",
    required: true,
    placeholder: "Example: V6B1A1",
    prompt: "Step 1 of 8 — Please enter the property postal code.",
    helpText:
      "Use a valid Canadian postal code, for example: V6B1A1",
  },
  {
    field: "LEGAL_TYPE",
    label: "Legal Type",
    kind: "option",
    required: true,
    placeholder: "Choose a legal type",
    prompt: "Step 2 of 8 — Please choose the legal type.",
    helpText:
      "This list is now filtered by the context collected so far.",
  },
  {
    field: "ZONING_DISTRICT",
    label: "Zoning District",
    kind: "option",
    required: true,
    placeholder: "Choose a zoning district",
    prompt: "Step 3 of 8 — Please choose the zoning district.",
    helpText:
      "This list is filtered by postal code / context when possible.",
  },
  {
    field: "ZONING_CLASSIFICATION",
    label: "Zoning Classification",
    kind: "option",
    required: true,
    placeholder: "Choose a zoning classification",
    prompt: "Step 4 of 8 — Please choose the zoning classification.",
    helpText:
      "This list is filtered by earlier selections when possible.",
  },
  {
    field: "NEIGHBOURHOOD_CODE",
    label: "Neighbourhood Code",
    kind: "option",
    required: true,
    placeholder: "Choose a neighbourhood code",
    prompt: "Step 5 of 8 — Please choose the neighbourhood code.",
    helpText:
      "This list is filtered by postal / zoning context when possible.",
  },
  {
    field: "YEAR_BUILT",
    label: "Year Built",
    kind: "number",
    required: true,
    placeholder: "Example: 1990",
    prompt: "Step 6 of 8 — Please enter the year built.",
    helpText:
      "Year Built should be realistic and should not be later than the selected report year.",
  },
  {
    field: "REPORT_YEAR",
    label: "Report Year (Optional)",
    kind: "number",
    required: false,
    placeholder: "Example: 2026 or type skip",
    prompt: "Step 7 of 8 — Report Year is optional. Enter a year, or type skip.",
    helpText:
      "For stability, Report Year should stay inside the available model years.",
  },
  {
    field: "BIG_IMPROVEMENT_YEAR",
    label: "Big Improvement Year (Optional)",
    kind: "number",
    required: false,
    placeholder: "Example: 2015 or type skip",
    prompt: "Step 8 of 8 — Big Improvement Year is optional. Enter a year, or type skip.",
    helpText:
      "If there is no major improvement year, type skip.",
  },
];

/**
 * ------------------------------------------------------------
 * 4. Helper Functions
 * ------------------------------------------------------------
 */

/**
 * Format money without decimals.
 * Example: 590746 -> CA$590,746
 */
function formatCurrency(value: number): string {
  return new Intl.NumberFormat("en-CA", {
    style: "currency",
    currency: "CAD",
    maximumFractionDigits: 0,
  }).format(value);
}

/**
 * Check if a field has a meaningful value
 */
function hasValue(value: unknown): boolean {
  return value !== "" && value !== null && value !== undefined;
}

/**
 * Find missing required fields
 */
function getMissingFields(profile: Profile): string[] {
  return REQUIRED_FIELDS.filter((field) => !hasValue(profile[field])).map(
    (field) => FIELD_LABELS[field]
  );
}

/**
 * Normalize user-facing text for matching
 */
function normalizeText(value: string): string {
  return value.trim().replace(/\s+/g, " ").toLowerCase();
}

/**
 * Normalize postal code:
 * - uppercase
 * - remove spaces / dashes
 */
function normalizePostalCode(raw: string): string | null {
  const match = raw.match(/\b([A-Za-z]\d[A-Za-z][ -]?\d[A-Za-z]\d)\b/);
  if (!match) return null;
  return match[1].replace(/\s|-/g, "").toUpperCase();
}

/**
 * Match an input against a list of valid options.
 * We support:
 * 1. exact match
 * 2. unique prefix match
 */
function matchOption(input: string, options: string[]): string | null {
  const normalizedInput = normalizeText(input);

  const exact = options.find((opt) => normalizeText(opt) === normalizedInput);
  if (exact) return exact;

  const prefixMatches = options.filter((opt) =>
    normalizeText(opt).startsWith(normalizedInput)
  );
  if (prefixMatches.length === 1) return prefixMatches[0];

  return null;
}

/**
 * Read options for the current field from backend response
 */
function getOptionsForField(
  field: keyof Profile,
  options: OptionsResponse | null
): string[] {
  if (!options) return [];

  switch (field) {
    case "LEGAL_TYPE":
      return options.LEGAL_TYPE;
    case "ZONING_DISTRICT":
      return options.ZONING_DISTRICT;
    case "ZONING_CLASSIFICATION":
      return options.ZONING_CLASSIFICATION;
    case "NEIGHBOURHOOD_CODE":
      return options.NEIGHBOURHOOD_CODE;
    default:
      return [];
  }
}

/**
 * Build query params for /options
 * This is the main v3 improvement:
 * the frontend now sends current context to backend,
 * so backend can return filtered option lists.
 */
function buildOptionsQuery(profile: Profile): string {
  const params = new URLSearchParams();

  if (profile.PROPERTY_POSTAL_CODE) {
    params.set("property_postal_code", profile.PROPERTY_POSTAL_CODE);
  }
  if (profile.LEGAL_TYPE) {
    params.set("legal_type", profile.LEGAL_TYPE);
  }
  if (profile.ZONING_DISTRICT) {
    params.set("zoning_district", profile.ZONING_DISTRICT);
  }
  if (profile.ZONING_CLASSIFICATION) {
    params.set("zoning_classification", profile.ZONING_CLASSIFICATION);
  }
  if (profile.NEIGHBOURHOOD_CODE) {
    params.set("neighbourhood_code", profile.NEIGHBOURHOOD_CODE);
  }
  if (hasValue(profile.REPORT_YEAR)) {
    params.set("report_year", String(profile.REPORT_YEAR));
  }

  return params.toString();
}

/**
 * Validate one step input.
 *
 * This function is stricter in v3:
 * - report year must stay inside backend year bounds
 * - year built cannot be after report year
 * - big improvement year cannot be after report year
 */
function validateStepInput(
  step: Step,
  raw: string,
  options: OptionsResponse | null,
  profile: Profile,
  maxReportYear: number,
  minReportYear: number
): { ok: true; value: string | number | "" } | { ok: false; message: string } {
  const text = raw.trim();

  // Optional steps support "skip"
  if (!step.required && /^skip$/i.test(text)) {
    return { ok: true, value: "" };
  }

  // Postal code validation
  if (step.field === "PROPERTY_POSTAL_CODE") {
    const postal = normalizePostalCode(text);
    if (!postal) {
      return {
        ok: false,
        message:
          "That doesn't look like a valid Canadian postal code. Try something like: V6B1A1",
      };
    }
    return { ok: true, value: postal };
  }

  // Option validation
  if (step.kind === "option") {
    const validOptions = getOptionsForField(step.field, options);

    if (validOptions.length === 0) {
      return {
        ok: false,
        message:
          "There are no filtered options available right now. Please check earlier fields or try again.",
      };
    }

    const matched = matchOption(text, validOptions);
    if (!matched) {
      return {
        ok: false,
        message:
          `I couldn't match that to a valid ${step.label}. ` +
          `Please use the filtered dropdown or type an exact known option.`,
      };
    }

    return { ok: true, value: matched };
  }

  // Number validation
  if (step.kind === "number") {
    const year = Number(text);

    if (!Number.isInteger(year)) {
      return {
        ok: false,
        message: `${step.label} should be a 4-digit year, for example 1990.`,
      };
    }

    // YEAR_BUILT
    if (step.field === "YEAR_BUILT") {
      const targetReportYear = hasValue(profile.REPORT_YEAR)
        ? Number(profile.REPORT_YEAR)
        : maxReportYear;

      if (year < 1800 || year > targetReportYear) {
        return {
          ok: false,
          message:
            `Year Built should be between 1800 and ${targetReportYear}.`,
        };
      }

      return { ok: true, value: year };
    }

    // REPORT_YEAR
    if (step.field === "REPORT_YEAR") {
      if (year < minReportYear || year > maxReportYear) {
        return {
          ok: false,
          message:
            `Report Year should stay between ${minReportYear} and ${maxReportYear}, ` +
            `because the current model only has stable lookup support in that range.`,
        };
      }

      // If YEAR_BUILT already exists, report year should not be earlier
      if (hasValue(profile.YEAR_BUILT) && year < Number(profile.YEAR_BUILT)) {
        return {
          ok: false,
          message:
            `Report Year cannot be earlier than Year Built (${profile.YEAR_BUILT}).`,
        };
      }

      return { ok: true, value: year };
    }

    // BIG_IMPROVEMENT_YEAR
    if (step.field === "BIG_IMPROVEMENT_YEAR") {
      const targetReportYear = hasValue(profile.REPORT_YEAR)
        ? Number(profile.REPORT_YEAR)
        : maxReportYear;

      if (year < 1800 || year > targetReportYear) {
        return {
          ok: false,
          message:
            `Big Improvement Year should be between 1800 and ${targetReportYear}.`,
        };
      }

      if (hasValue(profile.YEAR_BUILT) && year < Number(profile.YEAR_BUILT)) {
        return {
          ok: false,
          message:
            `Big Improvement Year should not be earlier than Year Built (${profile.YEAR_BUILT}).`,
        };
      }

      return { ok: true, value: year };
    }
  }

  if (!text) {
    return {
      ok: false,
      message: `${step.label} cannot be empty.`,
    };
  }

  return { ok: true, value: text };
}

/**
 * ------------------------------------------------------------
 * 5. Main Component
 * ------------------------------------------------------------
 */
export default function ChatEstimate() {
  /**
   * Chat history
   */
  const [messages, setMessages] = useState<Msg[]>([
    {
      role: "agent",
      text:
        "Hi! This page uses a guided chat flow.\n\n" +
        "To improve prediction stability, I will collect one field at a time.\n" +
        "For categorical fields, I will show filtered valid options whenever possible.",
    },
  ]);

  /**
   * Current text in input box
   */
  const [input, setInput] = useState("");

  /**
   * Current collected profile
   */
  const [profile, setProfile] = useState<Profile>(emptyProfile);

  /**
   * Latest predict result
   */
  const [result, setResult] = useState<PredictResult | null>(null);

  /**
   * Backend status
   */
  const [backendOk, setBackendOk] = useState<boolean | null>(null);

  /**
   * Backend health metadata
   */
  const [healthInfo, setHealthInfo] = useState<HealthResponse | null>(null);

  /**
   * Backend filtered options
   */
  const [options, setOptions] = useState<OptionsResponse | null>(null);

  /**
   * Current guided step index
   */
  const [currentStepIndex, setCurrentStepIndex] = useState(0);

  /**
   * Selected option from dropdown
   */
  const [selectedOption, setSelectedOption] = useState("");

  /**
   * Search text inside the option list
   */
  const [optionSearchText, setOptionSearchText] = useState("");

  /**
   * Whether prediction request is running
   */
  const [isPredicting, setIsPredicting] = useState(false);

  /**
   * Typewriter timer
   */
  const typingTimerRef = useRef<number | null>(null);

  /**
   * Bottom ref for auto-scroll
   */
  const bottomRef = useRef<HTMLDivElement | null>(null);

  /**
   * Current step object
   */
  const currentStep = currentStepIndex < STEPS.length ? STEPS[currentStepIndex] : null;

  /**
   * Current step options
   * These are already context-filtered by backend.
   */
  const currentStepOptions =
    currentStep && currentStep.kind === "option"
      ? getOptionsForField(currentStep.field, options)
      : [];

  /**
   * Apply client-side search on top of backend-filtered options.
   * This helps usability when there are still many values.
   */
  const visibleCurrentStepOptions = useMemo(() => {
    const search = normalizeText(optionSearchText);
    if (!search) return currentStepOptions;

    return currentStepOptions.filter((opt) =>
      normalizeText(opt).includes(search)
    );
  }, [currentStepOptions, optionSearchText]);

  /**
   * Auto-scroll on new messages
   */
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  /**
   * Initial boot:
   * - health
   * - first option load
   */
  useEffect(() => {
    async function boot() {
      try {
        const healthRes = await fetch(`${API_BASE}/health`);
        if (!healthRes.ok) throw new Error("Health request failed");

        const healthJson: HealthResponse = await healthRes.json();
        setHealthInfo(healthJson);
        setBackendOk(true);

        // Set default report year from backend metadata
        setProfile((prev) => ({
          ...prev,
          REPORT_YEAR: healthJson.default_report_year,
        }));
      } catch (error) {
        setBackendOk(false);
      }
    }

    boot();
  }, []);

  /**
   * Whenever profile changes, re-request filtered options.
   * This is the main v3 filtering improvement.
   */
  useEffect(() => {
    async function loadFilteredOptions() {
      try {
        const query = buildOptionsQuery(profile);
        const url = query ? `${API_BASE}/options?${query}` : `${API_BASE}/options`;

        const res = await fetch(url);
        if (!res.ok) throw new Error("Options request failed");

        const json: OptionsResponse = await res.json();
        setOptions(json);
      } catch (error) {
        // Keep previous options if request fails
      }
    }

    if (backendOk) {
      loadFilteredOptions();
    }
  }, [profile, backendOk]);

  /**
   * Whenever step changes, show guidance for the new step
   */
  useEffect(() => {
    if (!currentStep) return;

    animateOutput(
      `${currentStep.prompt}\n\n` +
        `Hint: ${currentStep.helpText}` +
        (!currentStep.required ? `\n\nYou can also type: skip` : "")
    );

    setSelectedOption("");
    setOptionSearchText("");
  }, [currentStepIndex]);

  /**
   * Typewriter animation
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
   * Call backend /predict
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
   * Process one user message
   */
  async function processUserMessage(rawValue: string) {
    const raw = rawValue.trim();
    if (!raw) return;

    // Show user message in chat first
    setMessages((prev) => [...prev, { role: "user", text: raw }]);

    // reset command
    if (/^reset$/i.test(raw)) {
      const defaultYear = healthInfo?.default_report_year ?? "";
      setProfile({
        ...emptyProfile,
        REPORT_YEAR: defaultYear,
      });
      setResult(null);
      setCurrentStepIndex(0);

      animateOutput("All fields have been cleared. Let's start again.");
      return;
    }

    // estimate command
    if (/^estimate$/i.test(raw)) {
      const missing = getMissingFields(profile);

      if (missing.length > 0) {
        animateOutput(
          "I’m not ready to estimate yet. I still need:\n- " +
            missing.join("\n- ")
        );
        return;
      }

      await runPrediction(profile);
      return;
    }

    // If all steps completed already
    if (!currentStep) {
      animateOutput(
        "All steps have already been completed.\n\n" +
          "Type `estimate` to run prediction, or `reset` to start over."
      );
      return;
    }

    // Validate current step
    const validation = validateStepInput(
      currentStep,
      raw,
      options,
      profile,
      healthInfo?.max_report_year ?? 2026,
      healthInfo?.min_report_year ?? 2020
    );

    if (!validation.ok) {
      animateOutput(validation.message);
      return;
    }

    // Update profile
    const nextProfile: Profile = {
      ...profile,
      [currentStep.field]: validation.value as never,
    };
    setProfile(nextProfile);

    const savedValue =
      validation.value === "" ? "Skipped" : String(validation.value);

    const reply =
      `Saved ${currentStep.label} = ${savedValue}` +
      (currentStepIndex < STEPS.length - 1
        ? "\n\nMoving to the next step."
        : "\n\nAll steps are complete. Type `estimate` when you are ready.");

    animateOutput(reply);

    // Move forward
    if (currentStepIndex < STEPS.length - 1) {
      setCurrentStepIndex((prev) => prev + 1);
    } else {
      setCurrentStepIndex(STEPS.length);
    }
  }

  /**
   * Send from text input
   */
  async function handleSend() {
    const raw = input.trim();
    if (!raw) return;

    setInput("");
    await processUserMessage(raw);
  }

  /**
   * Use selected dropdown option
   */
  async function handleUseSelectedOption() {
    if (!selectedOption) return;

    await processUserMessage(selectedOption);
    setSelectedOption("");
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">Chat Estimate</h1>
        <p className="mt-1 text-sm text-slate-500">
          Guided conversational UI for land value estimation. The actual
          prediction logic still runs in Python FastAPI.
        </p>
      </div>

      {/* Backend Status */}
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

      {/* Two-column Layout */}
      <div className="grid gap-6 lg:grid-cols-[1.25fr_0.95fr]">
        {/* Left: Guided Chat */}
        <div className="rounded-2xl border border-slate-200 bg-white shadow-sm">
          {/* Chat Header */}
          <div className="border-b border-slate-200 px-4 py-3">
            <div className="text-sm font-semibold">Conversation</div>
            <div className="text-xs text-slate-500">
              To improve stability, this page collects one field at a time.
            </div>
          </div>

          {/* Step Banner */}
          <div className="border-b border-slate-100 bg-slate-50 px-4 py-3">
            {currentStep ? (
              <div className="space-y-1">
                <div className="text-sm font-medium text-slate-800">
                  Current Step: {currentStep.label}
                  {!currentStep.required && (
                    <span className="ml-2 text-xs font-normal text-slate-500">
                      (Optional)
                    </span>
                  )}
                </div>
                <div className="text-xs text-slate-500">{currentStep.helpText}</div>

                {options && currentStep.kind === "option" && (
                  <div className="text-xs text-slate-500">
                    Filtered candidate rows: {options.context_row_count}
                  </div>
                )}
              </div>
            ) : (
              <div className="text-sm text-slate-700">
                All steps completed. Type <span className="font-medium">estimate</span> to run
                prediction, or <span className="font-medium">reset</span> to start over.
              </div>
            )}
          </div>

          {/* Messages Area */}
          <div className="h-[460px] overflow-y-auto overflow-x-hidden bg-slate-50 px-4 py-4">
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

          {/* Input Area */}
          <div className="border-t border-slate-200 p-3 space-y-3">
            {/* Context-aware dropdown for option fields */}
            {currentStep && currentStep.kind === "option" && currentStepOptions.length > 0 && (
              <div className="space-y-2">
                <label className="block text-xs font-medium text-slate-600">
                  Filtered options for {currentStep.label}
                </label>

                {/* Search box for options */}
                <input
                  value={optionSearchText}
                  onChange={(e) => setOptionSearchText(e.target.value)}
                  placeholder={`Search ${currentStep.label} options...`}
                  className="h-10 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm outline-none focus:border-slate-300 focus:ring-2 focus:ring-slate-200"
                />

                <div className="text-xs text-slate-500">
                  Showing {visibleCurrentStepOptions.length} option(s)
                  {optionSearchText ? " after search filtering" : ""}
                </div>

                <div className="flex gap-2">
                  <select
                    value={selectedOption}
                    onChange={(e) => setSelectedOption(e.target.value)}
                    className="h-11 flex-1 rounded-xl border border-slate-200 bg-white px-3 text-sm outline-none focus:border-slate-300 focus:ring-2 focus:ring-slate-200"
                  >
                    <option value="">Select an option...</option>
                    {visibleCurrentStepOptions.map((opt) => (
                      <option key={opt} value={opt}>
                        {opt}
                      </option>
                    ))}
                  </select>

                  <button
                    onClick={handleUseSelectedOption}
                    disabled={!selectedOption}
                    className="h-11 shrink-0 rounded-xl border border-slate-200 bg-white px-4 text-sm font-semibold text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    Use Selected
                  </button>
                </div>
              </div>
            )}

            <div className="text-xs text-slate-500">
              {currentStep
                ? `You can type the value for ${currentStep.label}${
                    currentStep.kind === "option"
                      ? ", or use the filtered dropdown above."
                      : "."
                  }`
                : "Type estimate to run prediction, or reset to clear everything."}
            </div>

            <div className="flex items-center gap-2">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={currentStep ? currentStep.placeholder : "Type estimate or reset"}
                className="h-11 w-full min-w-0 flex-1 rounded-xl border border-slate-200 bg-white px-3 text-sm outline-none focus:border-slate-300 focus:ring-2 focus:ring-slate-200"
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    handleSend();
                  }
                }}
                disabled={isPredicting}
              />
              <button
                onClick={handleSend}
                disabled={isPredicting}
                className="h-11 shrink-0 rounded-xl bg-slate-900 px-4 text-sm font-semibold text-white hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
              >
                Send
              </button>
            </div>
          </div>
        </div>

        {/* Right: Profile + Result */}
        <div className="space-y-4">
          {/* Current Profile */}
          <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="mb-4">
              <h2 className="text-lg font-semibold">Current Property Profile</h2>
              <p className="mt-1 text-sm text-slate-500">
                Collected through the guided chat flow.
              </p>
            </div>

            <div className="space-y-3 text-sm">
              {(Object.keys(profile) as (keyof Profile)[]).map((key) => (
                <div
                  key={key}
                  className="flex items-start justify-between gap-3 border-b border-slate-100 pb-2"
                >
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

            {healthInfo && (
              <div className="mt-3 text-xs text-slate-500">
                Supported report years: {healthInfo.min_report_year} - {healthInfo.max_report_year}
              </div>
            )}
          </div>

          {/* Estimated Result */}
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
                    Error band:{" "}
                    <span className="font-medium">
                      {formatCurrency(result.error_band)}
                    </span>
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
                No prediction yet. Complete the guided steps, then type{" "}
                <span className="font-medium">estimate</span>.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}