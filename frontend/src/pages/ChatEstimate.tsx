import { useEffect, useRef, useState } from "react";

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
 * One chat bubble message.
 */
type Msg = {
  role: "user" | "agent";
  text: string;
};

/**
 * Current collected property profile.
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
 * Prediction result returned by backend /predict.
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
 * Health response returned by backend /health.
 */
type HealthResponse = {
  ok: boolean;
  default_report_year: number;
  min_report_year: number;
  max_report_year: number;
};

/**
 * Options response returned by backend /options.
 * These options are already filtered by backend context.
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
 * One guided step definition.
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
    helpText: "Use a valid Canadian postal code, for example: V6B1A1",
  },
  {
    field: "LEGAL_TYPE",
    label: "Legal Type",
    kind: "option",
    required: true,
    placeholder: "Type one of the suggested legal types",
    prompt: "Step 2 of 8 — Please choose the legal type.",
    helpText: "This list is now filtered by the context collected so far.",
  },
  {
    field: "ZONING_DISTRICT",
    label: "Zoning District",
    kind: "option",
    required: true,
    placeholder: "Type one of the suggested zoning districts",
    prompt: "Step 3 of 8 — Please choose the zoning district.",
    helpText: "This list is filtered by postal code / context when possible.",
  },
  {
    field: "ZONING_CLASSIFICATION",
    label: "Zoning Classification",
    kind: "option",
    required: true,
    placeholder: "Type one of the suggested zoning classifications",
    prompt: "Step 4 of 8 — Please choose the zoning classification.",
    helpText: "This list is filtered by earlier selections when possible.",
  },
  {
    field: "NEIGHBOURHOOD_CODE",
    label: "Neighbourhood Code",
    kind: "option",
    required: true,
    placeholder: "Type one of the suggested neighbourhood codes",
    prompt: "Step 5 of 8 — Please choose the neighbourhood code.",
    helpText: "This list is filtered by postal / zoning context when possible.",
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
      "Report Year should stay inside the available model years.",
  },
  {
    field: "BIG_IMPROVEMENT_YEAR",
    label: "Big Improvement Year (Optional)",
    kind: "number",
    required: false,
    placeholder: "Example: 2015 or type skip",
    prompt:
      "Step 8 of 8 — Big Improvement Year is optional. Enter a year, or type skip.",
    helpText: "If there is no major improvement year, type skip.",
  },
];

/**
 * ------------------------------------------------------------
 * 4. Helper Functions
 * ------------------------------------------------------------
 */

/**
 * Format money without decimals.
 */
function formatCurrency(value: number): string {
  return new Intl.NumberFormat("en-CA", {
    style: "currency",
    currency: "CAD",
    maximumFractionDigits: 0,
  }).format(value);
}

/**
 * Check whether a value should be treated as filled.
 */
function hasValue(value: unknown): boolean {
  return value !== "" && value !== null && value !== undefined;
}

/**
 * Find missing required fields.
 */
function getMissingFields(profile: Profile): string[] {
  return REQUIRED_FIELDS.filter((field) => !hasValue(profile[field])).map(
    (field) => FIELD_LABELS[field]
  );
}

/**
 * Normalize free text for matching / comparison.
 */
function normalizeText(value: string): string {
  return value.trim().replace(/\s+/g, " ").toLowerCase();
}

/**
 * Normalize Canadian postal code:
 * - uppercase
 * - remove spaces / dashes
 */
function normalizePostalCode(raw: string): string | null {
  const match = raw.match(/\b([A-Za-z]\d[A-Za-z][ -]?\d[A-Za-z]\d)\b/);
  if (!match) return null;
  return match[1].replace(/\s|-/g, "").toUpperCase();
}

/**
 * Match user input to one valid option.
 *
 * Matching rules:
 * 1. exact match
 * 2. unique prefix match
 *
 * Example:
 * - "land" can match "LAND"
 * - "ha" can match "HA-2" only if it is the unique prefix match
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
 * Read filtered options for one field from backend response.
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
 * Build query params for backend /options
 * using the currently known profile context.
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
 * Build the chat prompt for the current step.
 *
 * IMPORTANT DESIGN CHOICE:
 * - The chat remains the only interaction area.
 * - We do NOT render any separate dropdown/filter UI block.
 * - Filtered options are injected directly into the chat prompt.
 *
 * Also:
 * - For option steps, we do NOT print a generic "Hint: ..." first
 *   and then another option hint.
 * - We only print ONE clean hint block.
 */
function buildStepPromptMessage(
  step: Step,
  options: OptionsResponse | null
): string {
  let text = step.prompt;

  // Non-option steps keep the regular help text
  if (step.kind !== "option") {
    text += `\n\nHint: ${step.helpText}`;
  }

  // Option steps only show the filtered option hint
  if (step.kind === "option") {
    const filteredOptions = getOptionsForField(step.field, options);

    if (filteredOptions.length === 0) {
      text +=
        "\n\nHint: I do not have filtered options yet. Please type a value manually, or check earlier fields.";
    } else if (filteredOptions.length === 1) {
      text += `\n\nHint: The filtered ${step.label.toLowerCase()} is: ${filteredOptions[0]}.`;
    } else {
      const previewCount = 8;
      const shown = filteredOptions.slice(0, previewCount);
      const remaining = filteredOptions.length - shown.length;

      text += `\n\nHint: The available ${step.label.toLowerCase()} options are: ${shown.join(", ")}`;

      if (remaining > 0) {
        text += `, and ${remaining} more`;
      }

      text += ".";
    }

    text += "\n\nPlease type one of the suggested options.";
  }

  // Optional steps support skip
  if (!step.required) {
    text += "\n\nYou can also type: skip";
  }

  return text;
}

/**
 * Validate one step input.
 *
 * Validation rules:
 * - report year must stay inside backend-supported year bounds
 * - year built cannot be after report year
 * - big improvement year cannot be after report year
 * - categorical fields must match filtered options
 *
 * NOTE:
 * Because REPORT_YEAR is no longer prefilled,
 * when REPORT_YEAR is still empty we use maxReportYear
 * as the upper fallback for related year checks.
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
          `Please type one of the suggested options from the latest chat hint.`,
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
          message: `Year Built should be between 1800 and ${targetReportYear}.`,
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
            `Report Year should stay between ${minReportYear} and ${maxReportYear}.`,
        };
      }

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
   * Chat history.
   *
   * We KEEP the typewriter animation because that is part of the
   * immersive conversation feel you want.
   */
  const [messages, setMessages] = useState<Msg[]>([
    {
      role: "agent",
      text:
        "Hi! This page uses a guided chat flow.\n\n" +
        "The chat remains the main interaction area. For each categorical step, I will show filtered suggested options directly inside the conversation.",
    },
  ]);

  /**
   * Current input text
   */
  const [input, setInput] = useState("");

  /**
   * Current collected profile
   */
  const [profile, setProfile] = useState<Profile>(emptyProfile);

  /**
   * Latest prediction result
   */
  const [result, setResult] = useState<PredictResult | null>(null);

  /**
   * Backend connection status
   */
  const [backendOk, setBackendOk] = useState<boolean | null>(null);

  /**
   * Backend health metadata
   */
  const [healthInfo, setHealthInfo] = useState<HealthResponse | null>(null);

  /**
   * Current filtered options from backend
   */
  const [options, setOptions] = useState<OptionsResponse | null>(null);

  /**
   * Current step index in the guided flow
   */
  const [currentStepIndex, setCurrentStepIndex] = useState(0);

  /**
   * Whether prediction request is running
   */
  const [isPredicting, setIsPredicting] = useState(false);

  /**
   * Typewriter timer
   */
  const typingTimerRef = useRef<number | null>(null);

  /**
   * Bottom target for scrolling
   */
  const bottomRef = useRef<HTMLDivElement | null>(null);

  /**
   * Queue of pending agent messages.
   *
   * Why do we need a queue?
   * Because with typewriter animation, if we start a new agent message
   * before the previous one has finished typing, the old one gets cut off.
   *
   * The queue guarantees:
   * - messages are typed one by one
   * - no truncated first prompt
   * - no duplicated / interrupted prompt sequence
   */
  const pendingAgentMessagesRef = useRef<string[]>([]);

  /**
   * Whether an agent message is currently animating.
   */
  const isAnimatingRef = useRef(false);

  /**
   * Guard against React StrictMode double-running the first effect in development.
   *
   * Without this, the initial boot flow may run twice,
   * which causes the first prompt to appear partially, then restart.
   */
  const hasBootedRef = useRef(false);

  /**
   * Current step object
   */
  const currentStep =
    currentStepIndex < STEPS.length ? STEPS[currentStepIndex] : null;

  /**
   * Auto-scroll on message changes.
   * We keep this because you want the chat to feel alive while typing.
   */
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  /**
   * Fetch filtered options for a given profile context.
   *
   * This asks the backend:
   * "Given what the user has already entered,
   * what are the filtered candidate options now?"
   */
  async function fetchOptionsForProfile(
    targetProfile: Profile
  ): Promise<OptionsResponse | null> {
    try {
      const query = buildOptionsQuery(targetProfile);
      const url = query ? `${API_BASE}/options?${query}` : `${API_BASE}/options`;

      const res = await fetch(url);
      if (!res.ok) {
        throw new Error("Options request failed");
      }

      const json: OptionsResponse = await res.json();
      return json;
    } catch (error) {
      return null;
    }
  }

  /**
   * Start typing the next queued agent message.
   *
   * If no queued message exists, stop.
   */
  function playNextAgentMessage() {
    // If already animating, do nothing.
    if (isAnimatingRef.current) return;

    const nextText = pendingAgentMessagesRef.current.shift();
    if (!nextText) return;

    isAnimatingRef.current = true;

    let agentIndex = -1;

    // Insert a new empty agent bubble
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
          text: nextText.slice(0, i),
        };
        return next;
      });

      if (i >= nextText.length && typingTimerRef.current !== null) {
        window.clearInterval(typingTimerRef.current);
        typingTimerRef.current = null;
        isAnimatingRef.current = false;

        // Automatically continue to the next queued agent message
        playNextAgentMessage();
      }
    }, 10);
  }

  /**
   * Queue one full agent message.
   *
   * This is the ONLY way we should create agent outputs now.
   * It prevents truncation and duplicate-start issues.
   */
  function queueAgentMessage(text: string) {
    pendingAgentMessagesRef.current.push(text);
    playNextAgentMessage();
  }

  /**
   * Initial boot:
   * - fetch backend health
   * - do NOT prefill REPORT_YEAR into the visible profile
   * - fetch first-step options using the empty profile
   * - start the first chat prompt
   */
  useEffect(() => {
    async function boot() {
      try {
        const healthRes = await fetch(`${API_BASE}/health`);
        if (!healthRes.ok) throw new Error("Health request failed");

        const healthJson: HealthResponse = await healthRes.json();
        setHealthInfo(healthJson);
        setBackendOk(true);

        // IMPORTANT:
        // We no longer prefill REPORT_YEAR into the visible profile.
        // The user can enter REPORT_YEAR manually later if they want.
        const initialProfile: Profile = { ...emptyProfile };
        setProfile(initialProfile);

        const initialOptions = await fetchOptionsForProfile(initialProfile);
        setOptions(initialOptions);

        queueAgentMessage(buildStepPromptMessage(STEPS[0], initialOptions));
      } catch (error) {
        setBackendOk(false);
      }
    }

    // Prevent duplicate boot in React StrictMode development mode
    if (hasBootedRef.current) return;
    hasBootedRef.current = true;

    boot();
  }, []);

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

      queueAgentMessage(
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
      queueAgentMessage(`Prediction failed: ${message}`);
    } finally {
      setIsPredicting(false);
    }
  }

  /**
   * Process one user message.
   *
   * Flow:
   * 1. Show the user bubble
   * 2. Handle reset / estimate
   * 3. Validate input for current step
   * 4. Save value into profile
   * 5. Fetch filtered options for the next step
   * 6. Queue the next prompt into the chat
   */
  async function processUserMessage(rawValue: string) {
    const raw = rawValue.trim();
    if (!raw) return;

    // Show user bubble first
    setMessages((prev) => [...prev, { role: "user", text: raw }]);

    // reset command
    if (/^reset$/i.test(raw)) {
      const resetProfile: Profile = { ...emptyProfile };

      setProfile(resetProfile);
      setResult(null);
      setCurrentStepIndex(0);

      const resetOptions = await fetchOptionsForProfile(resetProfile);
      setOptions(resetOptions);

      queueAgentMessage("All fields have been cleared. Let's start again.");
      queueAgentMessage(buildStepPromptMessage(STEPS[0], resetOptions));
      return;
    }

    // estimate command
    if (/^estimate$/i.test(raw)) {
      const missing = getMissingFields(profile);

      if (missing.length > 0) {
        queueAgentMessage(
          "I’m not ready to estimate yet. I still need:\n- " +
            missing.join("\n- ")
        );
        return;
      }

      await runPrediction(profile);
      return;
    }

    // all steps already completed
    if (!currentStep) {
      queueAgentMessage(
        "All steps have already been completed.\n\n" +
          "Type `estimate` to run prediction, or `reset` to start over."
      );
      return;
    }

    // validate current step
    const validation = validateStepInput(
      currentStep,
      raw,
      options,
      profile,
      healthInfo?.max_report_year ?? 2026,
      healthInfo?.min_report_year ?? 2020
    );

    if (!validation.ok) {
      queueAgentMessage(validation.message);
      return;
    }

    // save current value
    const nextProfile: Profile = {
      ...profile,
      [currentStep.field]: validation.value as never,
    };

    setProfile(nextProfile);
    setResult(null);

    const savedValue =
      validation.value === "" ? "Skipped" : String(validation.value);

    queueAgentMessage(`Saved ${currentStep.label} = ${savedValue}`);

    // move to next step
    if (currentStepIndex < STEPS.length - 1) {
      const nextStepIndex = currentStepIndex + 1;
      const nextOptions = await fetchOptionsForProfile(nextProfile);

      setOptions(nextOptions);
      setCurrentStepIndex(nextStepIndex);

      queueAgentMessage(buildStepPromptMessage(STEPS[nextStepIndex], nextOptions));
    } else {
      // all steps completed
      const finalOptions = await fetchOptionsForProfile(nextProfile);
      setOptions(finalOptions);
      setCurrentStepIndex(STEPS.length);

      queueAgentMessage(
        "All steps are complete. Type `estimate` when you are ready."
      );
    }
  }

  /**
   * Send from the chat input box.
   */
  async function handleSend() {
    const raw = input.trim();
    if (!raw) return;

    setInput("");
    await processUserMessage(raw);
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

        <div className="text-xs text-slate-500">API: {API_BASE}</div>
      </div>

      {/* Two-column Layout */}
      <div className="grid gap-6 lg:grid-cols-[1.25fr_0.95fr]">
        {/* Left: Chat only */}
        <div className="rounded-2xl border border-slate-200 bg-white shadow-sm">
          {/* Chat Header */}
          <div className="border-b border-slate-200 px-4 py-3">
            <div className="text-sm font-semibold">Conversation</div>
            <div className="text-xs text-slate-500">
              The chat stays simple. Suggested filtered choices are provided directly inside the conversation.
            </div>
          </div>

          {/* Current Step Banner */}
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
              </div>
            ) : (
              <div className="text-sm text-slate-700">
                All steps completed. Type <span className="font-medium">estimate</span> to run
                prediction, or <span className="font-medium">reset</span> to start over.
              </div>
            )}
          </div>

          {/* Messages Area */}
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

          {/* Chat Input */}
          <div className="border-t border-slate-200 p-3 space-y-2">
            <div className="text-xs text-slate-500">
              Useful commands:{" "}
              <span className="font-medium">estimate</span>,{" "}
              <span className="font-medium">reset</span>,{" "}
              <span className="font-medium">skip</span> (for optional steps).
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