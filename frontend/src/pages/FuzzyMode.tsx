import { useEffect, useState } from "react";

import { API_BASE } from "../config";

/**
 * ------------------------------------------------------------
 * 2. Basic Types
 * ------------------------------------------------------------
 */

/**
 * Health response from backend /health
 */
type HealthResponse = {
  ok: boolean;
  default_report_year: number;
  min_report_year: number;
  max_report_year: number;
};

/**
 * One fuzzy lookup candidate returned by /fuzzy_lookup
 */
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

/**
 * Full fuzzy lookup response
 */
type FuzzyLookupResponse = {
  match_count: number;
  auto_selected: boolean;
  used_report_year: number;
  postal_match_mode: "exact" | "fsa" | "none";
  candidates: FuzzyCandidate[];
};

/**
 * Prediction result returned by backend /predict
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
 * ------------------------------------------------------------
 * 3. Small Helper Functions
 * ------------------------------------------------------------
 */

/**
 * Format currency in CAD without decimals.
 */
function formatCurrency(value: number): string {
  return new Intl.NumberFormat("en-CA", {
    style: "currency",
    currency: "CAD",
    maximumFractionDigits: 0,
  }).format(value);
}

/**
 * Normalize postal code on frontend.
 *
 * - uppercase
 * - remove spaces
 * - remove dashes
 */
function normalizePostalCode(value: string): string {
  return value.trim().toUpperCase().replace(/\s|-/g, "");
}

/**
 * Convert nullable value into UI-friendly string.
 */
function valueOrDash(value: unknown): string {
  if (value === null || value === undefined || value === "") {
    return "—";
  }
  return String(value);
}

/**
 * Turn postal match mode into a human-friendly label.
 */
function getPostalMatchLabel(mode: "exact" | "fsa" | "none"): string {
  if (mode === "exact") return "Exact postal match";
  if (mode === "fsa") return "FSA-level postal match";
  return "No postal filter";
}

/**
 * ------------------------------------------------------------
 * 4. Main Component
 * ------------------------------------------------------------
 */
export default function FuzzyMode() {
  /**
   * ----------------------------------------------------------
   * 4.1 Basic backend / UI state
   * ----------------------------------------------------------
   */
  const [backendOk, setBackendOk] = useState<boolean | null>(null);
  const [healthInfo, setHealthInfo] = useState<HealthResponse | null>(null);

  /**
   * ----------------------------------------------------------
   * 4.2 Fuzzy input form state
   * ----------------------------------------------------------
   * User only needs to know:
   * - street number
   * - street name
   * - optional postal code
   * - optional report year
   */
  const [addressInput, setAddressInput] = useState("");
  const [postalCode, setPostalCode] = useState("");
  const [reportYearInput, setReportYearInput] = useState("");

  /**
   * ----------------------------------------------------------
   * 4.3 Lookup / candidate / prediction state
   * ----------------------------------------------------------
   */
  const [lookupResult, setLookupResult] = useState<FuzzyLookupResponse | null>(null);
  const [selectedCandidate, setSelectedCandidate] = useState<FuzzyCandidate | null>(null);
  const [result, setResult] = useState<PredictResult | null>(null);

  /**
   * ----------------------------------------------------------
   * 4.4 Loading / feedback state
   * ----------------------------------------------------------
   */
  const [isLookingUp, setIsLookingUp] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [statusMessage, setStatusMessage] = useState("");

  /**
   * ----------------------------------------------------------
   * 4.5 Load backend health on mount
   * ----------------------------------------------------------
   */
  useEffect(() => {
    async function loadHealth() {
      try {
        const res = await fetch(`${API_BASE}/health`);
        if (!res.ok) {
          throw new Error("Health request failed");
        }

        const json: HealthResponse = await res.json();
        setHealthInfo(json);
        setBackendOk(true);
      } catch {
        setBackendOk(false);
      }
    }

    loadHealth();
  }, []);

  /**
   * ----------------------------------------------------------
   * 4.6 Validate optional report year
   * ----------------------------------------------------------
   * Fuzzy mode also supports an optional report year.
   * If empty, backend will use its default report year.
   */
  function validateReportYear(): number | null {
    const raw = reportYearInput.trim();

    if (!raw) {
      return null;
    }

    const year = Number(raw);

    if (!Number.isInteger(year)) {
      throw new Error("Report Year should be a 4-digit integer year.");
    }

    if (healthInfo && (year < healthInfo.min_report_year || year > healthInfo.max_report_year)) {
      throw new Error(
        `Report Year should stay between ${healthInfo.min_report_year} and ${healthInfo.max_report_year}.`
      );
    }

    return year;
  }

  /**
   * ----------------------------------------------------------
   * 4.7 Reset all fuzzy-mode states
   * ----------------------------------------------------------
   */
  function resetAll() {
    setAddressInput("");
    setPostalCode("");
    setReportYearInput("");

    setLookupResult(null);
    setSelectedCandidate(null);
    setResult(null);

    setErrorMessage("");
    setStatusMessage("");
  }

  /**
   * ----------------------------------------------------------
   * 4.8 Call backend /fuzzy_lookup
   * ----------------------------------------------------------
   * This does NOT predict yet.
   * It only tries to resolve the partial address into one or more
   * candidate property rows.
   */
  async function handleFindProperty() {
    try {
      setErrorMessage("");
      setStatusMessage("");
      setResult(null);
      setSelectedCandidate(null);
      setLookupResult(null);

      const trimmedAddress = addressInput.trim();
      const normalizedPostal = postalCode.trim()
        ? normalizePostalCode(postalCode)
        : "";

      if (!trimmedAddress) {
        throw new Error("Enter a street address, for example 1050 26TH AVE W.");
      }

      // Split the address into its leading civic number and the street name,
      // so the user only has to type one natural "address" field.
      const addrMatch = trimmedAddress.match(/^(\d+[A-Za-z]?)\s+(.+)$/);
      if (!addrMatch) {
        throw new Error(
          "Start with the street number, then the street name — e.g. 1050 26TH AVE W."
        );
      }
      const trimmedStreetNumber = addrMatch[1];
      const trimmedStreetName = addrMatch[2].trim();

      const reportYear = validateReportYear();

      const params = new URLSearchParams();
      params.set("street_number", trimmedStreetNumber);
      params.set("street_name", trimmedStreetName);

      if (normalizedPostal) {
        params.set("property_postal_code", normalizedPostal);
      }

      if (reportYear !== null) {
        params.set("report_year", String(reportYear));
      }

      setIsLookingUp(true);

      const res = await fetch(`${API_BASE}/fuzzy_lookup?${params.toString()}`);
      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(errorText || "Fuzzy lookup failed");
      }

      const data: FuzzyLookupResponse = await res.json();
      setLookupResult(data);

      if (data.match_count === 0) {
        setStatusMessage("No match found. Check the street number and name, or try the postal code. Note this only covers City of Vancouver addresses (e.g. Burnaby or Richmond aren't included).");
        return;
      }

      if (data.auto_selected && data.candidates.length === 1) {
        setSelectedCandidate(data.candidates[0]);
        setStatusMessage("One property candidate was found and selected automatically.");
        return;
      }

      setStatusMessage(`Found ${data.match_count} property candidate(s). Please choose one.`);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Unknown fuzzy lookup error";
      setErrorMessage(message);
    } finally {
      setIsLookingUp(false);
    }
  }

  /**
   * ----------------------------------------------------------
   * 4.9 Select one candidate manually
   * ----------------------------------------------------------
   * This is used when fuzzy lookup returns multiple possible addresses.
   */
  function handleSelectCandidate(candidate: FuzzyCandidate) {
    setSelectedCandidate(candidate);
    setResult(null);
    setErrorMessage("");
    setStatusMessage(`Selected property: ${candidate.display_address}`);
  }

  /**
   * ----------------------------------------------------------
   * 4.10 Call backend /predict using the selected candidate
   * ----------------------------------------------------------
   * This is the key bridge:
   * fuzzy lookup resolves the address,
   * then reuse the SAME /predict logic as precise mode.
   */
  async function handleEstimate() {
    try {
      setErrorMessage("");
      setStatusMessage("");

      if (!selectedCandidate) {
        throw new Error("Please select a matched property first.");
      }

      if (selectedCandidate.YEAR_BUILT === null) {
        throw new Error("This matched property does not have a valid YEAR_BUILT value.");
      }

      setIsPredicting(true);
      setResult(null);

      const payload = {
        PROPERTY_POSTAL_CODE: selectedCandidate.PROPERTY_POSTAL_CODE,
        LEGAL_TYPE: selectedCandidate.LEGAL_TYPE,
        ZONING_DISTRICT: selectedCandidate.ZONING_DISTRICT,
        ZONING_CLASSIFICATION: selectedCandidate.ZONING_CLASSIFICATION,
        NEIGHBOURHOOD_CODE: selectedCandidate.NEIGHBOURHOOD_CODE,
        YEAR_BUILT: selectedCandidate.YEAR_BUILT,
        BIG_IMPROVEMENT_YEAR: selectedCandidate.BIG_IMPROVEMENT_YEAR,
        REPORT_YEAR: selectedCandidate.REPORT_YEAR,
        PID: selectedCandidate.PID,
      };

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
      setStatusMessage("Prediction completed successfully.");
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Unknown prediction error";
      setErrorMessage(message);
    } finally {
      setIsPredicting(false);
    }
  }

  return (
    <div className="space-y-6">
      {/* Page title */}
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">Search by address</h1>
      </div>

      {/* Backend status */}
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

      {/* Feedback message area */}
      {(statusMessage || errorMessage) && (
        <div
          className={[
            "rounded-2xl border px-4 py-3 text-sm shadow-sm",
            errorMessage
              ? "border-red-200 bg-red-50 text-red-700"
              : "border-slate-200 bg-white text-slate-700",
          ].join(" ")}
        >
          {errorMessage || statusMessage}
        </div>
      )}

      {/* Main two-column layout */}
      <div className="grid gap-6 lg:grid-cols-[1.25fr_0.95fr]">
        {/* Left side: address lookup + candidate list */}
        <div className="space-y-4">
          {/* Address form */}
          <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="mb-4">
              <h2 className="text-lg font-semibold">Address Lookup</h2>
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              <div className="sm:col-span-2">
                <label className="mb-2 block text-sm font-medium text-slate-700">
                  Street address
                </label>
                <input
                  value={addressInput}
                  onChange={(e) => setAddressInput(e.target.value)}
                  placeholder="Example: 1050 26TH AVE W"
                  className="h-11 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm outline-none focus:border-slate-300 focus:ring-2 focus:ring-slate-200"
                />
                <p className="mt-1.5 text-xs text-slate-500">
                  Just the street number and name — we'll fill in the rest for you.
                </p>
              </div>

              <div className="sm:col-span-1">
                <label className="mb-2 block text-sm font-medium text-slate-700">
                  Postal code{" "}
                  <span className="font-normal text-slate-400">(optional)</span>
                </label>
                <input
                  value={postalCode}
                  onChange={(e) => setPostalCode(e.target.value)}
                  placeholder="Example: V6H 2A5"
                  className="h-11 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm outline-none focus:border-slate-300 focus:ring-2 focus:ring-slate-200"
                />
              </div>

              <div className="sm:col-span-1">
                <label className="mb-2 block text-sm font-medium text-slate-700">
                  Year{" "}
                  <span className="font-normal text-slate-400">(optional)</span>
                </label>
                <input
                  value={reportYearInput}
                  onChange={(e) => setReportYearInput(e.target.value)}
                  placeholder={
                    healthInfo
                      ? `${healthInfo.min_report_year}–${healthInfo.max_report_year}`
                      : "Example: 2026"
                  }
                  className="h-11 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm outline-none focus:border-slate-300 focus:ring-2 focus:ring-slate-200"
                />
              </div>
            </div>

            <div className="mt-4 flex items-center gap-2">
              <button
                onClick={handleFindProperty}
                disabled={isLookingUp}
                className="h-11 rounded-xl bg-slate-900 px-4 text-sm font-semibold text-white hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {isLookingUp ? "Finding..." : "Find Property"}
              </button>

              <button
                onClick={resetAll}
                disabled={isLookingUp || isPredicting}
                className="h-11 rounded-xl border border-slate-200 bg-white px-4 text-sm font-semibold text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
              >
                Reset
              </button>
            </div>
          </div>

          {/* Candidate list */}
          <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="mb-4">
              <h2 className="text-lg font-semibold">Matched Candidates</h2>
              <p className="mt-1 text-sm text-slate-500">
                If multiple candidate properties are found, choose the one you want
                to estimate.
              </p>
            </div>

            {lookupResult ? (
              <div className="space-y-4">
                <div className="rounded-xl bg-slate-50 p-3 text-sm text-slate-700">
                  <div>
                    Match count: <span className="font-medium">{lookupResult.match_count}</span>
                  </div>
                  <div className="mt-1">
                    Used report year:{" "}
                    <span className="font-medium">{lookupResult.used_report_year}</span>
                  </div>
                  <div className="mt-1">
                    Postal match mode:{" "}
                    <span className="font-medium">
                      {getPostalMatchLabel(lookupResult.postal_match_mode)}
                    </span>
                  </div>
                </div>

                {lookupResult.candidates.length === 0 ? (
                  <div className="text-sm text-slate-600">
                    No candidates found yet.
                  </div>
                ) : (
                  <div className="space-y-3">
                    {lookupResult.candidates.map((candidate) => {
                      const isSelected =
                        selectedCandidate?.candidate_id === candidate.candidate_id;

                      return (
                        <div
                          key={candidate.candidate_id}
                          className={[
                            "rounded-xl border p-4",
                            isSelected
                              ? "border-slate-900 bg-slate-50"
                              : "border-slate-200 bg-white",
                          ].join(" ")}
                        >
                          <div className="flex items-start justify-between gap-3">
                            <div className="min-w-0">
                              <div className="font-medium text-slate-900">
                                {candidate.display_address}
                              </div>
                              <div className="mt-1 text-sm text-slate-500">
                                {candidate.LEGAL_TYPE} · {candidate.ZONING_DISTRICT} ·{" "}
                                {candidate.ZONING_CLASSIFICATION}
                              </div>
                              <div className="mt-1 text-sm text-slate-500">
                                Neighbourhood {candidate.NEIGHBOURHOOD_CODE} · Built{" "}
                                {valueOrDash(candidate.YEAR_BUILT)}
                              </div>
                            </div>

                            <button
                              onClick={() => handleSelectCandidate(candidate)}
                              className={[
                                "shrink-0 rounded-xl px-3 py-2 text-sm font-semibold",
                                isSelected
                                  ? "bg-slate-900 text-white"
                                  : "border border-slate-200 bg-white text-slate-700 hover:bg-slate-50",
                              ].join(" ")}
                            >
                              {isSelected ? "Selected" : "Use This"}
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            ) : (
              <div className="text-sm text-slate-600">
                No lookup has been performed yet.
              </div>
            )}
          </div>
        </div>

        {/* Right side: resolved profile + prediction */}
        <div className="space-y-4">
          {/* Resolved property profile */}
          <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="mb-4">
              <h2 className="text-lg font-semibold">What we found</h2>
              <p className="mt-1 text-sm text-slate-500">
                Pulled from city property records for the address you picked — you
                didn't have to know any of this.
              </p>
            </div>

            <div className="space-y-3 text-sm">
              <div className="flex items-start justify-between gap-3 border-b border-slate-100 pb-2">
                <span className="text-slate-500">Display Address</span>
                <span className="text-right font-medium text-slate-900">
                  {selectedCandidate ? selectedCandidate.display_address : "—"}
                </span>
              </div>

              <div className="flex items-start justify-between gap-3 border-b border-slate-100 pb-2">
                <span className="text-slate-500">Postal Code</span>
                <span className="text-right font-medium text-slate-900">
                  {valueOrDash(selectedCandidate?.PROPERTY_POSTAL_CODE)}
                </span>
              </div>

              <div className="flex items-start justify-between gap-3 border-b border-slate-100 pb-2">
                <span className="text-slate-500">Property type</span>
                <span className="text-right font-medium text-slate-900">
                  {valueOrDash(selectedCandidate?.LEGAL_TYPE)}
                </span>
              </div>

              <div className="flex items-start justify-between gap-3 border-b border-slate-100 pb-2">
                <span className="text-slate-500">Zoning District</span>
                <span className="text-right font-medium text-slate-900">
                  {valueOrDash(selectedCandidate?.ZONING_DISTRICT)}
                </span>
              </div>

              <div className="flex items-start justify-between gap-3 border-b border-slate-100 pb-2">
                <span className="text-slate-500">Zoning Classification</span>
                <span className="text-right font-medium text-slate-900">
                  {valueOrDash(selectedCandidate?.ZONING_CLASSIFICATION)}
                </span>
              </div>

              <div className="flex items-start justify-between gap-3 border-b border-slate-100 pb-2">
                <span className="text-slate-500">Neighbourhood</span>
                <span className="text-right font-medium text-slate-900">
                  {valueOrDash(selectedCandidate?.NEIGHBOURHOOD_CODE)}
                </span>
              </div>

              <div className="flex items-start justify-between gap-3 border-b border-slate-100 pb-2">
                <span className="text-slate-500">Year Built</span>
                <span className="text-right font-medium text-slate-900">
                  {valueOrDash(selectedCandidate?.YEAR_BUILT)}
                </span>
              </div>

              <div className="flex items-start justify-between gap-3 border-b border-slate-100 pb-2">
                <span className="text-slate-500">Last major improvement</span>
                <span className="text-right font-medium text-slate-900">
                  {valueOrDash(selectedCandidate?.BIG_IMPROVEMENT_YEAR)}
                </span>
              </div>

              <div className="flex items-start justify-between gap-3 border-b border-slate-100 pb-2">
                <span className="text-slate-500">Assessment year</span>
                <span className="text-right font-medium text-slate-900">
                  {valueOrDash(selectedCandidate?.REPORT_YEAR)}
                </span>
              </div>
            </div>

            <div className="mt-4">
              <button
                onClick={handleEstimate}
                disabled={!selectedCandidate || isPredicting}
                className="h-11 w-full rounded-xl bg-slate-900 px-4 text-sm font-semibold text-white hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {isPredicting ? "Estimating..." : "Estimate property value"}
              </button>
            </div>
          </div>

          {/* Estimated result */}
          <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="mb-4">
              <h2 className="text-lg font-semibold">Estimated Result</h2>
            </div>

            {isPredicting ? (
              <div className="text-sm text-slate-600">Running prediction...</div>
            ) : result ? (
              <div className="space-y-4">
                <div>
                  <div className="text-sm text-slate-500">
                    Estimated property value
                  </div>
                  <div className="mt-1 text-4xl font-semibold tracking-tight text-slate-900">
                    {formatCurrency(result.point_estimate)}
                  </div>
                </div>

                <div>
                  <div className="text-sm font-medium text-slate-800">
                    Likely range
                  </div>
                  <div className="mt-1 text-sm text-slate-700">
                    {formatCurrency(result.lower_bound)} to{" "}
                    {formatCurrency(result.upper_bound)}
                  </div>
                </div>

                <p className="text-sm text-slate-600">
                  This is a model estimate of the total assessed property value
                  (land plus building) — not a guaranteed sale price or an official
                  appraisal. The range reflects how much similar properties in this
                  area typically vary (about {formatCurrency(result.error_band)}{" "}
                  either way).
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
                No prediction yet. Please provide a valid address.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}