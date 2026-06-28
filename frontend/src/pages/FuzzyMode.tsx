import { useEffect, useState } from "react";

import { API_BASE } from "../config";

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

type ParsedAddress = { unit: string; streetNumber: string; streetName: string; postal: string; raw: string };

function formatCurrency(value: number): string {
  return new Intl.NumberFormat("en-CA", {
    style: "currency",
    currency: "CAD",
    maximumFractionDigits: 0,
  }).format(value);
}

function normalizePostalCode(value: string): string {
  return value.trim().toUpperCase().replace(/\s|-/g, "");
}

function valueOrDash(value: unknown): string {
  if (value === null || value === undefined || value === "") return "—";
  return String(value);
}

function parseAddress(text: string): ParsedAddress | null {
  const raw = text.trim();
  let addr = raw;
  let unit = "";
  // Optional "UNIT-BUILDING" prefix, e.g. "2301-1128 Hastings St W".
  const unitMatch = addr.match(/^(\d+[A-Za-z]?)\s*-\s*(\d.*)$/);
  if (unitMatch) {
    unit = unitMatch[1];
    addr = unitMatch[2].trim();
  }
  const match = addr.match(/^(\d+[A-Za-z]?)\s+(.+)$/);
  if (!match) return null;
  return { unit, streetNumber: match[1], streetName: match[2].trim(), postal: "", raw };
}

export default function FuzzyMode() {
  const [backendOk, setBackendOk] = useState<boolean | null>(null);
  const [addressInput, setAddressInput] = useState("");
  const [postalInput, setPostalInput] = useState("");
  const [unitInput, setUnitInput] = useState("");
  const [needsUnit, setNeedsUnit] = useState(false);
  const [selected, setSelected] = useState<ResolvedProperty | null>(null);
  const [result, setResult] = useState<PredictResult | null>(null);
  const [isBusy, setIsBusy] = useState(false);
  const [message, setMessage] = useState<{ text: string; error: boolean } | null>(null);

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

  function resetAll() {
    setAddressInput("");
    setPostalInput("");
    setUnitInput("");
    setNeedsUnit(false);
    setSelected(null);
    setResult(null);
    setMessage(null);
  }

  async function estimate(candidate: ResolvedProperty) {
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
      setMessage({ text: `Estimated ${candidate.display_address}.`, error: false });
    } catch (e) {
      setMessage({ text: `Estimate failed: ${e instanceof Error ? e.message : "unknown error"}`, error: true });
    } finally {
      setIsBusy(false);
    }
  }

  async function handleFind() {
    const parsed = parseAddress(addressInput);
    if (!parsed) {
      setMessage({
        text: "Start with the street number, then the street name — e.g. 1128 Hastings St W.",
        error: true,
      });
      return;
    }
    const effectiveUnit = parsed.unit || unitInput.trim();
    const postal = postalInput.trim() ? normalizePostalCode(postalInput) : "";

    setIsBusy(true);
    setMessage(null);
    setResult(null);
    setSelected(null);
    try {
      const params = new URLSearchParams();
      params.set("street_number", parsed.streetNumber);
      params.set("street_name", parsed.streetName);
      if (postal) params.set("property_postal_code", postal);
      if (effectiveUnit) params.set("unit", effectiveUnit);

      const res = await fetch(`${API_BASE}/resolve_address?${params.toString()}`);
      if (!res.ok) throw new Error(await res.text());
      const data: ResolveResponse = await res.json();

      if (data.status === "single" && data.candidate) {
        setNeedsUnit(false);
        setSelected(data.candidate);
        await estimate(data.candidate);
        return;
      }
      if (data.status === "need_unit") {
        setNeedsUnit(true);
        setMessage({
          text: `${parsed.raw} is a multi-unit building. Enter your unit number and search again.`,
          error: false,
        });
        return;
      }
      // none
      if (effectiveUnit) {
        setNeedsUnit(true);
        setMessage({ text: `Couldn't find unit ${effectiveUnit} at ${parsed.raw}. Try another unit number.`, error: true });
      } else {
        setNeedsUnit(false);
        setMessage({
          text:
            "Couldn't find that address. Check the street number and name, or add a postal code. " +
            "This only covers City of Vancouver addresses (Burnaby, Richmond, etc. aren't included).",
          error: true,
        });
      }
    } catch (e) {
      setMessage({ text: `Lookup failed: ${e instanceof Error ? e.message : "unknown error"}`, error: true });
    } finally {
      setIsBusy(false);
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">Search</h1>
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

      {message && (
        <div
          className={[
            "rounded-2xl border px-4 py-3 text-sm shadow-sm",
            message.error
              ? "border-red-200 bg-red-50 text-red-700"
              : "border-slate-200 bg-white text-slate-700",
          ].join(" ")}
        >
          {message.text}
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-[1.25fr_0.95fr]">
        {/* Left: address form */}
        <div className="self-start rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-4">
            <h2 className="text-lg font-semibold">Find a property</h2>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <div className="sm:col-span-2">
              <label className="mb-2 block text-sm font-medium text-slate-700">Street address</label>
              <input
                value={addressInput}
                onChange={(e) => setAddressInput(e.target.value)}
                placeholder="Example: 1128 Hastings St W"
                className="h-11 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm outline-none focus:border-slate-300 focus:ring-2 focus:ring-slate-200"
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleFind();
                }}
              />
              <p className="mt-1.5 text-xs text-slate-500">
                Just the street number and name — we'll fill in the rest for you.
              </p>
            </div>

            {needsUnit && (
              <div className="sm:col-span-1">
                <label className="mb-2 block text-sm font-medium text-slate-700">Unit number</label>
                <input
                  value={unitInput}
                  onChange={(e) => setUnitInput(e.target.value)}
                  placeholder="Example: 2308"
                  className="h-11 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm outline-none focus:border-slate-300 focus:ring-2 focus:ring-slate-200"
                  onKeyDown={(e) => {
                    if (e.key === "Enter") handleFind();
                  }}
                />
              </div>
            )}

            <div className="sm:col-span-1">
              <label className="mb-2 block text-sm font-medium text-slate-700">
                Postal code <span className="font-normal text-slate-400">(optional)</span>
              </label>
              <input
                value={postalInput}
                onChange={(e) => setPostalInput(e.target.value)}
                placeholder="Example: V6H 2A5"
                className="h-11 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm outline-none focus:border-slate-300 focus:ring-2 focus:ring-slate-200"
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleFind();
                }}
              />
            </div>
          </div>

          <div className="mt-4 flex items-center gap-2">
            <button
              onClick={handleFind}
              disabled={isBusy}
              className="h-11 rounded-xl bg-slate-900 px-4 text-sm font-semibold text-white hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isBusy ? "Working..." : needsUnit ? "Find unit" : "Find property"}
            </button>
            <button
              onClick={resetAll}
              disabled={isBusy}
              className="h-11 rounded-xl border border-slate-200 bg-white px-4 text-sm font-semibold text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Reset
            </button>
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
                  <span className="text-right font-medium text-slate-900">{valueOrDash(value)}</span>
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
                  <summary className="cursor-pointer font-medium text-slate-800">Technical details</summary>
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
                No estimate yet. Enter a Vancouver address and search.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
