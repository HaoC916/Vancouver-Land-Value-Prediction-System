import { useEffect, useState } from "react";

import { API_BASE } from "../config";

type MarketResult = {
  point_estimate: number;
  lower_bound: number;
  upper_bound: number;
  error_band: number;
  method: string;
  used_features: {
    property_type: string;
    bedrooms: number | null;
    bathrooms: number | null;
    floor_area_sqft: number | null;
    area_name: string | null;
    year_built: number | null;
  };
};

// Real areas the model covers (curated from the model's known areas — a few
// internal placeholder labels are omitted). The backend also fuzzy-matches, so
// this list is only an autocomplete hint.
const AREAS = [
  "Abbotsford", "Bowen Island", "Burnaby East", "Burnaby North", "Burnaby South",
  "Cloverdale", "Coquitlam", "Ladner", "Langley", "Maple Ridge", "Mission",
  "N. Delta", "New Westminster", "North Surrey", "North Vancouver", "Pemberton",
  "Pitt Meadows", "Port Coquitlam", "Port Moody", "Richmond",
  "South Surrey White Rock", "Squamish", "Sunshine Coast", "Surrey", "Tsawwassen",
  "Vancouver East", "Vancouver West", "West Vancouver", "Whistler",
];

const PROPERTY_TYPES = ["House", "Condo", "Townhouse"];

function formatCurrency(value: number): string {
  return new Intl.NumberFormat("en-CA", {
    style: "currency",
    currency: "CAD",
    maximumFractionDigits: 0,
  }).format(value);
}

export default function MarketMode() {
  const [backendOk, setBackendOk] = useState<boolean | null>(null);
  const [propertyType, setPropertyType] = useState("Condo");
  const [areaName, setAreaName] = useState("");
  const [bedrooms, setBedrooms] = useState("");
  const [bathrooms, setBathrooms] = useState("");
  const [floorArea, setFloorArea] = useState("");
  const [yearBuilt, setYearBuilt] = useState("");
  const [result, setResult] = useState<MarketResult | null>(null);
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
    setAreaName("");
    setBedrooms("");
    setBathrooms("");
    setFloorArea("");
    setYearBuilt("");
    setResult(null);
    setMessage(null);
  }

  async function estimate() {
    if (!areaName.trim()) {
      setMessage({ text: "Enter an area — e.g. Vancouver West, Richmond, or Surrey.", error: true });
      return;
    }
    setIsBusy(true);
    setMessage(null);
    setResult(null);
    try {
      const toNum = (s: string) => (s.trim() === "" ? null : Number(s));
      const body = {
        property_type: propertyType,
        area_name: areaName.trim(),
        bedrooms: toNum(bedrooms),
        bathrooms: toNum(bathrooms),
        floor_area_sqft: toNum(floorArea),
        year_built: toNum(yearBuilt),
      };
      const res = await fetch(`${API_BASE}/predict_market`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(await res.text());
      const data: MarketResult = await res.json();
      setResult(data);
      if (!data.used_features.area_name) {
        setMessage({
          text: `We didn't recognise "${areaName.trim()}" as a known area, so this uses a region-wide average. Try one of the listed areas for a sharper estimate.`,
          error: false,
        });
      }
    } catch (e) {
      setMessage({ text: `Estimate failed: ${e instanceof Error ? e.message : "unknown error"}`, error: true });
    } finally {
      setIsBusy(false);
    }
  }

  const inputClass =
    "h-11 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm outline-none focus:border-slate-300 focus:ring-2 focus:ring-slate-200";

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">Market price</h1>
        <p className="mt-1 text-sm text-slate-600">
          Estimate what a home would list for on the market today from its features —
          across Greater Vancouver and the Fraser Valley. (The Chat and Search tabs give
          the City of Vancouver assessed value by address; this is a market list price.)
        </p>
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
              : "border-amber-200 bg-amber-50 text-amber-800",
          ].join(" ")}
        >
          {message.text}
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-[1.25fr_0.95fr]">
        {/* Left: feature form */}
        <div className="self-start rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-4">
            <h2 className="text-lg font-semibold">Describe the home</h2>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <div className="sm:col-span-1">
              <label className="mb-2 block text-sm font-medium text-slate-700">Property type</label>
              <select value={propertyType} onChange={(e) => setPropertyType(e.target.value)} className={inputClass}>
                {PROPERTY_TYPES.map((t) => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
            </div>

            <div className="sm:col-span-1">
              <label className="mb-2 block text-sm font-medium text-slate-700">Area</label>
              <input
                list="market-areas"
                value={areaName}
                onChange={(e) => setAreaName(e.target.value)}
                placeholder="Example: Vancouver West"
                className={inputClass}
                onKeyDown={(e) => { if (e.key === "Enter") estimate(); }}
              />
              <datalist id="market-areas">
                {AREAS.map((a) => <option key={a} value={a} />)}
              </datalist>
            </div>

            <div className="sm:col-span-2">
              <label className="mb-2 block text-sm font-medium text-slate-700">
                Floor area (sq ft)
              </label>
              <input
                type="number"
                value={floorArea}
                onChange={(e) => setFloorArea(e.target.value)}
                placeholder="Example: 900"
                className={inputClass}
                onKeyDown={(e) => { if (e.key === "Enter") estimate(); }}
              />
              <p className="mt-1.5 text-xs text-slate-500">
                The single biggest driver of price — worth filling in.
              </p>
            </div>

            <div className="sm:col-span-1">
              <label className="mb-2 block text-sm font-medium text-slate-700">Bedrooms</label>
              <input type="number" value={bedrooms} onChange={(e) => setBedrooms(e.target.value)}
                placeholder="2" className={inputClass}
                onKeyDown={(e) => { if (e.key === "Enter") estimate(); }} />
            </div>

            <div className="sm:col-span-1">
              <label className="mb-2 block text-sm font-medium text-slate-700">Bathrooms</label>
              <input type="number" value={bathrooms} onChange={(e) => setBathrooms(e.target.value)}
                placeholder="2" className={inputClass}
                onKeyDown={(e) => { if (e.key === "Enter") estimate(); }} />
            </div>

            <div className="sm:col-span-1">
              <label className="mb-2 block text-sm font-medium text-slate-700">
                Year built <span className="font-normal text-slate-400">(optional)</span>
              </label>
              <input type="number" value={yearBuilt} onChange={(e) => setYearBuilt(e.target.value)}
                placeholder="2015" className={inputClass}
                onKeyDown={(e) => { if (e.key === "Enter") estimate(); }} />
            </div>
          </div>

          <div className="mt-4 flex items-center gap-2">
            <button
              onClick={estimate}
              disabled={isBusy}
              className="h-11 rounded-xl bg-slate-900 px-4 text-sm font-semibold text-white hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isBusy ? "Working..." : "Estimate market price"}
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

        {/* Right: result */}
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-4">
            <h2 className="text-lg font-semibold">Estimated market price</h2>
          </div>
          {isBusy && !result ? (
            <div className="text-sm text-slate-600">Running estimate...</div>
          ) : result ? (
            <div className="space-y-4">
              <div>
                <div className="text-sm text-slate-500">Estimated list price</div>
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
                This is a model estimate of the market <span className="font-medium">list</span>{" "}
                price — not a guaranteed sale price or an appraisal. The range reflects how much
                similar homes typically vary (about {formatCurrency(result.error_band)} either way).
              </p>
              <details className="text-sm text-slate-600">
                <summary className="cursor-pointer font-medium text-slate-800">Technical details</summary>
                <pre className="mt-2 max-h-48 overflow-auto rounded-xl bg-slate-50 p-3 text-xs text-slate-700">
                  {JSON.stringify({ method: result.method, ...result.used_features }, null, 2)}
                </pre>
              </details>
            </div>
          ) : (
            <div className="text-sm text-slate-600">
              No estimate yet. Describe a home and choose an area, then estimate.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
