import { useEffect, useMemo, useRef, useState } from "react";
import L from "leaflet";
import type { Feature, FeatureCollection, Geometry } from "geojson";

import "leaflet/dist/leaflet.css";
import "./MapMode.css";
import { API_BASE } from "../config";

type Level = "municipalities" | "communities";
type PropertyType = "APTU" | "HOUSE" | "TWIN";
type Metric = "price" | "livability" | "change";

type MarketRow = {
  typical_price_cad: number | null;
  sales_last_12m: number | null;
  median_days_on_market: number | null;
  price_change_pct_year_over_year: number | null;
};

type MapProperties = {
  region_id: string;
  name: string;
  area?: string;
  municipality?: string;
  municipality_id?: string;
  community_count?: number;
  display_community_count?: number;
  geometry_source: "modified_geom" | "modified_geom_union";
  market: Partial<Record<PropertyType, MarketRow>>;
  livability_score: number | null;
  amenity_score?: number | null;
  transit_score?: number | null;
  safety_score?: number | null;
  best_school_score?: number | null;
};

type MapCollection = FeatureCollection<Geometry, MapProperties> & {
  metadata?: {
    geometry_source?: string;
    excluded_missing_modified?: string[];
  };
};

type MapModeProps = {
  onAsk: (text: string) => void;
};

const PROPERTY_LABELS: Record<PropertyType, string> = {
  APTU: "Condo",
  HOUSE: "House",
  TWIN: "Townhouse",
};

const PRICE_PALETTE = ["#e8f2ff", "#c9ddf7", "#9fc2e8", "#6d9fcf", "#386fa4", "#18466f"];
const SCORE_PALETTE = ["#e7f6f2", "#c6eadf", "#91d4c1", "#58b59e", "#248772", "#0c594b"];

function formatCurrency(value: number | null | undefined) {
  if (value == null) return "No data";
  return new Intl.NumberFormat("en-CA", {
    style: "currency",
    currency: "CAD",
    maximumFractionDigits: 0,
  }).format(value);
}

function metricValue(properties: MapProperties, propertyType: PropertyType, metric: Metric) {
  if (metric === "livability") return properties.livability_score;
  const market = properties.market[propertyType];
  return metric === "change"
    ? market?.price_change_pct_year_over_year ?? null
    : market?.typical_price_cad ?? null;
}

function metricLabel(value: number | null, metric: Metric) {
  if (value == null) return "No data";
  if (metric === "price") return formatCurrency(value);
  if (metric === "change") return `${value > 0 ? "+" : ""}${value.toFixed(1)}% YoY`;
  return `${value.toFixed(1)} / 100`;
}

function fillColor(value: number | null, metric: Metric, min: number, max: number) {
  if (value == null) return "#d9dee5";
  if (metric === "change") {
    if (value <= -5) return "#b94a48";
    if (value < 0) return "#e69c73";
    if (value < 5) return "#f0d58b";
    if (value < 10) return "#91c9a7";
    return "#287a58";
  }
  const palette = metric === "price" ? PRICE_PALETTE : SCORE_PALETTE;
  const ratio = max === min ? 0.5 : Math.max(0, Math.min(1, (value - min) / (max - min)));
  return palette[Math.min(palette.length - 1, Math.floor(ratio * palette.length))];
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl bg-slate-50 px-3 py-2.5">
      <div className="text-[11px] uppercase tracking-wide text-slate-400">{label}</div>
      <div className="mt-1 text-sm font-semibold text-slate-800">{value}</div>
    </div>
  );
}

export default function MapMode({ onAsk }: MapModeProps) {
  const mapNodeRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<L.Map | null>(null);
  const layerRef = useRef<L.GeoJSON | null>(null);
  const selectedElementRef = useRef<Element | null>(null);
  const [level, setLevel] = useState<Level>("municipalities");
  const [propertyType, setPropertyType] = useState<PropertyType>("APTU");
  const [metric, setMetric] = useState<Metric>("price");
  const [cityFilter, setCityFilter] = useState<string | null>(null);
  const [data, setData] = useState<MapCollection | null>(null);
  const [selected, setSelected] = useState<Feature<Geometry, MapProperties> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!mapNodeRef.current || mapRef.current) return;
    const map = L.map(mapNodeRef.current, { zoomControl: false, minZoom: 7 }).setView([49.15, -122.75], 9);
    L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
      maxZoom: 19,
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
    }).addTo(map);
    L.control.zoom({ position: "bottomright" }).addTo(map);
    mapRef.current = map;
    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    const params = level === "communities" && cityFilter
      ? `?municipality=${encodeURIComponent(cityFilter)}`
      : "";
    queueMicrotask(() => {
      if (controller.signal.aborted) return;
      setLoading(true);
      setError(null);
      setSelected(null);
    });
    fetch(`${API_BASE}/map/${level}${params}`, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`Map request failed (${response.status})`);
        return response.json() as Promise<MapCollection>;
      })
      .then(setData)
      .catch((reason: unknown) => {
        if (reason instanceof DOMException && reason.name === "AbortError") return;
        setError(reason instanceof Error ? reason.message : "Unable to load map data");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [level, cityFilter]);

  const domain = useMemo(() => {
    const values = (data?.features ?? [])
      .map((feature) => metricValue(feature.properties, propertyType, metric))
      .filter((value): value is number => value != null && Number.isFinite(value));
    return {
      min: values.length ? Math.min(...values) : 0,
      max: values.length ? Math.max(...values) : 1,
    };
  }, [data, metric, propertyType]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !data) return;
    layerRef.current?.remove();
    const layer = L.geoJSON(data, {
      style: (feature) => {
        const properties = feature?.properties as MapProperties | undefined;
        const value = properties ? metricValue(properties, propertyType, metric) : null;
        return {
          color: "#ffffff",
          weight: level === "municipalities" ? 1.5 : 1,
          fillColor: fillColor(value, metric, domain.min, domain.max),
          fillOpacity: 0.78,
        };
      },
      onEachFeature: (feature, featureLayer) => {
        const typed = feature as Feature<Geometry, MapProperties>;
        const value = metricValue(typed.properties, propertyType, metric);
        const selectFeature = () => {
          selectedElementRef.current?.classList.remove("map-area-selected");
          const element = featureLayer instanceof L.Path ? featureLayer.getElement() ?? null : null;
          element?.classList.add("map-area-selected");
          selectedElementRef.current = element;
          setSelected(typed);
          if (featureLayer instanceof L.Path) featureLayer.bringToFront();
        };
        featureLayer.bindTooltip(
          `<strong>${typed.properties.name}</strong><br>${metricLabel(value, metric)}`,
          { sticky: true, className: "map-tooltip" },
        );
        featureLayer.on("add", () => {
          if (!(featureLayer instanceof L.Path)) return;
          const element = featureLayer.getElement();
          if (!element) return;
          element.setAttribute("data-map-name", typed.properties.name);
          element.setAttribute("role", "button");
          element.setAttribute("tabindex", "0");
          element.setAttribute("aria-label", `Select ${typed.properties.name}`);
          element.addEventListener("keydown", (event) => {
            if (event.key !== "Enter" && event.key !== " ") return;
            event.preventDefault();
            selectFeature();
          });
        });
        featureLayer.on({
          click: selectFeature,
          mouseover: () => {
            if (featureLayer instanceof L.Path) {
              featureLayer.setStyle({ weight: 2.5, color: "#334155", fillOpacity: 0.9 });
            }
          },
          mouseout: () => layer.resetStyle(featureLayer),
        });
      },
    }).addTo(map);
    layerRef.current = layer;
    const bounds = layer.getBounds();
    if (bounds.isValid()) map.fitBounds(bounds, { padding: [18, 18], maxZoom: cityFilter ? 12 : 10 });
    return () => {
      selectedElementRef.current?.classList.remove("map-area-selected");
      selectedElementRef.current = null;
      layer.remove();
      if (layerRef.current === layer) layerRef.current = null;
    };
  }, [data, domain.max, domain.min, level, metric, propertyType, cityFilter]);

  function changeLevel(next: Level) {
    setLevel(next);
    setCityFilter(null);
    if (next === "communities" && metric === "change") setMetric("price");
  }

  function exploreCommunities() {
    if (!selected) return;
    setCityFilter(selected.properties.name);
    setLevel("communities");
    if (metric === "change") setMetric("price");
  }

  const selectedMarket = selected?.properties.market[propertyType];
  const entityName = selected?.properties.name ?? "Select an area";
  const entityContext = level === "communities"
    ? selected?.properties.municipality ?? selected?.properties.area
    : selected
      ? selected.properties.display_community_count != null
        && selected.properties.display_community_count !== selected.properties.community_count
        ? `${selected.properties.display_community_count} urban communities shown · ${selected.properties.community_count ?? 0} market communities`
        : `${selected.properties.community_count ?? 0} market communities`
      : "Click a boundary to compare it";
  const askText = selected
    ? `Tell me about ${selected.properties.name}${level === "communities" && selected.properties.municipality ? ` in ${selected.properties.municipality}` : ""} for buying a ${PROPERTY_LABELS[propertyType].toLowerCase()}. Include price, market trend, livability, transit and schools.`
    : "Help me choose an area in Greater Vancouver based on budget, commute and livability.";

  return (
    <section className="space-y-5">
      <div className="flex flex-col gap-4 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm lg:flex-row lg:items-end lg:justify-between">
        <div>
          <div className="text-xs font-semibold uppercase tracking-[0.16em] text-blue-700">Explore the region</div>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight">City & Community Map</h1>
        </div>
        <div className="flex flex-wrap gap-3">
          <label className="text-xs font-medium text-slate-500">
            Property type
            <select value={propertyType} onChange={(event) => setPropertyType(event.target.value as PropertyType)} className="mt-1 block rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-800 outline-none focus:ring-2 focus:ring-slate-200">
              {Object.entries(PROPERTY_LABELS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
            </select>
          </label>
          <label className="text-xs font-medium text-slate-500">
            Colour by
            <select value={metric} onChange={(event) => setMetric(event.target.value as Metric)} className="mt-1 block rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-800 outline-none focus:ring-2 focus:ring-slate-200">
              <option value="price">Typical price</option>
              <option value="livability">Livability</option>
              {level === "municipalities" && <option value="change">Historical change</option>}
            </select>
          </label>
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="inline-flex rounded-full border border-slate-200 bg-white p-1 shadow-sm">
          <button type="button" onClick={() => changeLevel("municipalities")} className={`rounded-full px-4 py-2 text-sm font-medium transition ${level === "municipalities" ? "bg-slate-900 text-white" : "text-slate-600 hover:bg-slate-50"}`}>Cities</button>
          <button type="button" onClick={() => changeLevel("communities")} className={`rounded-full px-4 py-2 text-sm font-medium transition ${level === "communities" ? "bg-slate-900 text-white" : "text-slate-600 hover:bg-slate-50"}`}>Communities</button>
        </div>
        <div className="text-xs text-slate-500">
          {level === "municipalities" ? "21 city areas" : `${data?.features.length ?? 357} refined community boundaries`} · modified geometry only
        </div>
      </div>

      {cityFilter && (
        <button type="button" onClick={() => changeLevel("municipalities")} className="text-sm font-medium text-blue-700 hover:text-blue-900">
          ← Back to Cities · viewing {cityFilter}
        </button>
      )}

      <div className="map-layout overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
        <div className="relative min-w-0">
          <div ref={mapNodeRef} className="map-canvas" aria-label="Interactive Greater Vancouver city and community map" />
          {loading && <div className="map-status">Loading refined boundaries…</div>}
          {error && <div className="map-status map-status-error">{error}</div>}
          <div className="map-legend">
            <div className="font-semibold text-slate-700">{metric === "price" ? "Typical price" : metric === "change" ? "Year-over-year" : "Livability score"}</div>
            <div className={`mt-2 h-2.5 rounded-full ${metric === "change" ? "legend-change" : metric === "price" ? "legend-price" : "legend-score"}`} />
            <div className="mt-1 flex justify-between text-[10px] text-slate-500">
              <span>{metricLabel(domain.min, metric)}</span><span>{metricLabel(domain.max, metric)}</span>
            </div>
          </div>
        </div>

        <aside className="map-detail-panel border-t border-slate-200 p-5 lg:border-l lg:border-t-0">
          <div className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-400">{level === "municipalities" ? "City overview" : "Community overview"}</div>
          <h2 className="mt-2 text-xl font-semibold text-slate-900">{entityName}</h2>
          <p className="mt-1 text-sm text-slate-500">{entityContext}</p>

          {selected ? (
            <>
              <div className="mt-5 grid grid-cols-2 gap-2">
                <Stat label={`${PROPERTY_LABELS[propertyType]} price`} value={formatCurrency(selectedMarket?.typical_price_cad)} />
                <Stat label="Sales · 12m" value={selectedMarket?.sales_last_12m?.toLocaleString("en-CA") ?? "No data"} />
                <Stat label="Median DOM" value={selectedMarket?.median_days_on_market != null ? `${selectedMarket.median_days_on_market} days` : "No data"} />
                <Stat label="YoY change" value={selectedMarket?.price_change_pct_year_over_year != null ? `${selectedMarket.price_change_pct_year_over_year > 0 ? "+" : ""}${selectedMarket.price_change_pct_year_over_year.toFixed(1)}%` : "No data"} />
              </div>

              <div className="mt-5">
                <div className="flex items-center justify-between text-sm"><span className="font-medium text-slate-700">Livability</span><span className="font-semibold text-slate-900">{selected.properties.livability_score?.toFixed(1) ?? "—"}</span></div>
                <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-100"><div className="h-full rounded-full bg-emerald-500" style={{ width: `${Math.max(0, Math.min(100, selected.properties.livability_score ?? 0))}%` }} /></div>
              </div>

              {level === "communities" && (
                <div className="mt-4 flex flex-wrap gap-2 text-xs">
                  {[['Amenities', selected.properties.amenity_score], ['Transit', selected.properties.transit_score], ['Safety', selected.properties.safety_score], ['Schools', selected.properties.best_school_score]].map(([label, value]) => (
                    <span key={String(label)} className="rounded-full border border-slate-200 bg-white px-2.5 py-1.5 text-slate-600">{label} {typeof value === "number" ? value.toFixed(0) : "—"}</span>
                  ))}
                </div>
              )}

              {level === "municipalities" && <button type="button" onClick={exploreCommunities} className="mt-5 w-full rounded-xl border border-slate-300 bg-white px-4 py-2.5 text-sm font-semibold text-slate-700 transition hover:bg-slate-50">Explore communities</button>}
            </>
          ) : (
            <div className="mt-6 rounded-xl border border-dashed border-slate-200 bg-slate-50 p-4 text-sm leading-relaxed text-slate-500">Select a coloured boundary to see its market snapshot and livability profile.</div>
          )}

          <button type="button" onClick={() => onAsk(askText)} className="mt-4 w-full rounded-xl bg-slate-900 px-4 py-3 text-sm font-semibold text-white transition hover:bg-slate-800">
            Ask in Chat
          </button>
          <p className="mt-3 text-[11px] leading-relaxed text-slate-400">Boundaries use modified geometry only. Kawkawa Lake is omitted because no modified boundary is available.</p>
        </aside>
      </div>
    </section>
  );
}
