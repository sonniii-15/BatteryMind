import React, { useState, useEffect, useMemo, useCallback, Component } from "react";
import {
  Activity,
  AlertOctagon,
  AlertTriangle,
  Battery,
  BatteryCharging,
  CheckCircle2,
  Cpu,
  Flame,
  Gauge,
  Info,
  Layers,
  RefreshCw,
  ShieldAlert,
  Sliders,
  Sparkles,
  Thermometer,
  TrendingDown,
  Zap,
} from "lucide-react";
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine,
  BarChart,
  Bar,
  Cell,
} from "recharts";

// -----------------------------------------------------------------------------
// Safe Formatting Helpers
// -----------------------------------------------------------------------------
function formatNum(val, decimals = 1, fallback = "--") {
  if (val === null || val === undefined || isNaN(Number(val))) return fallback;
  return Number(val).toFixed(decimals);
}

function safeRound(val, fallback = 0) {
  if (val === null || val === undefined || isNaN(Number(val))) return fallback;
  return Math.round(Number(val));
}

// -----------------------------------------------------------------------------
// Constants & Domain Profiles
// -----------------------------------------------------------------------------
const API_BASE = "http://localhost:8000";

const DOMAINS = {
  ev: {
    id: "ev",
    name: "EV Mobility",
    eolThreshold: 80.0,
    eolCapacity: 1.60,
    label: "EV Mobility (80% EOL)",
    desc: "Strict automotive safety & acceleration criteria (retirement at 80% SOH)",
    badgeColor: "border-cyan-500/40 text-cyan-400 bg-cyan-500/10",
  },
  aerospace: {
    id: "aerospace",
    name: "Aerospace / Satellite",
    eolThreshold: 85.0,
    eolCapacity: 1.70,
    label: "Aerospace / Satellite (85% EOL)",
    desc: "Ultra-conservative mission margins; zero tolerance for unexpected orbital loss",
    badgeColor: "border-indigo-500/40 text-indigo-400 bg-indigo-500/10",
  },
  grid: {
    id: "grid",
    name: "Grid Storage BESS",
    eolThreshold: 70.0,
    eolCapacity: 1.40,
    label: "Grid Storage BESS (70% EOL)",
    desc: "Stationary utility storage maximized for throughput life down to 70% SOH",
    badgeColor: "border-emerald-500/40 text-emerald-400 bg-emerald-500/10",
  },
};

const CELLS = [
  { id: "B0005", label: "B0005", isTest: false },
  { id: "B0006", label: "B0006", isTest: false },
  { id: "B0007", label: "B0007", isTest: false },
  { id: "B0018", label: "B0018", isTest: true },
];

const PRESETS = {
  healthy: {
    name: "Healthy Nominal",
    icon: CheckCircle2,
    color: "emerald",
    values: {
      Cycle_Count: 25,
      Ambient_Temp_C: 24.1,
      Avg_Voltage_V: 3.82,
      Avg_Current_A: 2.00,
      Max_Temp_C: 32.5,
      Internal_Resistance_Ohm: 0.076,
      Discharge_Capacity_Ah: 1.84,
    },
  },
  runaway: {
    name: "Thermal Runaway Risk",
    icon: Flame,
    color: "rose",
    values: {
      Cycle_Count: 110,
      Ambient_Temp_C: 24.2,
      Avg_Voltage_V: 3.32,
      Avg_Current_A: 2.02,
      Max_Temp_C: 51.5,
      Internal_Resistance_Ohm: 0.215,
      Discharge_Capacity_Ah: 1.32,
    },
  },
  aged: {
    name: "Aged Cell",
    icon: Battery,
    color: "amber",
    values: {
      Cycle_Count: 140,
      Ambient_Temp_C: 24.0,
      Avg_Voltage_V: 3.52,
      Avg_Current_A: 1.99,
      Max_Temp_C: 41.8,
      Internal_Resistance_Ohm: 0.198,
      Discharge_Capacity_Ah: 1.37,
    },
  },
};

// Robust default fallback when backend is disconnected
const MOCK_HEALTH = {
  status: "healthy",
  model_architecture: "XGBoost Multi-Target Regressor",
  test_cell: "B0018",
  metrics: {
    Remaining_Useful_Life_RUL: { rmse: 23.16, mae: 20.25, r2: 0.5420 },
    SOH_Percent: { rmse: 1.34, mae: 1.13, r2: 0.9772 },
  },
  anomaly_detection: {
    test_anomalies_detected: 3,
    anomalous_cycles: [100, 124, 131],
  },
};

// Defensive cycle generator for offline/fallback mode
function generateFallbackCycles(cellId) {
  const count = cellId === "B0018" ? 132 : 168;
  const initialCap = cellId === "B0006" ? 2.03 : 1.86;
  const items = [];
  const anomalyCycles = cellId === "B0018" ? [100, 124, 131] : [35, 72, 110];

  for (let c = 1; c <= count; c++) {
    const fade = 0.0028 * c + 0.08 * Math.pow(c / count, 2);
    const cap = Math.max(1.22, initialCap - fade);
    const soh = (cap / 2.0) * 100;
    const isAnomaly = anomalyCycles.includes(c);
    const maxTemp = isAnomaly ? 48.5 : 32.0 + 8.5 * (c / count);
    const rul = Math.max(0, Math.round(count * 0.82 - c));

    items.push({
      cycle_count: c,
      discharge_capacity_ah: Number(cap.toFixed(4)),
      soh_percent: Number(soh.toFixed(2)),
      actual_soh: Number(soh.toFixed(2)),
      actual_rul: rul,
      predicted_rul: Math.max(0, rul + Math.sin(c) * 4),
      predicted_soh: Number((soh + Math.sin(c * 0.2) * 0.4).toFixed(2)),
      is_anomaly: isAnomaly,
      max_temp_c: Number(maxTemp.toFixed(1)),
      internal_resistance_ohm: Number((0.075 + 0.0008 * c).toFixed(4)),
      avg_voltage_v: Number((3.82 - 0.0018 * c).toFixed(4)),
      avg_current_a: 2.0,
      ambient_temp_c: 24.0,
    });
  }
  return {
    battery_id: cellId,
    total_cycles: count,
    anomaly_count: anomalyCycles.length,
    anomalous_cycles: anomalyCycles,
    summary: {
      initial_capacity_ah: initialCap,
      final_capacity_ah: items[items.length - 1].discharge_capacity_ah,
      initial_soh_percent: Number(((initialCap / 2.0) * 100).toFixed(1)),
      final_soh_percent: items[items.length - 1].soh_percent,
      nominal_rated_capacity_ah: 2.0,
    },
    cycles: items,
  };
}

// -----------------------------------------------------------------------------
// React Error Boundary Component
// -----------------------------------------------------------------------------
class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("BatteryMind ErrorBoundary caught:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-[#090D16] text-slate-100 p-8 flex flex-col items-center justify-center space-y-4">
          <div className="p-4 rounded-2xl bg-rose-500/10 border border-rose-500/30 text-rose-400 max-w-lg text-center space-y-3">
            <AlertOctagon className="w-12 h-12 mx-auto text-rose-400" />
            <h2 className="text-xl font-bold">Telemetry Console Recovered</h2>
            <p className="text-xs text-slate-300">
              An unexpected render anomaly was caught and neutralized.
            </p>
            <pre className="text-[11px] font-mono bg-slate-950 p-2.5 rounded text-rose-300 overflow-x-auto text-left">
              {String(this.state.error?.message || "Unknown error")}
            </pre>
            <button
              onClick={() => {
                this.setState({ hasError: false, error: null });
                window.location.reload();
              }}
              className="px-4 py-2 text-xs font-bold rounded-lg bg-cyan-500 text-slate-950 hover:bg-cyan-400 transition"
            >
              Reset &amp; Reload Console
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

// -----------------------------------------------------------------------------
// Main Application Component
// -----------------------------------------------------------------------------
function BatteryMindDashboard() {
  // Navigation & Domain Context State
  const [activeDomain, setActiveDomain] = useState("ev");
  const [selectedCell, setSelectedCell] = useState("B0018");

  // Live Data States
  const [batteryData, setBatteryData] = useState(null);
  const [healthData, setHealthData] = useState(MOCK_HEALTH);
  const [globalShap, setGlobalShap] = useState(null);
  const [apiConnected, setApiConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [fetchError, setFetchError] = useState(null);

  // Live Telemetry Simulator State
  const [simValues, setSimValues] = useState(PRESETS.healthy.values);
  const [simResult, setSimResult] = useState({
    predicted_rul: 114.2,
    predicted_soh_percent: 92.5,
    is_anomaly: false,
    anomaly_score: 0.1245,
    base_rul: 69.12,
    shap_contributions: {
      Internal_Resistance_Ohm: 24.5,
      Discharge_Capacity_Ah: 18.2,
      Cycle_Count: 4.8,
      Max_Temp_C: 1.2,
      Avg_Voltage_V: 0.8,
      Avg_Current_A: 0.2,
      Ambient_Temp_C: 0.1,
    },
    prediction_summary: "Nominal cell operation. Low impedance and high retained capacity support healthy prognostic life.",
  });
  const [isSimulating, setIsSimulating] = useState(false);

  // Connection & Health check on mount
  const checkHealth = useCallback(async () => {
    try {
      const [resHealth, resShap] = await Promise.all([
        fetch(`${API_BASE}/health`),
        fetch(`${API_BASE}/explain/global`),
      ]);

      if (resHealth.ok) {
        const hData = await resHealth.json();
        setHealthData(hData);
        setApiConnected(true);
        setFetchError(null);
      }
      if (resShap.ok) {
        const sData = await resShap.json();
        setGlobalShap(sData);
      }
    } catch (err) {
      console.warn("Backend not detected at localhost:8000; using standalone NASA telemetry engine.", err);
      setApiConnected(false);
    }
  }, []);

  useEffect(() => {
    checkHealth();
  }, [checkHealth]);

  // Fetch Battery Data when selectedCell changes
  useEffect(() => {
    let isCancelled = false;
    async function fetchBattery() {
      setIsLoading(true);
      try {
        const res = await fetch(`${API_BASE}/batteries/${selectedCell}`);
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        const data = await res.json();
        if (!isCancelled) {
          // Normalize schema defensively so no property access ever returns undefined
          const normalizedCycles = Array.isArray(data?.cycles)
            ? data.cycles.map((c) => {
                const sohVal = c.soh_percent ?? c.actual_soh ?? c.SOH_Percent ?? 0;
                const capVal = c.discharge_capacity_ah ?? c.Discharge_Capacity_Ah ?? 0;
                const rulVal = c.actual_rul ?? c.Remaining_Useful_Life_RUL ?? 0;
                const predRul = c.predicted_rul ?? rulVal;
                const predSoh = c.predicted_soh ?? sohVal;

                return {
                  cycle_count: Number(c.cycle_count ?? c.Cycle_Count ?? 1),
                  soh_percent: Number(sohVal),
                  actual_soh: Number(sohVal),
                  predicted_soh: Number(predSoh),
                  discharge_capacity_ah: Number(capVal),
                  actual_rul: Number(rulVal),
                  predicted_rul: Number(predRul),
                  is_anomaly: Boolean(c.is_anomaly),
                  anomaly_score: Number(c.anomaly_score ?? 0),
                  max_temp_c: Number(c.max_temp_c ?? c.Max_Temp_C ?? 0),
                  internal_resistance_ohm: Number(c.internal_resistance_ohm ?? c.Internal_Resistance_Ohm ?? 0),
                  avg_voltage_v: Number(c.avg_voltage_v ?? c.Avg_Voltage_V ?? 0),
                  avg_current_a: Number(c.avg_current_a ?? c.Avg_Current_A ?? 0),
                  ambient_temp_c: Number(c.ambient_temp_c ?? c.Ambient_Temp_C ?? 24),
                };
              })
            : [];

          setBatteryData({
            ...data,
            cycles: normalizedCycles,
          });
          setApiConnected(true);
          setFetchError(null);
        }
      } catch (err) {
        if (!isCancelled) {
          console.info(`Using fallback telemetry for ${selectedCell}:`, err.message);
          setBatteryData(generateFallbackCycles(selectedCell));
        }
      } finally {
        if (!isCancelled) setIsLoading(false);
      }
    }
    fetchBattery();
    return () => {
      isCancelled = true;
    };
  }, [selectedCell]);

  // Simulator Prediction Execution
  const executePrediction = useCallback(async (params) => {
    setIsSimulating(true);
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          Cycle_Count: Number(params.Cycle_Count),
          Ambient_Temp_C: Number(params.Ambient_Temp_C),
          Avg_Voltage_V: Number(params.Avg_Voltage_V),
          Avg_Current_A: Number(params.Avg_Current_A),
          Max_Temp_C: Number(params.Max_Temp_C),
          Internal_Resistance_Ohm: Number(params.Internal_Resistance_Ohm),
          Discharge_Capacity_Ah: Number(params.Discharge_Capacity_Ah),
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setSimResult({
          ...data,
          predicted_rul: Number(data.predicted_rul ?? 0),
          predicted_soh_percent: Number(data.predicted_soh_percent ?? 0),
          anomaly_score: Number(data.anomaly_score ?? 0),
          shap_contributions: data.shap_contributions || {},
        });
        setApiConnected(true);
      } else {
        throw new Error("API call error");
      }
    } catch {
      // Local physics fallback simulation
      const cap = Number(params.Discharge_Capacity_Ah || 1.8);
      const resVal = Number(params.Internal_Resistance_Ohm || 0.08);
      const temp = Number(params.Max_Temp_C || 32);
      const cyc = Number(params.Cycle_Count || 25);
      const volt = Number(params.Avg_Voltage_V || 3.8);

      const soh = (cap / 2.0) * 100;
      const isAnom = temp > 46.0 || resVal > 0.20 || volt < 3.40;
      const eolCycles = 140;
      const rul = Math.max(0, Math.round(eolCycles - cyc - (resVal - 0.08) * 400 + (cap - 1.4) * 80));

      setSimResult({
        predicted_rul: rul,
        predicted_soh_percent: Number(soh.toFixed(1)),
        is_anomaly: isAnom,
        anomaly_score: isAnom ? -0.14 : 0.09,
        base_rul: 69.12,
        shap_contributions: {
          Internal_Resistance_Ohm: Number(((0.10 - resVal) * 220).toFixed(2)),
          Discharge_Capacity_Ah: Number(((cap - 1.55) * 65).toFixed(2)),
          Cycle_Count: Number(((70 - cyc) * 0.15).toFixed(2)),
          Max_Temp_C: Number(((35 - temp) * 0.35).toFixed(2)),
          Avg_Voltage_V: Number(((volt - 3.7) * 1.5).toFixed(2)),
          Avg_Current_A: 0.12,
          Ambient_Temp_C: 0.08,
        },
        prediction_summary: isAnom
          ? "⚠️ CRITICAL ANOMALY: Excessive operating temperature or elevated impedance. Immediate thermal cooling and cutoff required."
          : "Nominal cell operation. Telemetry within stable operational safety bounds.",
      });
    } finally {
      setIsSimulating(false);
    }
  }, []);

  const handleSliderChange = (key, value) => {
    const updated = { ...simValues, [key]: Number(value) };
    setSimValues(updated);
    executePrediction(updated);
  };

  const applyPreset = (presetKey) => {
    const preset = PRESETS[presetKey];
    if (preset) {
      setSimValues(preset.values);
      executePrediction(preset.values);
    }
  };

  const domain = DOMAINS[activeDomain] || DOMAINS.ev;

  // Latest cycle data
  const latestCycle = useMemo(() => {
    if (!batteryData?.cycles?.length) return null;
    return batteryData.cycles[batteryData.cycles.length - 1];
  }, [batteryData]);

  // Dynamic EOL status
  const eolStatus = useMemo(() => {
    if (!latestCycle) {
      return { label: "OPTIMAL", color: "text-emerald-400 bg-emerald-500/10 border-emerald-500/30" };
    }
    const soh = Number(latestCycle.soh_percent ?? latestCycle.actual_soh ?? 0);
    if (soh <= domain.eolThreshold) {
      return {
        label: "EOL RETIRED",
        color: "text-rose-400 bg-rose-500/10 border-rose-500/30",
        alert: true,
      };
    } else if (soh <= domain.eolThreshold + 5) {
      return {
        label: "NEAR RETIREMENT",
        color: "text-amber-400 bg-amber-500/10 border-amber-500/30",
        alert: false,
      };
    }
    return {
      label: "HEALTHY",
      color: "text-emerald-400 bg-emerald-500/10 border-emerald-500/30",
      alert: false,
    };
  }, [latestCycle, domain]);

  // SHAP Chart data array
  const shapChartData = useMemo(() => {
    const contribs = simResult?.shap_contributions || {};
    return Object.entries(contribs)
      .map(([feature, val]) => ({
        feature: feature.replace(/_/g, " "),
        rawFeature: feature,
        impact: Number(val || 0),
        isPositive: Number(val || 0) >= 0,
      }))
      .sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));
  }, [simResult]);

  return (
    <div className="min-h-screen bg-[#090D16] text-slate-100 p-4 sm:p-6 lg:p-8 space-y-6">
      {/* ------------------------------------------------------------------- */}
      {/* Header & Controls Bar */}
      {/* ------------------------------------------------------------------- */}
      <header className="flex flex-col xl:flex-row xl:items-center justify-between gap-6 pb-6 border-b border-slate-800/80">
        <div className="space-y-1">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-gradient-to-br from-cyan-500/20 to-blue-600/10 border border-cyan-500/30 text-cyan-400 shadow-lg shadow-cyan-500/10">
              <Zap className="w-7 h-7" />
            </div>
            <div>
              <div className="flex items-center gap-2.5">
                <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight bg-gradient-to-r from-white via-slate-100 to-slate-400 bg-clip-text text-transparent">
                  BatteryMind
                </h1>
                <span className="px-2 py-0.5 text-xs font-semibold uppercase tracking-wider rounded-full border border-cyan-500/30 text-cyan-300 bg-cyan-500/10">
                  v1.0 Live XAI
                </span>
                <span
                  className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 text-xs font-medium rounded-full border cursor-pointer ${
                    apiConnected
                      ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-400"
                      : "border-amber-500/30 bg-amber-500/10 text-amber-400"
                  }`}
                  onClick={checkHealth}
                  title="Click to reconnect with FastAPI server"
                >
                  <span className={`w-1.5 h-1.5 rounded-full ${apiConnected ? "bg-emerald-400 animate-pulse" : "bg-amber-400"}`} />
                  {apiConnected ? "FastAPI Connected" : "Local Telemetry Mode"}
                </span>
              </div>
              <p className="text-xs sm:text-sm text-slate-400">
                Li-Ion Battery Degradation Prognostics, Anomaly Detection &amp; Explainable AI
              </p>
            </div>
          </div>
        </div>

        {/* Global Selectors */}
        <div className="flex flex-wrap items-center gap-4">
          {/* Domain Context Switcher */}
          <div className="flex flex-col gap-1.5">
            <label className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1">
              <Layers className="w-3.5 h-3.5 text-cyan-400" /> Domain Context
            </label>
            <div className="inline-flex p-1 rounded-xl bg-[#111827] border border-slate-800 shadow-inner">
              {Object.values(DOMAINS).map((d) => {
                const isActive = activeDomain === d.id;
                return (
                  <button
                    key={d.id}
                    onClick={() => setActiveDomain(d.id)}
                    className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all duration-150 ${
                      isActive
                        ? "bg-cyan-500 text-slate-950 shadow-md shadow-cyan-500/25 font-bold"
                        : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
                    }`}
                  >
                    {d.name} ({d.eolThreshold}%)
                  </button>
                );
              })}
            </div>
          </div>

          {/* Battery Cell Switcher */}
          <div className="flex flex-col gap-1.5">
            <label className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1">
              <BatteryCharging className="w-3.5 h-3.5 text-cyan-400" /> Battery Cell
            </label>
            <div className="inline-flex p-1 rounded-xl bg-[#111827] border border-slate-800 shadow-inner">
              {CELLS.map((cell) => {
                const isActive = selectedCell === cell.id;
                return (
                  <button
                    key={cell.id}
                    onClick={() => setSelectedCell(cell.id)}
                    className={`relative px-3 py-1.5 text-xs font-semibold rounded-lg transition-all duration-150 flex items-center gap-1.5 ${
                      isActive
                        ? "bg-slate-100 text-slate-950 font-bold shadow-md shadow-white/10"
                        : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
                    }`}
                  >
                    {cell.label}
                    {cell.isTest && (
                      <span
                        className={`text-[9px] px-1.5 py-0.2 rounded font-mono uppercase tracking-tight ${
                          isActive
                            ? "bg-indigo-600 text-white font-extrabold"
                            : "bg-indigo-500/20 text-indigo-300 border border-indigo-500/40"
                        }`}
                      >
                        Unseen Test
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </header>

      {/* ------------------------------------------------------------------- */}
      {/* Row 1: 4 KPI Cards */}
      {/* ------------------------------------------------------------------- */}
      <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* KPI 1: State of Health (SOH) */}
        <div className="p-5 rounded-2xl bg-[#111827] border border-slate-800/90 shadow-xl relative overflow-hidden group hover:border-slate-700 transition">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider flex items-center gap-1.5">
              <Battery className="w-4 h-4 text-cyan-400" /> State of Health (SOH)
            </span>
            <span className={`px-2 py-0.5 text-[10px] font-bold uppercase rounded border ${eolStatus.color}`}>
              {eolStatus.label}
            </span>
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-black tracking-tight text-white font-mono">
              {latestCycle ? `${formatNum(latestCycle.soh_percent ?? latestCycle.actual_soh, 1)}%` : "--"}
            </span>
            <span className="text-xs text-slate-400">
              of rated 2.0 Ah
            </span>
          </div>
          <div className="mt-3 flex items-center justify-between text-xs text-slate-400 border-t border-slate-800/60 pt-2.5">
            <span>Capacity: <strong className="text-slate-200 font-mono">{latestCycle ? `${formatNum(latestCycle.discharge_capacity_ah, 3)} Ah` : "--"}</strong></span>
            <span>EOL Gate: <strong className="text-rose-400 font-mono">{domain.eolThreshold}%</strong></span>
          </div>
        </div>

        {/* KPI 2: Predicted Remaining Useful Life (RUL) */}
        <div className="p-5 rounded-2xl bg-[#111827] border border-slate-800/90 shadow-xl relative overflow-hidden group hover:border-slate-700 transition">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider flex items-center gap-1.5">
              <Gauge className="w-4 h-4 text-cyan-400" /> Remaining Useful Life
            </span>
            <span className="px-2 py-0.5 text-[10px] font-bold uppercase rounded border border-cyan-500/30 text-cyan-300 bg-cyan-500/10">
              Prognostic Horizon
            </span>
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-black tracking-tight text-cyan-400 font-mono">
              {latestCycle ? safeRound(latestCycle.predicted_rul ?? latestCycle.actual_rul) : "--"}
            </span>
            <span className="text-xs text-slate-400">
              cycles remaining
            </span>
          </div>
          <div className="mt-3 flex items-center justify-between text-xs text-slate-400 border-t border-slate-800/60 pt-2.5">
            <span>Cycle Index: <strong className="text-slate-200 font-mono">{latestCycle?.cycle_count || "--"}</strong></span>
            <span>Total Logged: <strong className="text-slate-200 font-mono">{batteryData?.total_cycles || "--"}</strong></span>
          </div>
        </div>

        {/* KPI 3: Anomaly Alerts Count */}
        <div className="p-5 rounded-2xl bg-[#111827] border border-slate-800/90 shadow-xl relative overflow-hidden group hover:border-slate-700 transition">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider flex items-center gap-1.5">
              <ShieldAlert className="w-4 h-4 text-amber-400" /> Anomaly Detections
            </span>
            <span className="px-2 py-0.5 text-[10px] font-bold uppercase rounded border border-amber-500/30 text-amber-400 bg-amber-500/10">
              Isolation Forest
            </span>
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-black tracking-tight text-white font-mono">
              {batteryData?.anomaly_count ?? 0}
            </span>
            <span className="text-xs text-slate-400">
              anomalous cycles flagged
            </span>
          </div>
          <div className="mt-3 flex items-center justify-between text-xs text-slate-400 border-t border-slate-800/60 pt-2.5">
            <span className="truncate">
              Cycles: <strong className="text-rose-400 font-mono">{batteryData?.anomalous_cycles?.length ? batteryData.anomalous_cycles.join(", ") : "None detected"}</strong>
            </span>
          </div>
        </div>

        {/* KPI 4: Unseen Cell Test Metrics */}
        <div className="p-5 rounded-2xl bg-[#111827] border border-slate-800/90 shadow-xl relative overflow-hidden group hover:border-slate-700 transition">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider flex items-center gap-1.5">
              <Cpu className="w-4 h-4 text-indigo-400" /> Generalization (B0018)
            </span>
            <span className="px-2 py-0.5 text-[10px] font-bold uppercase rounded border border-indigo-500/30 text-indigo-300 bg-indigo-500/10">
              Zero Leakage
            </span>
          </div>
          <div className="flex items-baseline gap-3">
            <div>
              <span className="text-xs text-slate-400 block font-medium">SOH R²</span>
              <span className="text-2xl font-black text-emerald-400 font-mono">
                {formatNum(healthData?.metrics?.SOH_Percent?.r2, 3, "0.977")}
              </span>
            </div>
            <div className="h-7 w-[1px] bg-slate-800" />
            <div>
              <span className="text-xs text-slate-400 block font-medium">RUL RMSE</span>
              <span className="text-2xl font-black text-white font-mono">
                {formatNum(healthData?.metrics?.Remaining_Useful_Life_RUL?.rmse, 1, "23.2")}
              </span>
            </div>
          </div>
          <div className="mt-3 flex items-center justify-between text-xs text-slate-400 border-t border-slate-800/60 pt-2.5">
            <span>RUL R²: <strong className="text-slate-200 font-mono">{formatNum(healthData?.metrics?.Remaining_Useful_Life_RUL?.r2, 3, "0.542")}</strong></span>
            <span>SOH MAE: <strong className="text-emerald-400 font-mono">{formatNum(healthData?.metrics?.SOH_Percent?.mae, 2, "1.13")}%</strong></span>
          </div>
        </div>
      </section>

      {/* ------------------------------------------------------------------- */}
      {/* Row 2: Analytical Charts (60% Trajectory vs 40% SHAP) */}
      {/* ------------------------------------------------------------------- */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left (60%): Recharts Degradation Trajectory & Anomaly Markers */}
        <div className="lg:col-span-7 xl:col-span-7 p-5 rounded-2xl bg-[#111827] border border-slate-800/90 shadow-xl flex flex-col justify-between">
          <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
            <div>
              <h2 className="text-base font-bold text-white flex items-center gap-2">
                <TrendingDown className="w-4 h-4 text-cyan-400" />
                Cell Degradation Trajectory &amp; Anomaly Overlay
              </h2>
              <p className="text-xs text-slate-400">
                Discharge capacity fade and State of Health tracking with {domain.name} EOL threshold
              </p>
            </div>
            <div className="flex items-center gap-3 text-xs">
              <span className="inline-flex items-center gap-1.5 text-cyan-400">
                <span className="w-3 h-0.5 bg-cyan-400 rounded" /> SOH %
              </span>
              <span className="inline-flex items-center gap-1.5 text-indigo-400">
                <span className="w-3 h-0.5 bg-indigo-400 rounded" /> Capacity (Ah)
              </span>
              <span className="inline-flex items-center gap-1.5 text-rose-400">
                <span className="w-2 h-2 rounded-full bg-rose-500" /> Anomaly
              </span>
            </div>
          </div>

          <div className="h-[340px] w-full">
            {isLoading || !batteryData?.cycles ? (
              <div className="h-full flex items-center justify-center text-slate-500 text-sm">
                <RefreshCw className="w-5 h-5 animate-spin mr-2" /> Loading cell telemetry...
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={batteryData.cycles} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis
                    dataKey="cycle_count"
                    stroke="#64748b"
                    tick={{ fill: "#64748b", fontSize: 11 }}
                    tickLine={false}
                  />
                  <YAxis
                    yAxisId="soh"
                    domain={[55, 105]}
                    stroke="#64748b"
                    tick={{ fill: "#64748b", fontSize: 11 }}
                    tickLine={false}
                    unit="%"
                  />
                  <YAxis
                    yAxisId="cap"
                    orientation="right"
                    domain={[1.1, 2.1]}
                    stroke="#64748b"
                    tick={{ fill: "#64748b", fontSize: 11 }}
                    tickLine={false}
                    unit=" Ah"
                  />
                  <Tooltip content={<CustomTrajectoryTooltip domainThreshold={domain.eolThreshold} />} />
                  <ReferenceLine
                    yAxisId="soh"
                    y={domain.eolThreshold}
                    stroke="#EF4444"
                    strokeDasharray="4 4"
                    strokeWidth={1.5}
                    label={{
                      value: `EOL: ${domain.eolThreshold}%`,
                      fill: "#EF4444",
                      position: "insideTopRight",
                      fontSize: 10,
                      fontWeight: 700,
                    }}
                  />
                  <Line
                    yAxisId="soh"
                    type="monotone"
                    dataKey="soh_percent"
                    name="SOH %"
                    stroke="#06B6D4"
                    strokeWidth={2.2}
                    dot={<CustomAnomalyDot />}
                    activeDot={{ r: 6, fill: "#06B6D4" }}
                  />
                  <Line
                    yAxisId="cap"
                    type="monotone"
                    dataKey="discharge_capacity_ah"
                    name="Capacity (Ah)"
                    stroke="#818CF8"
                    strokeWidth={1.5}
                    strokeDasharray="2 2"
                    dot={false}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            )}
          </div>

          <div className="mt-3 flex items-center justify-between text-xs text-slate-400 border-t border-slate-800/80 pt-2.5">
            <span>Tested Range: <strong className="text-slate-200">1 → {batteryData?.total_cycles || "--"} cycles</strong></span>
            <span>Cell Profile: <strong className="text-cyan-400 font-mono">{selectedCell}</strong></span>
            <span className="text-slate-500 hidden sm:inline">Dashed Red Line: Domain EOL Retirement Gate</span>
          </div>
        </div>

        {/* Right (40%): Real-Time SHAP Explainability Bar Chart */}
        <div className="lg:col-span-5 xl:col-span-5 p-5 rounded-2xl bg-[#111827] border border-slate-800/90 shadow-xl flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-base font-bold text-white flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-cyan-400" />
                SHAP Explainability (RUL Drivers)
              </h2>
              <span className="text-[11px] px-2 py-0.5 rounded font-mono font-semibold bg-slate-800 text-slate-300">
                TreeExplainer
              </span>
            </div>
            <p className="text-xs text-slate-400 mb-3">
              Additive feature impacts relative to base RUL ({formatNum(simResult?.base_rul, 1, "69.1")} cycles).
              <span className="text-emerald-400 ml-1 font-medium">Green = Extends RUL</span>,
              <span className="text-rose-400 ml-1 font-medium">Red = Accelerates EOL</span>.
            </p>
          </div>

          <div className="h-[310px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                layout="vertical"
                data={shapChartData}
                margin={{ top: 5, right: 20, left: 25, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                <XAxis
                  type="number"
                  stroke="#64748b"
                  tick={{ fill: "#64748b", fontSize: 10 }}
                  tickLine={false}
                  unit=" cyc"
                />
                <YAxis
                  type="category"
                  dataKey="feature"
                  stroke="#64748b"
                  tick={{ fill: "#94a3b8", fontSize: 10, fontWeight: 600 }}
                  tickLine={false}
                  width={110}
                />
                <Tooltip content={<CustomShapTooltip />} />
                <ReferenceLine x={0} stroke="#475569" strokeWidth={1.5} />
                <Bar dataKey="impact" radius={[4, 4, 4, 4]}>
                  {shapChartData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={entry.impact >= 0 ? "#10B981" : "#EF4444"}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="mt-3 p-2.5 rounded-xl bg-slate-900/80 border border-slate-800 text-[11px] text-slate-300">
            <strong className="text-cyan-400 block mb-0.5">Top Insight:</strong>
            {simResult?.prediction_summary || "Telemetry within nominal boundaries."}
          </div>
        </div>
      </section>

      {/* ------------------------------------------------------------------- */}
      {/* Row 3: Live "What-If" Telemetry Simulator */}
      {/* ------------------------------------------------------------------- */}
      <section className="p-5 sm:p-6 rounded-2xl bg-[#111827] border border-slate-800/90 shadow-2xl">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-5 border-b border-slate-800">
          <div>
            <div className="flex items-center gap-2">
              <div className="p-1.5 rounded-lg bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
                <Sliders className="w-5 h-5" />
              </div>
              <h2 className="text-lg font-bold text-white">
                Live "What-If" Telemetry Simulator
              </h2>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Adjust operating conditions in real time to simulate thermal degradation, impedance growth, and voltage collapse.
            </p>
          </div>

          {/* Quick Presets */}
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs text-slate-400 font-semibold mr-1">Presets:</span>
            {Object.entries(PRESETS).map(([key, preset]) => {
              const Icon = preset.icon;
              return (
                <button
                  key={key}
                  onClick={() => applyPreset(key)}
                  className={`px-3 py-1.5 text-xs font-semibold rounded-xl border transition flex items-center gap-1.5 ${
                    preset.color === "emerald"
                      ? "border-emerald-500/40 text-emerald-400 bg-emerald-500/10 hover:bg-emerald-500/20"
                      : preset.color === "rose"
                      ? "border-rose-500/40 text-rose-400 bg-rose-500/10 hover:bg-rose-500/20"
                      : "border-amber-500/40 text-amber-400 bg-amber-500/10 hover:bg-amber-500/20"
                  }`}
                >
                  <Icon className="w-3.5 h-3.5" />
                  {preset.name}
                </button>
              );
            })}
          </div>
        </div>

        {/* Sliders & Real-Time Output Panel */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 mt-6">
          {/* Sliders (7 Cols) */}
          <div className="lg:col-span-7 grid grid-cols-1 sm:grid-cols-2 gap-5">
            {/* Slider 1: Max Temp */}
            <div className="space-y-2 p-3.5 rounded-xl bg-slate-900/60 border border-slate-800/80">
              <div className="flex justify-between items-center text-xs">
                <span className="font-semibold text-slate-300 flex items-center gap-1.5">
                  <Thermometer className="w-3.5 h-3.5 text-rose-400" /> Max Temperature
                </span>
                <span className={`font-mono font-bold ${(simValues?.Max_Temp_C || 0) >= 45 ? "text-rose-400" : "text-cyan-400"}`}>
                  {formatNum(simValues?.Max_Temp_C, 1)} °C
                </span>
              </div>
              <input
                type="range"
                min="24.0"
                max="55.0"
                step="0.5"
                value={simValues?.Max_Temp_C || 32}
                onChange={(e) => handleSliderChange("Max_Temp_C", e.target.value)}
                className="w-full accent-cyan-400 bg-slate-800 rounded-lg h-2 cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-slate-500">
                <span>24 °C (Ambient)</span>
                <span>45 °C (Alarm)</span>
                <span>55 °C</span>
              </div>
            </div>

            {/* Slider 2: Internal Resistance */}
            <div className="space-y-2 p-3.5 rounded-xl bg-slate-900/60 border border-slate-800/80">
              <div className="flex justify-between items-center text-xs">
                <span className="font-semibold text-slate-300 flex items-center gap-1.5">
                  <Activity className="w-3.5 h-3.5 text-amber-400" /> Internal Resistance
                </span>
                <span className={`font-mono font-bold ${(simValues?.Internal_Resistance_Ohm || 0) >= 0.18 ? "text-amber-400" : "text-cyan-400"}`}>
                  {formatNum(simValues?.Internal_Resistance_Ohm, 3)} Ω
                </span>
              </div>
              <input
                type="range"
                min="0.065"
                max="0.260"
                step="0.002"
                value={simValues?.Internal_Resistance_Ohm || 0.08}
                onChange={(e) => handleSliderChange("Internal_Resistance_Ohm", e.target.value)}
                className="w-full accent-cyan-400 bg-slate-800 rounded-lg h-2 cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-slate-500">
                <span>0.065 Ω (Fresh)</span>
                <span>0.180 Ω (Aged)</span>
                <span>0.260 Ω</span>
              </div>
            </div>

            {/* Slider 3: Discharge Voltage */}
            <div className="space-y-2 p-3.5 rounded-xl bg-slate-900/60 border border-slate-800/80">
              <div className="flex justify-between items-center text-xs">
                <span className="font-semibold text-slate-300 flex items-center gap-1.5">
                  <Zap className="w-3.5 h-3.5 text-cyan-400" /> Avg Voltage (Plateau)
                </span>
                <span className={`font-mono font-bold ${(simValues?.Avg_Voltage_V || 0) < 3.45 ? "text-rose-400" : "text-cyan-400"}`}>
                  {formatNum(simValues?.Avg_Voltage_V, 2)} V
                </span>
              </div>
              <input
                type="range"
                min="3.10"
                max="3.90"
                step="0.02"
                value={simValues?.Avg_Voltage_V || 3.8}
                onChange={(e) => handleSliderChange("Avg_Voltage_V", e.target.value)}
                className="w-full accent-cyan-400 bg-slate-800 rounded-lg h-2 cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-slate-500">
                <span>3.10 V (Cutoff)</span>
                <span>3.60 V</span>
                <span>3.90 V (Fresh)</span>
              </div>
            </div>

            {/* Slider 4: Cycle Count */}
            <div className="space-y-2 p-3.5 rounded-xl bg-slate-900/60 border border-slate-800/80">
              <div className="flex justify-between items-center text-xs">
                <span className="font-semibold text-slate-300 flex items-center gap-1.5">
                  <RefreshCw className="w-3.5 h-3.5 text-indigo-400" /> Cycle Count
                </span>
                <span className="font-mono font-bold text-cyan-400">
                  {simValues?.Cycle_Count || 1}
                </span>
              </div>
              <input
                type="range"
                min="1"
                max="180"
                step="1"
                value={simValues?.Cycle_Count || 1}
                onChange={(e) => handleSliderChange("Cycle_Count", e.target.value)}
                className="w-full accent-cyan-400 bg-slate-800 rounded-lg h-2 cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-slate-500">
                <span>1</span>
                <span>90 (Mid)</span>
                <span>180</span>
              </div>
            </div>
          </div>

          {/* Simulator Live Inference Cards (5 Cols) */}
          <div className="lg:col-span-5 p-5 rounded-xl bg-gradient-to-br from-slate-900 to-[#0e1626] border border-slate-800 flex flex-col justify-between space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800/80">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                <Cpu className="w-4 h-4 text-cyan-400" /> Real-Time Inference
              </span>
              <span
                className={`px-2.5 py-0.5 text-xs font-bold rounded-full border flex items-center gap-1.5 ${
                  simResult?.is_anomaly
                    ? "border-rose-500/50 bg-rose-500/20 text-rose-400 animate-pulse"
                    : "border-emerald-500/40 bg-emerald-500/10 text-emerald-400"
                }`}
              >
                {simResult?.is_anomaly ? (
                  <>
                    <AlertTriangle className="w-3.5 h-3.5" /> ANOMALY DETECTED
                  </>
                ) : (
                  <>
                    <CheckCircle2 className="w-3.5 h-3.5" /> NOMINAL STATE
                  </>
                )}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800/80">
                <span className="text-[11px] text-slate-400 uppercase font-semibold block">
                  Simulated RUL
                </span>
                <div className="flex items-baseline gap-1 mt-1">
                  <span className="text-3xl font-black text-cyan-400 font-mono">
                    {safeRound(simResult?.predicted_rul)}
                  </span>
                  <span className="text-xs text-slate-400">cycles</span>
                </div>
              </div>

              <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800/80">
                <span className="text-[11px] text-slate-400 uppercase font-semibold block">
                  Simulated SOH
                </span>
                <div className="flex items-baseline gap-1 mt-1">
                  <span className="text-3xl font-black text-white font-mono">
                    {formatNum(simResult?.predicted_soh_percent, 1)}%
                  </span>
                  <span className={`text-[10px] font-bold ${(simResult?.predicted_soh_percent || 0) <= domain.eolThreshold ? "text-rose-400" : "text-emerald-400"}`}>
                    {(simResult?.predicted_soh_percent || 0) <= domain.eolThreshold ? "≤ EOL" : "> EOL"}
                  </span>
                </div>
              </div>
            </div>

            <div className="text-xs text-slate-400 space-y-1 bg-slate-950/50 p-3 rounded-lg border border-slate-800/60 font-mono">
              <div className="flex justify-between">
                <span>Anomaly Score:</span>
                <strong className={(simResult?.anomaly_score || 0) < 0 ? "text-rose-400" : "text-emerald-400"}>
                  {formatNum(simResult?.anomaly_score, 4)}
                </strong>
              </div>
              <div className="flex justify-between">
                <span>Selected Domain:</span>
                <strong className="text-cyan-400">{domain.name} ({domain.eolThreshold}% EOL)</strong>
              </div>
            </div>

            <div className="text-[11px] text-slate-400 flex items-center justify-between pt-1">
              <span className="flex items-center gap-1">
                <Info className="w-3.5 h-3.5 text-slate-500" />
                POST /predict endpoint
              </span>
              {isSimulating && (
                <span className="text-cyan-400 flex items-center gap-1 text-[10px]">
                  <RefreshCw className="w-3 h-3 animate-spin" /> Inferring...
                </span>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* ------------------------------------------------------------------- */}
      {/* Footer */}
      {/* ------------------------------------------------------------------- */}
      <footer className="pt-4 pb-2 border-t border-slate-800/80 text-center text-xs text-slate-500 flex flex-col sm:flex-row items-center justify-between gap-2">
        <span>
          BatteryMind • Grounded on NASA Ames Li-Ion Battery Prognostics Benchmark
        </span>
        <span className="flex items-center gap-3">
          <span className="font-mono text-slate-400">XGBoost Regressor + Isolation Forest + SHAP</span>
        </span>
      </footer>
    </div>
  );
}

// -----------------------------------------------------------------------------
// Wrapped Root Export with ErrorBoundary
// -----------------------------------------------------------------------------
export default function App() {
  return (
    <ErrorBoundary>
      <BatteryMindDashboard />
    </ErrorBoundary>
  );
}

// -----------------------------------------------------------------------------
// Custom Subcomponents & Tooltips
// -----------------------------------------------------------------------------

function CustomAnomalyDot(props) {
  const { cx, cy, payload } = props;
  if (!payload?.is_anomaly || typeof cx !== "number" || typeof cy !== "number" || isNaN(cx) || isNaN(cy)) {
    return null;
  }
  return (
    <svg x={cx - 7} y={cy - 7} width={14} height={14} viewBox="0 0 14 14" className="overflow-visible">
      <circle cx="7" cy="7" r="7" fill="#EF4444" fillOpacity="0.3" className="animate-ping" />
      <circle cx="7" cy="7" r="4.5" fill="#EF4444" stroke="#090D16" strokeWidth="1.5" />
    </svg>
  );
}

function CustomTrajectoryTooltip({ active, payload, domainThreshold }) {
  if (!active || !payload?.length) return null;
  const data = payload[0]?.payload;
  if (!data) return null;

  const soh = Number(data.soh_percent ?? data.actual_soh ?? 0);
  const isBelowEol = soh <= (domainThreshold || 80);

  return (
    <div className="p-3 rounded-xl bg-slate-900 border border-slate-700 shadow-2xl text-xs space-y-1.5 font-mono min-w-[200px]">
      <div className="flex items-center justify-between border-b border-slate-800 pb-1 font-sans">
        <strong className="text-white font-bold">Cycle #{data.cycle_count ?? "--"}</strong>
        {data.is_anomaly ? (
          <span className="px-1.5 py-0.5 text-[10px] font-bold rounded bg-rose-500/20 text-rose-400 border border-rose-500/40">
            ⚠️ ANOMALY
          </span>
        ) : (
          <span className="text-slate-400 text-[11px]">Normal</span>
        )}
      </div>
      <div className="flex justify-between text-cyan-400">
        <span>SOH %:</span>
        <strong className={isBelowEol ? "text-rose-400" : "text-cyan-400"}>
          {formatNum(soh, 2)}%
        </strong>
      </div>
      <div className="flex justify-between text-indigo-400">
        <span>Capacity:</span>
        <strong>{formatNum(data.discharge_capacity_ah, 4)} Ah</strong>
      </div>
      <div className="flex justify-between text-slate-300">
        <span>Predicted RUL:</span>
        <strong className="text-cyan-300">{safeRound(data.predicted_rul ?? data.actual_rul)} cyc</strong>
      </div>
      <div className="flex justify-between text-slate-400">
        <span>Max Temp:</span>
        <strong className={(data.max_temp_c || 0) >= 45 ? "text-rose-400" : "text-slate-200"}>
          {formatNum(data.max_temp_c, 1)} °C
        </strong>
      </div>
      <div className="flex justify-between text-slate-400">
        <span>Internal Res:</span>
        <strong className="text-slate-200">{formatNum(data.internal_resistance_ohm, 4)} Ω</strong>
      </div>
    </div>
  );
}

function CustomShapTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const data = payload[0]?.payload;
  if (!data) return null;

  const impact = Number(data.impact || 0);
  const isPos = impact >= 0;

  return (
    <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-700 shadow-xl text-xs font-mono">
      <div className="font-sans font-semibold text-white capitalize mb-1">
        {data.feature}
      </div>
      <div className={`font-bold ${isPos ? "text-emerald-400" : "text-rose-400"}`}>
        {isPos ? "+" : ""}
        {formatNum(impact, 2)} cycles
      </div>
      <div className="text-[10px] text-slate-400 mt-1 font-sans">
        {isPos
          ? "Extends cell remaining life beyond baseline."
          : "Degradation factor accelerating end-of-life."}
      </div>
    </div>
  );
}
