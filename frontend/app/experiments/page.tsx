"use client";

import { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { api, BASE } from "@/lib/api";
import { CHART_COLORS, tooltipStyle } from "@/lib/chartTheme";
import {
  AlertBanner,
  GlassPanel,
  PageHeader,
  GlassBadge,
} from "@/components/Glass";

function DynamicMetricCard({ row }: { row: Record<string, unknown> }) {
  const keys = Object.keys(row);
  
  const identifiers = keys.filter(k => ["model", "month", "strategy", "reference"].includes(k));
  const strings = keys.filter(k => typeof row[k] === "string" && !identifiers.includes(k) && String(row[k]).length > 30);
  const metrics = keys.filter(k => !identifiers.includes(k) && !strings.includes(k));

  const formatNumber = (val: unknown, key: string) => {
    if (typeof val === "number" || (typeof val === "string" && val.trim() !== "" && !isNaN(Number(val)))) {
      const num = Number(val);
      if (Number.isInteger(num)) return num.toString();
      if (key.includes("rouge") || key.includes("score") || key.includes("rate") || key.includes("accuracy") || key.includes("completeness") || key.includes("clarity")) {
        return num.toFixed(3);
      }
      return num.toFixed(2);
    }
    return String(val ?? "");
  };

  return (
    <GlassPanel className="flex flex-col gap-4">
      {identifiers.length > 0 && (
        <div className="flex flex-wrap items-center gap-4 border-b border-white/10 pb-3">
          {identifiers.map(id => (
            <div key={id} className="flex items-center gap-1.5">
              <span className="text-[10px] uppercase tracking-wider text-white/40">{id}</span>
              <GlassBadge tone={id === "model" ? "accent" : "default"}>{String(row[id] || "")}</GlassBadge>
            </div>
          ))}
        </div>
      )}
      
      {metrics.length > 0 && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8">
          {metrics.map(m => (
            <div key={m} className="flex flex-col gap-1 rounded-xl bg-white/5 p-3">
              <span className="truncate text-[10px] font-medium uppercase tracking-wider text-white/50" title={m.replace(/_/g, " ")}>
                {m.replace(/_/g, " ")}
              </span>
              <span className="text-lg font-semibold text-white/90">
                {formatNumber(row[m], m)}
              </span>
            </div>
          ))}
        </div>
      )}

      {strings.length > 0 && (
        <div className="flex flex-col gap-3">
          {strings.map(s => (
            <div key={s} className="flex flex-col gap-1.5">
              <span className="text-[10px] font-medium uppercase tracking-wider text-white/50">
                {s.replace(/_/g, " ")}
              </span>
              <div className="rounded-xl bg-black/20 p-3 text-xs leading-relaxed text-white/70">
                {String(row[s] || "")}
              </div>
            </div>
          ))}
        </div>
      )}
    </GlassPanel>
  );
}


const tabs = [
  { id: 1, label: "Exp 1 — ROUGE" },
  { id: 2, label: "Exp 2 — Insights" },
  { id: 3, label: "Exp 3 — Prompts" },
  { id: 4, label: "Exp 4 — Models" },
  { id: 5, label: "Exp 5 — Resources" },
];

export default function ExperimentsPage() {
  const [tab, setTab] = useState(1);
  const [rows, setRows] = useState<Record<string, unknown>[]>([]);
  const [meta, setMeta] = useState<{ exists: boolean } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [runModel, setRunModel] = useState<"all" | "llama" | "mistral" | "qwen">("all");
  const [runStatus, setRunStatus] = useState<{ type: "success" | "error" | "loading", message: string } | null>(null);

  const handleRun = async () => {
    setRunStatus({ type: "loading", message: "Starting experiment runner..." });
    try {
      const res = await api<{ message: string }>("/api/experiments/run", {
        method: "POST",
        body: JSON.stringify({ model_id: runModel }),
      });
      setRunStatus({ type: "success", message: res.message });
    } catch (e: any) {
      setRunStatus({ type: "error", message: e.message || "Failed to start runner" });
    }
  };

  useEffect(() => {
    setError(null);
    api<{ experiments: { id: number; exists: boolean }[] }>("/api/experiments")
      .then((r) => {
        const e = r.experiments.find((x) => x.id === tab);
        setMeta({ exists: e?.exists ?? false });
      })
      .catch(() => setMeta(null));

    api<{ rows: Record<string, unknown>[] }>(`/api/experiments/${tab}`)
      .then((r) => setRows(r.rows))
      .catch(() => {
        setRows([]);
        setError(`experiment_${tab}.csv not found — run scripts/run_experiments.py`);
      });
  }, [tab]);

  const downloadUrl = `${BASE.replace(/\/$/, "")}/api/experiments/${tab}/download`;

  const exp3Chart =
    tab === 3 && rows.length
      ? rows.map((r) => ({
          strategy: String(r.strategy ?? ""),
          accuracy: Number(r.accuracy_score ?? 0),
          completeness: Number(r.completeness_score ?? 0),
          clarity: Number(r.clarity_score ?? 0),
        }))
      : [];

  const exp5Chart =
    tab === 5 && rows.length
      ? rows.map((r) => ({
          model: String(r.model ?? ""),
          time: Number(r.mean_inference_time ?? 0),
          ram: Number(r.mean_ram_gb ?? 0),
        }))
      : [];

  const columns = rows.length > 0 ? Object.keys(rows[0]) : [];

  return (
    <div className="w-full space-y-8">
      <PageHeader
        title="Experiment results"
        subtitle="CSVs from evaluation/results/ via the API"
      />

      <GlassPanel className="flex items-center gap-4">
        <div>
          <label className="mb-1 block text-xs font-semibold uppercase tracking-wider text-white/50">
            Select Model
          </label>
          <select
            value={runModel}
            onChange={(e) => setRunModel(e.target.value as any)}
            className="w-48 rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-white/20"
          >
            <option value="all" className="bg-slate-900">All Models</option>
            <option value="llama" className="bg-slate-900">Llama</option>
            <option value="mistral" className="bg-slate-900">Mistral</option>
            <option value="qwen" className="bg-slate-900">Qwen</option>
          </select>
        </div>
        <button
          onClick={handleRun}
          disabled={runStatus?.type === "loading"}
          className="mt-5 rounded-xl bg-white/10 px-5 py-2 text-sm font-medium text-white transition hover:bg-white/20 disabled:opacity-50"
        >
          {runStatus?.type === "loading" ? "Starting..." : "Run Evaluation"}
        </button>
      </GlassPanel>

      {runStatus && (
        <AlertBanner tone={runStatus.type === "error" ? "warn" : "info" as any}>
          {runStatus.message}
        </AlertBanner>
      )}

      <div className="glass flex flex-wrap gap-2 rounded-3xl p-2">
        {tabs.map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => setTab(t.id)}
            className={`rounded-2xl px-4 py-2.5 text-sm font-medium transition-all ${
              tab === t.id
                ? "glass-nav-active text-white"
                : "text-white/55 hover:bg-white/10 hover:text-white/85"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      <a
        href={meta?.exists ? downloadUrl : "#"}
        className={`inline-flex items-center rounded-2xl border px-5 py-2.5 text-sm font-medium transition ${
          meta?.exists
            ? "glass-btn-secondary"
            : "cursor-not-allowed border-white/10 text-white/25"
        }`}
        {...(meta?.exists ? { download: true } : {})}
      >
        Download experiment_{tab}.csv
      </a>

      {error && <AlertBanner tone="warn">{error}</AlertBanner>}

      {tab === 3 && exp3Chart.length > 0 && (
        <GlassPanel>
          <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-white/50">
            Mistral scores by strategy
          </h2>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={exp3Chart}>
                <XAxis dataKey="strategy" />
                <YAxis domain={[0, 10]} />
                <Tooltip {...tooltipStyle} />
                <Legend />
                <Bar dataKey="accuracy" fill={CHART_COLORS[0]} name="Accuracy" radius={[4, 4, 0, 0]} />
                <Bar dataKey="completeness" fill={CHART_COLORS[1]} name="Completeness" radius={[4, 4, 0, 0]} />
                <Bar dataKey="clarity" fill={CHART_COLORS[2]} name="Clarity" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </GlassPanel>
      )}

      {tab === 5 && exp5Chart.length > 0 && (
        <GlassPanel>
          <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-white/50">
            Resource efficiency
          </h2>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={exp5Chart}>
                <XAxis dataKey="model" />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" />
                <Tooltip {...tooltipStyle} />
                <Legend />
                <Bar yAxisId="left" dataKey="time" fill={CHART_COLORS[0]} name="Mean time (s)" radius={[4, 4, 0, 0]} />
                <Bar yAxisId="right" dataKey="ram" fill={CHART_COLORS[4]} name="Mean RAM (GB)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </GlassPanel>
      )}

      {rows.length > 0 && (
        <div className="flex flex-col gap-4">
          {rows.map((row, i) => (
            <DynamicMetricCard key={i} row={row} />
          ))}
        </div>
      )}
    </div>
  );
}
