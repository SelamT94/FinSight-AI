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
import { api } from "@/lib/api";
import { tooltipStyle } from "@/lib/chartTheme";
import {
  AlertBanner,
  GlassButton,
  GlassField,
  GlassPanel,
  GlassSelect,
  PageHeader,
} from "@/components/Glass";
import { MetricCard } from "@/components/MetricCard";

type CompareOut = {
  model_a_id: string;
  model_b_id: string;
  a: { summary: string; metadata: Record<string, number> };
  b: { summary: string; metadata: Record<string, number> };
  evaluation: {
    accuracy_score: number;
    completeness_score: number;
    clarity_score: number;
    alternative_summary: string;
    factual_errors: string;
  };
  rouge_a_vs_b: {
    rouge1_f: number | null;
    rouge2_f: number | null;
    rougeL_f: number | null;
  };
};

export default function ComparisonPage() {
  const [months, setMonths] = useState<string[]>([]);
  const [month, setMonth] = useState("");
  const [modelA, setModelA] = useState("llama");
  const [modelB, setModelB] = useState("mistral");
  const [data, setData] = useState<CompareOut | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api<{ months: string[] }>("/api/months").then((r) => {
      setMonths(r.months);
      if (r.months.length) setMonth(r.months[0]);
    });
  }, []);

  const run = async () => {
    if (!month) return;
    setLoading(true);
    setError(null);
    try {
      const out = await api<CompareOut>("/api/compare", {
        method: "POST",
        body: JSON.stringify({ month, model_a: modelA, model_b: modelB }),
      });
      setData(out);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  const rougeChart =
    data?.rouge_a_vs_b && data.rouge_a_vs_b.rouge1_f != null
      ? [
        { name: "ROUGE-1", score: data.rouge_a_vs_b.rouge1_f },
        {
          name: "ROUGE-2",
          score: data.rouge_a_vs_b.rouge2_f ?? 0,
        },
        {
          name: "ROUGE-L",
          score: data.rouge_a_vs_b.rougeL_f ?? 0,
        },
      ]
      : [];

  return (
    <div className="mx-auto max-w-6xl space-y-8">
      <PageHeader
        title="Model comparison"
        subtitle="Same month, both models zero-shot — Model B judges Model A"
      />

      <GlassPanel className="flex flex-wrap items-end gap-4">
        <GlassField label="Month">
          <GlassSelect value={month} onChange={(e) => setMonth(e.target.value)}>
            {months.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </GlassSelect>
        </GlassField>

        <GlassField label="Model A">
          <GlassSelect
            value={modelA}
            onChange={(e) => setModelA(e.target.value)}
            title={
              modelA === "llama"
                ? "meta-llama/Llama-3.2-3B-Instruct"
                : modelA === "mistral"
                  ? "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
                  : "Qwen/Qwen2.5-7B-Instruct"
            }
          >
            <option value="llama" title="meta-llama/Llama-3.2-3B-Instruct">LLaMA 3.2 3B</option>
            <option value="mistral" title="TheBloke/Mistral-7B-Instruct-v0.2-GPTQ">Mistral 7B</option>
            <option value="qwen" title="Qwen/Qwen2.5-7B-Instruct">Qwen 2.5 7B</option>
          </GlassSelect>
        </GlassField>

        <GlassField label="Model B (Judge)">
          <GlassSelect
            value={modelB}
            onChange={(e) => setModelB(e.target.value)}
            title={
              modelB === "llama"
                ? "meta-llama/Llama-3.2-3B-Instruct"
                : modelB === "mistral"
                  ? "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
                  : "Qwen/Qwen2.5-7B-Instruct"
            }
          >
            <option value="llama" title="meta-llama/Llama-3.2-3B-Instruct">LLaMA 3.2 3B</option>
            <option value="mistral" title="TheBloke/Mistral-7B-Instruct-v0.2-GPTQ">Mistral 7B</option>
            <option value="qwen" title="Qwen/Qwen2.5-7B-Instruct">Qwen 2.5 7B</option>
          </GlassSelect>
        </GlassField>
        <GlassButton onClick={run} disabled={loading}>
          {loading ? "Running…" : "Run comparison"}
        </GlassButton>
      </GlassPanel>

      {error && <AlertBanner tone="error">{error}</AlertBanner>}

      {data && (
        <>
          <div className="grid gap-4 sm:grid-cols-3">
            <MetricCard
              label="Accuracy (judge)"
              value={String(data.evaluation.accuracy_score)}
            />
            <MetricCard
              label="Completeness"
              value={String(data.evaluation.completeness_score)}
            />
            <MetricCard
              label="Clarity"
              value={String(data.evaluation.clarity_score)}
            />
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            <GlassPanel strong>
              <h2 className="mb-2 text-sm font-semibold text-sky-300">Model A ({data.model_a_id})</h2>
              <p className="mb-3 text-xs text-white/40">
                {data.a.metadata.inference_time_sec?.toFixed?.(2)}s ·{" "}
                {data.a.metadata.tokens_per_sec?.toFixed?.(1)} tok/s
              </p>
              <p className="glass-inset whitespace-pre-wrap rounded-2xl p-4 text-sm leading-relaxed text-white/85">
                {data.a.summary}
              </p>
            </GlassPanel>
            <GlassPanel strong>
              <h2 className="mb-2 text-sm font-semibold text-violet-300">Model B ({data.model_b_id})</h2>
              <p className="mb-3 text-xs text-white/40">
                {data.b.metadata.inference_time_sec?.toFixed?.(2)}s ·{" "}
                {data.b.metadata.tokens_per_sec?.toFixed?.(1)} tok/s
              </p>
              <p className="glass-inset whitespace-pre-wrap rounded-2xl p-4 text-sm leading-relaxed text-white/85">
                {data.b.summary}
              </p>
            </GlassPanel>
          </div>

          {rougeChart.length > 0 && (
            <GlassPanel>
              <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-white/50">
                ROUGE (A vs B)
              </h2>
              <div className="h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={rougeChart}>
                    <XAxis dataKey="name" />
                    <YAxis domain={[0, 1]} />
                    <Tooltip {...tooltipStyle} />
                    <Legend />
                    <Bar dataKey="score" fill="#64d2ff" name="F1" radius={[6, 6, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </GlassPanel>
          )}

          {(data.evaluation.factual_errors ||
            data.evaluation.alternative_summary) && (
              <AlertBanner tone="warn" title="Judge notes">
                {data.evaluation.factual_errors && (
                  <p className="mb-2">{data.evaluation.factual_errors}</p>
                )}
                {data.evaluation.alternative_summary && (
                  <p>
                    <strong>Alternative:</strong>{" "}
                    {data.evaluation.alternative_summary}
                  </p>
                )}
              </AlertBanner>
            )}
        </>
      )}
    </div>
  );
}
