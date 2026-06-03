"use client";

import { useEffect, useState, useCallback } from "react";
import { api } from "@/lib/api";
import {
  AlertBanner,
  GlassBadge,
  GlassButton,
  GlassField,
  GlassPanel,
  GlassSelect,
  GlassTextarea,
  PageHeader,
} from "@/components/Glass";

type SummarizeOut = {
  summary: string;
  prompt_strategy: string;
  metadata: {
    inference_time_sec: number;
    tokens_per_sec: number;
    total_tokens: number;
  };
};

type InsightsOut = {
  insights: string[];
  metadata: { inference_time_sec: number; tokens_per_sec: number };
};

export default function SummariesPage() {
  const [months, setMonths] = useState<string[]>([]);
  const [month, setMonth] = useState("");
  const [strategy, setStrategy] = useState<
    "zero_shot" | "few_shot" | "chain_of_thought"
  >("zero_shot");
  const [model, setModel] = useState<"llama" | "mistral" | "qwen">("llama");
  const [summary, setSummary] = useState<SummarizeOut | null>(null);
  const [insights, setInsights] = useState<InsightsOut | null>(null);
  const [loading, setLoading] = useState(false);
  const [insLoading, setInsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api<{ months: string[] }>("/api/months").then((r) => {
      setMonths(r.months);
      if (r.months.length) setMonth(r.months[0]);
    });
  }, []);

  const cacheKey = `${month}|${strategy}|${model}`;
  const [cache, setCache] = useState<Record<string, SummarizeOut>>({});

  const generateSummary = useCallback(async () => {
    if (!month) return;
    setError(null);
    if (cache[cacheKey]) {
      setSummary(cache[cacheKey]);
      return;
    }
    setLoading(true);
    try {
      const out = await api<SummarizeOut>("/api/summarize", {
        method: "POST",
        body: JSON.stringify({ month, strategy, model }),
      });
      setSummary(out);
      setCache((c) => ({ ...c, [cacheKey]: out }));
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [month, strategy, model, cacheKey, cache]);

  const generateInsights = async () => {
    if (!month || !summary?.summary) return;
    setInsLoading(true);
    setError(null);
    try {
      const out = await api<InsightsOut>("/api/insights", {
        method: "POST",
        body: JSON.stringify({ month, summary: summary.summary, model }),
      });
      setInsights(out);
    } catch (e) {
      setError(String(e));
    } finally {
      setInsLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-4xl space-y-8">
      <PageHeader
        title="AI summaries"
        subtitle="Zero-shot, few-shot, or chain-of-thought — Llama vs Mistral tracks"
      />

      <GlassPanel className="flex flex-wrap items-end gap-5">
        <GlassField label="Month">
          <GlassSelect value={month} onChange={(e) => setMonth(e.target.value)}>
            {months.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </GlassSelect>
        </GlassField>

        <GlassField label="Prompt strategy">
          <div className="mt-1 flex flex-wrap gap-3">
            {(
              [
                ["zero_shot", "Zero-shot"],
                ["few_shot", "Few-shot"],
                ["chain_of_thought", "CoT"],
              ] as const
            ).map(([v, label]) => (
              <label
                key={v}
                className={`flex cursor-pointer items-center gap-2 rounded-xl px-3 py-2 text-sm transition ${strategy === v
                  ? "glass-nav-active text-white"
                  : "glass-inset text-white/60 hover:text-white/90"
                  }`}
              >
                <input
                  type="radio"
                  name="strat"
                  className="sr-only"
                  checked={strategy === v}
                  onChange={() => setStrategy(v)}
                />
                {label}
              </label>
            ))}
          </div>
        </GlassField>

        <GlassField label="Model">
          <GlassSelect
            value={model}
            onChange={(e) => setModel(e.target.value as "llama" | "mistral" | "qwen")}
            title={
              model === "llama"
                ? "meta-llama/Llama-3.2-3B-Instruct"
                : model === "mistral"
                  ? "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
                  : "Qwen/Qwen2.5-7B-Instruct"
            }
          >
            <option value="llama" title="meta-llama/Llama-3.2-3B-Instruct">LLaMA 3.2 3B</option>
            <option value="mistral" title="TheBloke/Mistral-7B-Instruct-v0.2-GPTQ">Mistral 7B</option>
            <option value="qwen" title="Qwen/Qwen2.5-7B-Instruct">Qwen 2.5 7B</option>
          </GlassSelect>
        </GlassField>

        <GlassButton onClick={generateSummary} disabled={loading || !month}>
          {loading ? "Generating…" : "Generate summary"}
        </GlassButton>
      </GlassPanel>

      {error && <AlertBanner tone="error">{error}</AlertBanner>}

      {summary && (
        <GlassPanel strong className="space-y-4">
          <div className="flex flex-wrap items-center gap-2">
            <span
              title={
                model === "llama"
                  ? "meta-llama/Llama-3.2-3B-Instruct"
                  : model === "mistral"
                    ? "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
                    : "Qwen/Qwen2.5-7B-Instruct"
              }
            >
              <GlassBadge tone="accent">
                {model === "llama" ? "LLaMA 3.2" : model === "mistral" ? "Mistral 7B" : "Qwen 2.5 7B"}
              </GlassBadge>
            </span>
            <GlassBadge>{summary.prompt_strategy}</GlassBadge>
            <span className="text-xs text-white/40">
              {summary.metadata.inference_time_sec.toFixed(2)}s ·{" "}
              {summary.metadata.tokens_per_sec.toFixed(1)} tok/s ·{" "}
              {summary.metadata.total_tokens} tokens
            </span>
          </div>
          <div className="glass-inset whitespace-pre-wrap break-words rounded-2xl p-5 text-sm leading-relaxed text-white/85">
            {summary.summary}
          </div>
          <GlassButton variant="secondary" onClick={generateInsights} disabled={insLoading}>
            {insLoading ? "Generating insights…" : "Generate financial insights"}
          </GlassButton>
        </GlassPanel>
      )}

      {insights && (
        <GlassPanel>
          <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-white/50">
            Insights
          </h2>
          <ol className="list-decimal space-y-3 pl-5 text-white/85">
            {insights.insights.map((line, i) => (
              <li key={i}>{line}</li>
            ))}
          </ol>
          <p className="mt-4 text-xs text-white/40">
            {insights.metadata.inference_time_sec.toFixed(2)}s ·{" "}
            {insights.metadata.tokens_per_sec.toFixed(1)} tok/s
          </p>
        </GlassPanel>
      )}
    </div>
  );
}
