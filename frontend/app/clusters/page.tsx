"use client";

import { useEffect, useState } from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ZAxis,
  BarChart,
  Bar,
  Cell,
} from "recharts";
import { api } from "@/lib/api";
import { CHART_COLORS, tooltipStyle } from "@/lib/chartTheme";
import {
  AlertBanner,
  GlassPanel,
  PageHeader,
} from "@/components/Glass";
import { TransactionChat } from "@/components/TransactionChat";

type Card = {
  id: string;
  name: string;
  size: number;
  top_category: string;
  mean_amount: number;
};

type ScatterPayload = {
  mode: "tsne" | "bubble";
  points: { x: number; y: number; cluster: number; amount: number; category: string }[];
  bubbles: {
    id: string;
    label: string;
    size: number;
    mean_amount: number;
    top_category: string;
  }[];
};

export default function ClustersPage() {
  const [cards, setCards] = useState<Card[]>([]);
  const [scatter, setScatter] = useState<ScatterPayload | null>(null);

  useEffect(() => {
    api<{ cards: Card[] }>("/api/clusters/summary")
      .then((r) => setCards(r.cards))
      .catch(() => setCards([]));
    api<ScatterPayload>("/api/clusters/scatter?max_points=400")
      .then(setScatter)
      .catch(() => setScatter(null));
  }, []);

  const bubbleData =
    scatter?.bubbles.map((b) => ({
      name: b.label.slice(0, 24),
      size: b.size,
      mean: b.mean_amount,
      fill: CHART_COLORS[parseInt(b.id, 10) % CHART_COLORS.length],
    })) ?? [];

  return (
    <div className="mx-auto max-w-7xl flex gap-6 items-start">
      <div className="flex-1 space-y-8 min-w-0">
      <PageHeader
        title="Semantic clusters"
        subtitle="Cluster summaries, embedding projection, and similarity search"
      />

      {cards.length === 0 && (
        <AlertBanner tone="warn" title="No cluster labels">
          Run <code className="font-mono text-xs">02_embeddings.ipynb</code> first.
        </AlertBanner>
      )}

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {cards.map((c) => (
          <GlassPanel key={c.id} className="hover:bg-white/[0.14] transition">
            <p className="text-xs font-medium uppercase tracking-wider text-white/40">
              Cluster {c.id}
            </p>
            <p className="mt-1 font-semibold text-white">{c.name}</p>
            <p className="mt-3 text-sm text-white/55">
              Size: <strong className="text-white/90">{c.size}</strong>
            </p>
            <p className="text-sm text-white/55">
              Top category:{" "}
              <strong className="text-white/90">{c.top_category || "—"}</strong>
            </p>
            <p className="text-sm text-white/55">
              Mean:{" "}
              <strong className="text-white/90">
                {c.mean_amount.toLocaleString(undefined, {
                  style: "currency",
                  currency: "USD",
                })}
              </strong>
            </p>
          </GlassPanel>
        ))}
      </div>

      <GlassPanel>
        <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-white/50">
          {scatter?.mode === "tsne"
            ? "t-SNE of transaction embeddings"
            : "Cluster sizes"}
        </h2>
        {scatter?.mode === "tsne" && scatter.points.length > 0 ? (
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 0 }}>
                <XAxis type="number" dataKey="x" name="x" />
                <YAxis type="number" dataKey="y" name="y" />
                <ZAxis range={[60, 60]} />
                <Tooltip
                  cursor={{ strokeDasharray: "3 3" }}
                  content={({ payload }) => {
                    const p = payload?.[0]?.payload as
                      | { category?: string; cluster?: number; amount?: number }
                      | undefined;
                    if (!p) return null;
                    return (
                      <div className="glass-strong rounded-xl px-3 py-2 text-xs text-white/90">
                        {p.category} · cluster {p.cluster} · $
                        {typeof p.amount === "number"
                          ? p.amount.toFixed(2)
                          : p.amount}
                      </div>
                    );
                  }}
                />
                <Scatter data={scatter.points} fill="#64d2ff" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={bubbleData} layout="vertical">
                <XAxis type="number" />
                <YAxis dataKey="name" type="category" width={140} tick={{ fontSize: 11 }} />
                <Tooltip {...tooltipStyle} />
                <Bar dataKey="size" name="Cluster size" radius={[0, 6, 6, 0]}>
                  {bubbleData.map((e, i) => (
                    <Cell key={i} fill={e.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </GlassPanel>

      </div>
      <TransactionChat />
    </div>
  );
}
