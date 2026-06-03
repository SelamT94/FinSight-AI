"use client";

import { useEffect, useState, useMemo } from "react";
import {
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { api } from "@/lib/api";
import { CHART_COLORS, tooltipStyle } from "@/lib/chartTheme";
import { MetricCard } from "@/components/MetricCard";
import {
  AlertBanner,
  GlassField,
  GlassPanel,
  GlassSelect,
  LoadingPulse,
  PageHeader,
} from "@/components/Glass";

type Dashboard = {
  month: string;
  total_spent: number;
  total_received: number;
  net_flow: number;
  transaction_count: number;
  anomaly_count: number;
  category_pie: { name: string; value: number }[];
  top_transactions: Record<string, unknown>[];
  transactions_sample: Record<string, unknown>[];
};

function fmtMoney(n: number) {
  return n.toLocaleString(undefined, {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  });
}

export default function DashboardPage() {
  const [months, setMonths] = useState<string[]>([]);
  const [month, setMonth] = useState("");
  const [data, setData] = useState<Dashboard | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [sortConfig, setSortConfig] = useState<{ key: string; direction: "asc" | "desc" } | null>(null);

  const requestSort = (key: string) => {
    let direction: "asc" | "desc" = "asc";
    if (sortConfig && sortConfig.key === key && sortConfig.direction === "asc") {
      direction = "desc";
    }
    setSortConfig({ key, direction });
  };

  const sortedTransactions = useMemo(() => {
    if (!data?.transactions_sample) return [];
    let sortableItems = [...data.transactions_sample];
    if (sortConfig !== null) {
      sortableItems.sort((a, b) => {
        const aVal = a[sortConfig.key];
        const bVal = b[sortConfig.key];
        
        const aNum = Number(aVal);
        const bNum = Number(bVal);
        if (!isNaN(aNum) && !isNaN(bNum)) {
          return sortConfig.direction === "asc" ? aNum - bNum : bNum - aNum;
        }

        const aStr = String(aVal ?? "");
        const bStr = String(bVal ?? "");
        if (aStr < bStr) {
          return sortConfig.direction === "asc" ? -1 : 1;
        }
        if (aStr > bStr) {
          return sortConfig.direction === "asc" ? 1 : -1;
        }
        return 0;
      });
    }
    return sortableItems;
  }, [data?.transactions_sample, sortConfig]);

  useEffect(() => {
    api<{ months: string[] }>("/api/months")
      .then((r) => {
        setMonths(r.months);
        if (r.months.length) setMonth(r.months[0]);
      })
      .catch(() => setErr("Could not load months — is the API running?"));
  }, []);

  useEffect(() => {
    if (!month) return;
    setErr(null);
    api<Dashboard>(`/api/dashboard/${encodeURIComponent(month)}`)
      .then(setData)
      .catch((e) => setErr(String(e.message || e)));
  }, [month]);

  if (!months.length && !err) {
    return <LoadingPulse />;
  }

  if (err && !data) {
    return (
      <AlertBanner title="Setup required" tone="warn">
        <p>{err}</p>
        <p className="mt-2">
          From the repo root:{" "}
          <code className="rounded-lg bg-black/20 px-2 py-0.5 font-mono text-xs">
            python3 scripts/run_api.py
          </code>
        </p>
      </AlertBanner>
    );
  }

  return (
    <div className="mx-auto max-w-7xl space-y-8">
      <PageHeader
        title="AI Financial Insight"
        subtitle="Dashboard — monthly overview"
        action={
          <GlassField label="Month" className="min-w-[10rem]">
            <GlassSelect value={month} onChange={(e) => setMonth(e.target.value)}>
              {months.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </GlassSelect>
          </GlassField>
        }
      />

      {data && (
        <>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <MetricCard label="Total spent" value={fmtMoney(data.total_spent)} />
            <MetricCard
              label="Total received"
              value={fmtMoney(data.total_received)}
            />
            <MetricCard label="Net flow" value={fmtMoney(data.net_flow)} />
            <MetricCard
              label="Transactions"
              value={String(data.transaction_count)}
              prefix=""
            />
          </div>


          <div className="grid gap-6 lg:grid-cols-2">
            <GlassPanel>
              <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-white/50">
                Spending by category
              </h2>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={data.category_pie}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius={100}
                      stroke="rgba(255,255,255,0.15)"
                      label={({ name, percent }) =>
                        `${name} ${(percent * 100).toFixed(0)}%`
                      }
                    >
                      {data.category_pie.map((_, i) => (
                        <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip
                      formatter={(v: number) => fmtMoney(v)}
                      {...tooltipStyle}
                    />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </GlassPanel>

            <GlassPanel>
              <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-white/50">
                Top 5 transactions
              </h2>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={data.top_transactions.map((t) => ({
                      name: String(t.category ?? t.type ?? "—").slice(0, 12),
                      amount: Number(t.amount ?? 0),
                    }))}
                  >
                    <XAxis dataKey="name" tick={{ fontSize: 11, fill: "#ffffff" }} />
                    <YAxis tick={{ fontSize: 11, fill: "#ffffff" }} />
                    <Tooltip
                      formatter={(v: number) => fmtMoney(v)}
                      {...tooltipStyle}
                    />
                    <Bar
                      dataKey="amount"
                      fill="#64d2ff"
                      radius={[8, 8, 0, 0]}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </GlassPanel>
          </div>

          <GlassPanel>
            <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-white/50">
              Transactions (sample, up to 500 rows)
            </h2>
            <div className="glass-inset max-h-96 overflow-auto rounded-2xl text-sm">
              <table className="w-full border-collapse text-left">
                <thead className="sticky top-0 bg-white/5 backdrop-blur-md">
                  <tr>
                    {Object.keys(data.transactions_sample[0] || {})
                      .slice(0, 8)
                      .map((k) => (
                        <th
                          key={k}
                          onClick={() => requestSort(k)}
                          className="border-b border-white/10 px-3 py-2.5 font-medium text-white/60 cursor-pointer hover:bg-white/5 transition-colors select-none"
                        >
                          <div className="flex items-center gap-1">
                            {k}
                            <span className="ml-1 text-[10px] opacity-50">
                              {sortConfig?.key === k
                                ? sortConfig.direction === "asc"
                                  ? "▲"
                                  : "▼"
                                : "↕"}
                            </span>
                          </div>
                        </th>
                      ))}
                  </tr>
                </thead>
                <tbody>
                  {sortedTransactions.map((row, i) => (
                    <tr
                      key={i}
                      className="border-b border-white/5 transition hover:bg-white/5"
                    >
                      {Object.keys(data.transactions_sample[0] || {})
                        .slice(0, 8)
                        .map((k, j) => (
                          <td key={j} className="px-3 py-1.5 text-white/75">
                            {String(row[k] ?? "")}
                          </td>
                        ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </GlassPanel>
        </>
      )}
    </div>
  );
}
