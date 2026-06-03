export function MetricCard({
  label,
  value,
  prefix = "",
  warn,
}: {
  label: string;
  value: string;
  prefix?: string;
  warn?: boolean;
}) {
  return (
    <div
      className={`rounded-3xl p-5 transition ${
        warn
          ? "border border-amber-400/35 bg-amber-400/10 backdrop-blur-xl"
          : "glass"
      }`}
    >
      <p className="text-xs font-medium uppercase tracking-wider text-white/45">
        {label}
      </p>
      <p className="mt-2 text-2xl font-semibold tabular-nums tracking-tight text-white">
        {prefix}
        {value}
      </p>
    </div>
  );
}
