import { type ReactNode } from "react";

export function GlassPanel({
  children,
  className = "",
  strong,
}: {
  children: ReactNode;
  className?: string;
  strong?: boolean;
}) {
  return (
    <div
      className={`rounded-3xl p-5 ${strong ? "glass-strong" : "glass"} ${className}`}
    >
      {children}
    </div>
  );
}

export function PageHeader({
  title,
  subtitle,
  action,
}: {
  title: string;
  subtitle?: string;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight text-gradient">
          {title}
        </h1>
        {subtitle && (
          <p className="mt-1 text-sm text-white/55">{subtitle}</p>
        )}
      </div>
      {action}
    </div>
  );
}

export function GlassButton({
  children,
  variant = "primary",
  className = "",
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary";
}) {
  return (
    <button
      type="button"
      className={`rounded-2xl px-5 py-2.5 text-sm font-medium disabled:cursor-not-allowed disabled:opacity-40 ${
        variant === "primary" ? "glass-btn-primary" : "glass-btn-secondary"
      } ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}

export function GlassField({
  label,
  children,
  className = "",
}: {
  label: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <label className={`flex flex-col gap-1.5 text-sm ${className}`}>
      <span className="text-xs font-medium uppercase tracking-wider text-white/45">
        {label}
      </span>
      {children}
    </label>
  );
}

export function GlassSelect(
  props: React.SelectHTMLAttributes<HTMLSelectElement>
) {
  return <select className="glass-select" {...props} />;
}

export function GlassTextarea(
  props: React.TextareaHTMLAttributes<HTMLTextAreaElement>
) {
  return <textarea className="glass-input resize-y" {...props} />;
}

export function GlassBadge({
  children,
  tone = "default",
}: {
  children: ReactNode;
  tone?: "default" | "accent" | "warn";
}) {
  const tones = {
    default: "bg-white/10 text-white/80 border-white/20",
    accent: "bg-sky-400/20 text-sky-200 border-sky-400/30",
    warn: "bg-amber-400/15 text-amber-200 border-amber-400/25",
  };
  return (
    <span
      className={`inline-flex items-center rounded-full border px-3 py-0.5 text-xs font-medium backdrop-blur-md ${tones[tone]}`}
    >
      {children}
    </span>
  );
}

export function AlertBanner({
  title,
  children,
  tone = "warn",
}: {
  title?: string;
  children: ReactNode;
  tone?: "warn" | "error" | "info";
}) {
  const tones = {
    warn: "border-amber-400/30 bg-amber-400/10 text-amber-100",
    error: "border-red-400/30 bg-red-400/10 text-red-100",
    info: "border-sky-400/30 bg-sky-400/10 text-sky-100",
  };
  return (
    <div className={`glass rounded-2xl p-4 ${tones[tone]}`}>
      {title && <p className="font-medium">{title}</p>}
      <div className={`text-sm ${title ? "mt-1 opacity-90" : ""}`}>{children}</div>
    </div>
  );
}

export function LoadingPulse({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 text-white/50">
      <span className="relative flex h-3 w-3">
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-sky-400/60 opacity-75" />
        <span className="relative inline-flex h-3 w-3 rounded-full bg-sky-400" />
      </span>
      <span className="text-sm">{label}</span>
    </div>
  );
}
