"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/", label: "Dashboard", icon: "◉" },
  { href: "/summaries", label: "AI Summaries", icon: "✦" },
  // { href: "/comparison", label: "Model comparison", icon: "⇄" },
  { href: "/clusters", label: "Semantic clusters", icon: "◎" },
  { href: "/experiments", label: "Experiment results", icon: "▤" },
];

export function Sidebar() {
  const pathname = usePathname();
  return (
    <aside className="relative z-20 flex w-[17rem] shrink-0 flex-col p-4">
      <div className="glass-strong flex h-full flex-col rounded-[1.75rem] p-4">
        <div className="mb-8 flex items-center gap-3 px-2 pt-2">
          <div className="relative flex h-11 w-11 items-center justify-center">
            <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-sky-400/50 to-violet-500/50 blur-md" />
            <div className="glass-strong relative flex h-11 w-11 items-center justify-center rounded-2xl text-lg font-bold text-white">
              F
            </div>
          </div>
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-white/40">
              FinSight
            </p>
            <p className="text-sm font-semibold leading-tight text-white/90">
              Financial Insight
            </p>
          </div>
        </div>

        <nav className="flex flex-1 flex-col gap-1">
          {links.map(({ href, label, icon }) => {
            const active = pathname === href;
            return (
              <Link
                key={href}
                href={href}
                className={`flex items-center gap-3 rounded-2xl px-3 py-2.5 text-sm font-medium transition-all duration-200 ${
                  active
                    ? "glass-nav-active text-white"
                    : "text-white/60 hover:bg-white/10 hover:text-white/90"
                }`}
              >
                <span
                  className={`flex h-7 w-7 items-center justify-center rounded-xl text-xs ${
                    active ? "bg-white/20" : "bg-white/5"
                  }`}
                >
                  {icon}
                </span>
                {label}
              </Link>
            );
          })}
        </nav>

        <p className="mt-4 px-2 text-[10px] leading-relaxed text-white/30">
          Liquid Glass UI · PaySim analytics
        </p>
      </div>
    </aside>
  );
}
