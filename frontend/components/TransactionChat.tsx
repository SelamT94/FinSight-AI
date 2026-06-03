"use client";

import { useState, useRef, useEffect } from "react";
import { api } from "@/lib/api";
import { GlassPanel } from "@/components/Glass";

type SimilarRow = Record<string, unknown>;

type Message = {
  id: string;
  role: "user" | "bot";
  text?: string;
  results?: SimilarRow[];
  error?: string;
};

export function TransactionChat() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "init",
      role: "bot",
      text: "Hi! I can find similar transactions using semantic search. For example, ask me for 'Medium withdrawals' or 'Uber rides'.",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const query = input.trim();
    setInput("");
    
    const userMsgId = Date.now().toString();
    setMessages((prev) => [...prev, { id: userMsgId, role: "user", text: query }]);
    setLoading(true);

    const botMsgId = (Date.now() + 1).toString();
    try {
      const r = await api<{ results: SimilarRow[] }>("/api/similar", {
        method: "POST",
        body: JSON.stringify({ query, k: 5 }),
      });
      setMessages((prev) => [
        ...prev,
        {
          id: botMsgId,
          role: "bot",
          results: r.results,
          text: r.results.length === 0 ? "No similar transactions found." : "Here are the most similar transactions I found:",
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { id: botMsgId, role: "bot", error: String(err) },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <GlassPanel className="flex h-[calc(100vh-6rem)] flex-col p-4 w-80 md:w-96 shrink-0 sticky top-8">
      <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-white/50">
        Semantic Search
      </h2>
      
      <div 
        ref={scrollRef}
        className="flex-1 overflow-y-auto space-y-4 pr-2 mb-4 scrollbar-thin scrollbar-thumb-white/10"
      >
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex flex-col ${msg.role === "user" ? "items-end" : "items-start"}`}
          >
            <div
              className={`rounded-2xl px-4 py-3 max-w-[95%] text-sm ${
                msg.role === "user"
                  ? "bg-sky-500/20 text-white rounded-br-sm border border-sky-400/20"
                  : "glass-inset text-white/85 rounded-bl-sm"
              }`}
            >
              {msg.text && <div className="whitespace-pre-wrap">{msg.text}</div>}
              {msg.error && <div className="text-red-300">{msg.error}</div>}
              
              {msg.results && msg.results.length > 0 && (
                <div className="mt-3 space-y-2">
                  {msg.results.map((row, i) => {
                    const date = row.date || row.Date || "";
                    const amt = typeof row.amount === 'number' ? row.amount : Number(row.amount);
                    const isNeg = amt < 0;
                    const cat = row.category || row.Category || "Unknown";
                    const desc = row.merchant_name || row.description || "";
                    
                    return (
                      <div key={i} className="rounded-xl bg-white/5 p-3 text-xs border border-white/5 shadow-sm">
                        <div className="flex justify-between items-start mb-1 gap-2">
                          <span className="font-medium text-white/90 truncate" title={String(desc)}>{String(desc) || String(cat)}</span>
                          <span className={`font-semibold shrink-0 ${isNeg ? "text-red-300" : "text-emerald-300"}`}>
                            {isNaN(amt) ? String(row.amount) : Math.abs(amt).toLocaleString(undefined, { style: "currency", currency: "USD" })}
                          </span>
                        </div>
                        <div className="flex justify-between items-center text-white/40">
                          <span>{String(date)}</span>
                          {desc && <span className="truncate max-w-[100px] text-right" title={String(cat)}>{String(cat)}</span>}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex items-start">
            <div className="glass-inset rounded-2xl rounded-bl-sm px-4 py-3 text-sm text-white/50">
              <span className="animate-pulse">Searching...</span>
            </div>
          </div>
        )}
      </div>

      <form onSubmit={handleSubmit} className="mt-auto relative">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about transactions..."
          disabled={loading}
          className="w-full rounded-2xl bg-white/5 px-4 py-3 pr-12 text-sm text-white placeholder-white/30 border border-white/10 focus:outline-none focus:ring-1 focus:ring-white/20 transition-all disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={!input.trim() || loading}
          className="absolute right-2 top-1/2 -translate-y-1/2 rounded-xl p-1.5 text-white/50 hover:bg-white/10 hover:text-white transition disabled:opacity-50"
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
            <path d="M3.105 2.289a.75.75 0 00-.826.95l1.414 4.925A1.5 1.5 0 005.135 9.25h6.115a.75.75 0 010 1.5H5.135a1.5 1.5 0 00-1.442 1.086l-1.414 4.926a.75.75 0 00.826.95 28.896 28.896 0 0015.293-7.154.75.75 0 000-1.115A28.897 28.897 0 003.105 2.289z" />
          </svg>
        </button>
      </form>
    </GlassPanel>
  );
}
