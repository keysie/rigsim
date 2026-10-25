import { useState, type ReactNode } from "react";
import { NumInput } from "./ui";

export type LayoutBox = { kind: string; label: string; gb: number; exact: boolean; slots?: number };
export type Layout = {
  cards: { id: string; name: string; capacity_gb: number; boxes: LayoutBox[]; total_gb: number; overshoot_gb: number; cache_gb: number }[];
  pools: { cpu: number; name: string; capacity_gb: number; boxes: LayoutBox[]; total_gb: number; overshoot_gb: number; ram_bw_gbs: number;
    parking: { gb: number; slots: number; per_full_conversation_gb: number; fits_full: number; limited_by: string } | null }[];
  links: { a: string; b: string; kind: string; capacity_gbs: number; nominal_gbs: number; worst_gbs: number; ratio: number; parts: Record<string, number>; verdict: string }[];
  hit: number; hit_guessed: number; held: number; headline: string[];
  fixed_defaults: Record<string, { value: number | null; unit: string; help: string }>; fixed: Record<string, number | null>; worst_card: string | null;
};

const COLORS: Record<string, string> = {
  dense: "#5b6763", experts: "#1F6B5B", kv: "#2F6FB3", parking: "#B36B2F", scratch: "#8a8f8c", overhead: "#b9bdbb", reserve: "#e4e7e5", foreign: "#6b4f8a",
};
const LEGEND: [string, string][] = [["dense", "dense weights"], ["experts", "experts (cache / pack)"], ["kv", "KV cache"], ["parking", "parked conversations"],
  ["scratch", "scratch"], ["overhead", "context, display, OS"], ["reserve", "kept free"], ["foreign", "other engines"]];

function Stack({ title, capacity, boxes, total, overshoot, note, compact, children }: { title: string; capacity: number; boxes: LayoutBox[]; total: number;
  overshoot: number; note?: string; compact?: boolean; children?: ReactNode }) {
  const scale = Math.max(capacity, total);
  return (
    <div className={`${compact ? "min-w-[130px]" : "min-w-[150px]"} flex-1`}>
      <div className="text-sm font-medium">{title}</div>
      <div className="text-xs text-muted">{total.toFixed(1)} of {capacity.toFixed(0)} GB{note ? ` · ${note}` : ""}</div>
      <div className={`relative mt-1 ${compact ? "h-44" : "h-56"} w-full rounded border ${overshoot > 0 ? "border-red" : "border-line"} bg-paper`} title={`${title}: ${total.toFixed(1)} of ${capacity} GB`}>
        <div className="absolute inset-x-0 bottom-0 flex flex-col-reverse" style={{ height: `${(total / scale) * 100}%` }}>
          {boxes.map((b, i) => (
            <div key={i} title={`${b.label}: ${b.gb.toFixed(2)} GB${b.exact ? "" : " (estimate)"}`}
                 className={`flex items-center overflow-hidden px-1 text-[10px] leading-tight text-white ${b.exact ? "" : "border-y border-dashed border-white/50"}`}
                 style={{ height: `${(b.gb / total) * 100}%`, background: COLORS[b.kind] ?? "#999", color: ["reserve", "overhead"].includes(b.kind) ? "#2b2f2d" : undefined,
                          backgroundImage: b.kind === "parking" && b.slots ? `repeating-linear-gradient(90deg, transparent 0 ${100 / b.slots - 2}%, rgba(255,255,255,.6) ${100 / b.slots - 2}% ${100 / b.slots}%)` : undefined }}>
              {(b.gb / total) > 0.03 && <span className="truncate">{b.label} · {b.gb.toFixed(b.gb < 1 ? 2 : 1)}</span>}
            </div>
          ))}
        </div>
        {/* capacity line and the overshoot beyond it */}
        {overshoot > 0 && (
          <>
            <div className="absolute inset-x-0 border-t-2 border-red" style={{ bottom: `${(capacity / scale) * 100}%` }} />
            <div className="absolute inset-x-0 top-0 flex items-center justify-center bg-[repeating-linear-gradient(45deg,rgba(201,68,54,.35)_0_6px,transparent_6px_12px)] text-xs font-semibold text-red"
                 style={{ height: `${(overshoot / scale) * 100}%` }}>OVER BY {overshoot.toFixed(1)} GB</div>
          </>
        )}
      </div>
      {children ? <div className="mt-2 flex flex-col gap-1">{children}</div> : null}
    </div>
  );
}

function Link({ l }: { l: Layout["links"][number] }) {
  const pct = Math.round(l.ratio * 100);
  const over = l.ratio > 1;
  return (
    <div className={`rounded border px-2 py-1 text-xs ${over ? "border-red bg-red-soft" : "border-line bg-paper"}`} title={Object.entries(l.parts).map(([k, v]) => `${k}: ${v.toFixed(1)} GB/s`).join("\n")}>
      <div className="flex items-center gap-2">
        <span className="font-medium">{l.a} ↔ {l.b}</span>
        <span className="text-muted">{l.kind.toUpperCase()} {l.capacity_gbs.toFixed(1)} GB/s</span>
        <span className={`ml-auto font-semibold ${over ? "text-red" : ""}`}>{over ? `oversubscribed by ${pct - 100} %` : `worst case ${pct} %`}</span>
      </div>
      <div className="mt-1 h-1.5 w-full rounded bg-line">
        <div className={`h-1.5 rounded ${over ? "bg-red" : "bg-mask"}`} style={{ width: `${Math.min(100, pct)}%` }} />
      </div>
    </div>
  );
}

export default function StrataLayout({ layout, fixed, onFixed, hitGuessed, compact = false }: {
  layout: Layout; fixed: Record<string, number | null>; onFixed: (f: Record<string, number | null>) => void; hitGuessed: number;
  compact?: boolean;   // side by side with the settings: link bars under their stacks, shorter stacks
}) {
  const [open, setOpen] = useState(false);
  const bad = layout.cards.some((c) => c.overshoot_gb > 0) || layout.pools.some((p) => p.overshoot_gb > 0) || layout.links.some((l) => l.ratio > 1);
  return (
    <div className="mt-4 rounded-lg border border-line bg-panel p-3" data-testid="strata-layout">
      <div className="flex flex-wrap items-baseline gap-x-3">
        <div className="font-medium">Memory and links, static fit</div>
        <div className="text-xs text-muted">expert-cache hit {Math.round(layout.hit * 100)} % {layout.hit !== hitGuessed ? "(set by hand)" : `(guessed from ${Math.round(layout.held * 100)} % of the experts held)`}</div>
      </div>
      {layout.headline.length > 0 && (
        <div className={`mt-2 rounded px-3 py-2 text-sm ${bad ? "bg-red-soft text-red font-medium" : "bg-amber-soft text-amber"}`} role="status">
          {layout.headline.map((h, i) => <div key={i}>{h}</div>)}
        </div>
      )}
      {!layout.headline.length && <div className="mt-2 text-xs text-mask-deep">everything fits; no link above its usable rate in the worst case</div>}
      <div className={`mt-3 flex flex-wrap ${compact ? "gap-3" : "gap-4"}`}>
        {layout.cards.map((c) => (
          <Stack key={c.id} compact={compact} title={`${c.id} · ${c.name}`} capacity={c.capacity_gb} boxes={c.boxes} total={c.total_gb} overshoot={c.overshoot_gb} note={`cache ${c.cache_gb.toFixed(1)} GB`}>
            {compact ? layout.links.filter((l) => l.a === c.id).map((l, i) => <Link key={i} l={l} />) : null}
          </Stack>
        ))}
        {!compact && (
          <div className="flex w-56 flex-col justify-center gap-2">
            {layout.links.map((l, i) => <Link key={i} l={l} />)}
          </div>
        )}
        {layout.pools.map((p) => (
          <Stack key={p.cpu} compact={compact} title={`host RAM · ${p.name}`} capacity={p.capacity_gb} boxes={p.boxes} total={p.total_gb} overshoot={p.overshoot_gb}
                 note={p.parking ? `parking fits ${p.parking.fits_full} full conversations (${p.parking.per_full_conversation_gb.toFixed(2)} GB each, limited by ${p.parking.limited_by})` : "no parking"}>
            {compact ? layout.links.filter((l) => !layout.cards.some((c) => c.id === l.a)).map((l, i) => <Link key={i} l={l} />) : null}
          </Stack>
        ))}
      </div>
      <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-muted">
        {LEGEND.map(([k, t]) => <span key={k} className="flex items-center gap-1"><span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: COLORS[k] }} />{t}</span>)}
        <span>dashed border = estimate</span>
      </div>
      <details className="mt-3" open={open} onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
        <summary className="cursor-pointer text-sm text-muted">Fixed allocations and assumptions (override the estimates)</summary>
        <div className="mt-2 grid gap-x-4 gap-y-2 sm:grid-cols-2 lg:grid-cols-3">
          {Object.entries(layout.fixed_defaults).map(([k, d]) => (
            <label key={k} className="text-xs">
              <div className="flex items-baseline gap-2"><span className="font-medium">{k}</span><span className="text-muted">{d.unit}</span></div>
              <NumInput int={false} step="any" min={0} allowEmpty className="mt-0.5 w-28" aria-label={`fixed ${k}`} value={fixed[k] ?? null}
                        placeholder={d.value === null ? "guessed" : String(d.value)}
                        onChange={(v) => { const f = { ...fixed }; if (v === null) delete f[k]; else f[k] = v; onFixed(f); }} />
              <div className="text-muted">{d.help}</div>
            </label>
          ))}
        </div>
        <div className="mt-2 text-xs text-muted">Empty = the default above. These values move with driver, OS and engine version; they are not collected, only shown.</div>
      </details>
    </div>
  );
}
