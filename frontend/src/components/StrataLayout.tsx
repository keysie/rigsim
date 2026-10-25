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
  dense: "#5b6763", experts: "#1F6B5B", kv: "#2F6FB3", parking: "#B36B2F", scratch: "#8a8f8c", overhead: "#b9bdbb", reserve: "#dfe3e1", foreign: "#6b4f8a",
};
const DARK_TEXT = new Set(["reserve", "overhead"]);
const OVERHEAD_KINDS = new Set(["scratch", "overhead", "reserve"]);
const LEGEND: [string, string][] = [["dense", "dense weights"], ["experts", "experts (cache / pack)"], ["kv", "KV cache"], ["parking", "parked conversations"],
  ["overhead", "overhead (hover for the parts)"]];

/** Overhead boxes (scratch, context, display, OS, buffers, kept-free margins) become one block where they first appear. */
function group(boxes: LayoutBox[]): { box: LayoutBox; parts?: LayoutBox[] }[] {
  const parts = boxes.filter((b) => OVERHEAD_KINDS.has(b.kind));
  const out: { box: LayoutBox; parts?: LayoutBox[] }[] = [];
  let placed = false;
  for (const b of boxes) {
    if (!OVERHEAD_KINDS.has(b.kind)) { out.push({ box: b }); continue; }
    if (placed) continue;
    placed = true;
    out.push({ box: { kind: "overhead", label: "overhead", gb: parts.reduce((a, x) => a + x.gb, 0), exact: parts.every((x) => x.exact) }, parts });
  }
  return out;
}

const fmt = (gb: number) => gb.toFixed(gb < 1 ? 2 : 1);

function OverheadDetail({ parts, total, side }: { parts: LayoutBox[]; total: number; side: "left" | "right" }) {
  const H = Math.max(160, parts.length * 30);
  return (
    <div className={`absolute top-0 z-30 w-80 rounded-md border border-line bg-panel p-3 text-xs shadow-lg ${side === "left" ? "right-full mr-2" : "left-full ml-2"}`}
         role="tooltip">
      <div className="font-medium">overhead · {fmt(total)} GB</div>
      <div className="mt-2 flex flex-col overflow-hidden rounded border border-line" style={{ height: H }}>
        {[...parts].reverse().map((p, i) => (
          <div key={i} className={`flex items-center justify-between gap-2 px-2 ${p.exact ? "" : "border-y border-dashed border-white/60"}`}
               style={{ flexGrow: Math.max(p.gb, total * 0.12), flexBasis: 0, minHeight: 26, background: COLORS[p.kind] ?? "#999", color: DARK_TEXT.has(p.kind) ? "#2b2f2d" : "white" }}>
            <span className="leading-tight">{p.label}</span><span className="shrink-0 tabular-nums">{fmt(p.gb)} GB{p.exact ? "" : " · est."}</span>
          </div>
        ))}
      </div>
      <div className="mt-1 text-muted">heights are stretched so every part is readable; est. = estimate, override under "Fixed allocations"</div>
    </div>
  );
}

function Stack({ title, note, capacity, boxes, total, overshoot, compact, popSide, links }: {
  title: string; note: string; capacity: number; boxes: LayoutBox[]; total: number; overshoot: number; compact: boolean;
  popSide: "left" | "right"; links?: ReactNode;
}) {
  const [hover, setHover] = useState(false);
  const scale = Math.max(capacity, total);
  const grouped = group(boxes);
  return (
    <div className={`flex min-w-0 flex-1 flex-col ${compact ? "" : "min-w-[150px]"}`}>
      <div className="truncate text-sm font-medium" title={title}>{title}</div>
      <div className="truncate text-xs text-muted" title={note}>{total.toFixed(1)} of {capacity.toFixed(0)} GB · {note}</div>
      <div className={`relative mt-1 w-full rounded border ${overshoot > 0 ? "border-red" : "border-line"} bg-paper ${compact ? "min-h-[180px] flex-1" : "h-56"}`}
           title={`${title}: ${total.toFixed(1)} of ${capacity} GB`}>
        <div className="absolute inset-x-0 bottom-0 flex flex-col-reverse" style={{ height: `${(total / scale) * 100}%` }}>
          {grouped.map(({ box: b, parts }, i) => {
            const share = b.gb / total;
            const body = (
              <div className={`flex h-full items-center overflow-hidden px-1 text-[10px] leading-tight ${b.exact ? "" : "border-y border-dashed border-white/50"}`}
                   style={{ background: COLORS[b.kind] ?? "#999", color: DARK_TEXT.has(b.kind) ? "#2b2f2d" : "white",
                            backgroundImage: b.kind === "parking" && b.slots ? `repeating-linear-gradient(90deg, transparent 0 ${100 / b.slots - 2}%, rgba(255,255,255,.6) ${100 / b.slots - 2}% ${100 / b.slots}%)` : undefined }}>
                {share > 0.025 && <span className="truncate">{b.label} · {fmt(b.gb)}</span>}
              </div>
            );
            return parts ? (
              <div key={i} className="relative cursor-help" style={{ height: `${share * 100}%`, minHeight: 6 }}
                   title={`overhead ${fmt(b.gb)} GB: ` + parts.map((p) => `${p.label} ${fmt(p.gb)} GB`).join(" · ")}
                   onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)} data-testid="overhead-block">
                {body}
                {hover && <OverheadDetail parts={parts} total={b.gb} side={popSide} />}
              </div>
            ) : (
              <div key={i} style={{ height: `${share * 100}%` }} title={`${b.label}: ${b.gb.toFixed(2)} GB${b.exact ? "" : " (estimate)"}`}>{body}</div>
            );
          })}
        </div>
        {overshoot > 0 && (
          <>
            <div className="absolute inset-x-0 border-t-2 border-red" style={{ bottom: `${(capacity / scale) * 100}%` }} />
            <div className="absolute inset-x-0 top-0 flex items-center justify-center bg-[repeating-linear-gradient(45deg,rgba(201,68,54,.35)_0_6px,transparent_6px_12px)] text-xs font-semibold text-red"
                 style={{ height: `${(overshoot / scale) * 100}%` }}>OVER BY {overshoot.toFixed(1)} GB</div>
          </>
        )}
      </div>
      {links}
    </div>
  );
}

function Link({ l }: { l: Layout["links"][number] }) {
  const pct = Math.round(l.ratio * 100);
  const over = l.ratio > 1;
  return (
    <div className={`h-[46px] rounded border px-2 py-1 text-[11px] leading-tight ${over ? "border-red bg-red-soft" : "border-line bg-paper"}`}
         title={Object.entries(l.parts).map(([k, v]) => `${k}: ${v.toFixed(1)} GB/s`).join("\n")}>
      <div className="flex items-baseline gap-1">
        <span className="truncate font-medium">{l.a} ↔ {l.b}</span>
        <span className={`ml-auto shrink-0 font-semibold ${over ? "text-red" : ""}`} title={over ? `oversubscribed by ${pct - 100} % in the worst case` : `worst case ${pct} % of the usable rate`}>{over ? `+${pct - 100} % over` : `${pct} %`}</span>
      </div>
      <div className="truncate text-muted">{l.kind.toUpperCase()} · {l.capacity_gbs.toFixed(1)} GB/s usable</div>
      <div className="mt-0.5 h-1.5 w-full rounded bg-line"><div className={`h-1.5 rounded ${over ? "bg-red" : "bg-mask"}`} style={{ width: `${Math.min(100, pct)}%` }} /></div>
    </div>
  );
}

export default function StrataLayout({ layout, fixed, onFixed, hitGuessed, compact = false }: {
  layout: Layout; fixed: Record<string, number | null>; onFixed: (f: Record<string, number | null>) => void; hitGuessed: number;
  compact?: boolean;   // side by side with the settings: fills the height it is given; links under their stacks
}) {
  const bad = layout.cards.some((c) => c.overshoot_gb > 0) || layout.pools.some((p) => p.overshoot_gb > 0) || layout.links.some((l) => l.ratio > 1);
  const cardLinks = (id: string) => layout.links.filter((l) => l.a === id);
  const poolLinks = layout.links.filter((l) => !layout.cards.some((c) => c.id === l.a));
  // every stack reserves room for the same number of link rows, so all bars share top and bottom
  const rows = Math.max(1, ...layout.cards.map((c) => cardLinks(c.id).length), poolLinks.length);
  const linkArea = (ls: Layout["links"]) => (
    <div className="mt-2 flex flex-col gap-1" style={{ height: rows * 46 + (rows - 1) * 4 }}>{ls.map((l, i) => <Link key={i} l={l} />)}</div>
  );
  const stacks = [
    ...layout.cards.map((c, i) => (
      <Stack key={c.id} compact={compact} popSide={i === 0 ? "right" : "right"} title={`${c.id} · ${c.name}`} note={`expert cache ${c.cache_gb.toFixed(1)} GB`}
             capacity={c.capacity_gb} boxes={c.boxes} total={c.total_gb} overshoot={c.overshoot_gb} links={compact ? linkArea(cardLinks(c.id)) : undefined} />
    )),
    ...layout.pools.map((p) => (
      <Stack key={`pool-${p.cpu}`} compact={compact} popSide="left" title={`host RAM · ${p.name}`}
             note={p.parking ? `parking fits ${p.parking.fits_full} full conversations (${p.parking.per_full_conversation_gb.toFixed(2)} GB each, limited by ${p.parking.limited_by})` : "no parking"}
             capacity={p.capacity_gb} boxes={p.boxes} total={p.total_gb} overshoot={p.overshoot_gb} links={compact ? linkArea(poolLinks) : undefined} />
    )),
  ];
  return (
    <div className={`rounded-lg border border-line bg-panel p-3 ${compact ? "flex min-h-0 flex-1 flex-col overflow-y-auto" : "mt-4"}`} data-testid="strata-layout">
      <div className="flex flex-wrap items-baseline gap-x-3">
        <div className="font-medium">Memory and links, static fit</div>
        <div className="text-xs text-muted">expert-cache hit {Math.round(layout.hit * 100)} % {layout.hit !== hitGuessed ? "(set by hand)" : `(guessed from ${Math.round(layout.held * 100)} % of the experts held)`}</div>
      </div>
      {layout.headline.length > 0 ? (
        <div className={`mt-2 rounded px-3 py-1.5 text-sm ${bad ? "bg-red-soft font-medium text-red" : "bg-amber-soft text-amber"}`} role="status">
          {layout.headline.map((h, i) => <div key={i}>{h}</div>)}
        </div>
      ) : <div className="mt-1 text-xs text-mask-deep">everything fits; no link above its usable rate in the worst case</div>}
      {compact ? (
        <div className="mt-2 flex min-h-[300px] flex-1 gap-3">{stacks}</div>
      ) : (
        <div className="mt-3 flex flex-wrap gap-4">
          {stacks.slice(0, layout.cards.length)}
          <div className="flex w-56 flex-col justify-center gap-2">{layout.links.map((l, i) => <Link key={i} l={l} />)}</div>
          {stacks.slice(layout.cards.length)}
        </div>
      )}
      <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-muted">
        {LEGEND.map(([k, t]) => <span key={k} className="flex items-center gap-1"><span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: COLORS[k] }} />{t}</span>)}
        <span>dashed = estimate</span>
      </div>
      <details className="mt-2">
        <summary className="cursor-pointer text-sm text-muted">Fixed allocations and assumptions (override the estimates)</summary>
        <div className="mt-2 grid max-h-72 gap-x-4 gap-y-2 overflow-y-auto sm:grid-cols-2">
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
