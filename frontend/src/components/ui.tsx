import { useEffect, useState } from "react";
import type { ReactNode } from "react";

export function Field({ label, children, hint }: { label: string; children: ReactNode; hint?: ReactNode }) {
  return (
    <label className="flex flex-col gap-1">
      <span>{label}</span>
      {children}
      {hint && <span className="text-xs text-muted">{hint}</span>}
    </label>
  );
}

export function Select<T extends string | number>({ value, onChange, options, disabled, ariaLabel, className }: {
  value: T; onChange: (v: T) => void; options: { value: T; label: string; disabled?: boolean }[]; disabled?: boolean;
  ariaLabel?: string; className?: string;
}) {
  const numeric = typeof value === "number";
  return (
    <select className={className ?? "min-w-0"} value={String(value)} disabled={disabled} aria-label={ariaLabel}
            onChange={(e) => onChange((numeric ? Number(e.target.value) : e.target.value) as T)}>
      {options.map((o) => <option key={String(o.value)} value={String(o.value)} disabled={o.disabled}>{o.label}</option>)}
    </select>
  );
}

/** PCIe link drawn as its lanes: one tick per lane, the generation as the shade. */
export function LinkTicks({ gen, width, bw }: { gen: number; width: number; bw: number }) {
  const shade = ["", "#C9D1CE", "#A9B8B3", "#7FA399", "#4C8A7A", "#1F6B5B", "#154A3F"][Math.min(6, Math.max(0, gen))];
  return (
    <span className="inline-flex items-center gap-2" title={`PCIe ${gen}.0 x${width}: ${bw} GB/s each way`}>
      <svg width={16 * 5 + 2} height="12" aria-hidden="true">
        {Array.from({ length: 16 }).map((_, i) => (
          <rect key={i} x={i * 5} y={i < width ? 1 : 5} width="4" height={i < width ? 10 : 2} rx="1"
                fill={i < width ? shade : "#E3E8E6"} />
        ))}
      </svg>
      <span className="text-xs text-muted whitespace-nowrap">{gen ? `Gen${gen} x${width}` : "SATA"} · {bw} GB/s</span>
    </span>
  );
}

export function Notices({ errors, warnings }: { errors: string[]; warnings: string[] }) {
  if (!errors.length && !warnings.length) return null;
  return (
    <div className="mt-3 flex flex-col gap-1 text-sm">
      {errors.map((e, i) => <div key={`e${i}`} className="rounded bg-red-soft px-3 py-1.5 text-red">{e}</div>)}
      {warnings.map((w, i) => <div key={`w${i}`} className="rounded bg-amber-soft px-3 py-1.5 text-amber">{w}</div>)}
    </div>
  );
}

export function Bar({ value, max, label, danger }: { value: number; max: number; label: string; danger?: boolean }) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  return (
    <div className="flex items-center gap-2 text-xs">
      <div className="h-2 flex-1 rounded bg-line/60 overflow-hidden">
        <div className={`h-full ${danger || value > max ? "bg-red" : "bg-mask"}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-muted whitespace-nowrap">{label}</span>
    </div>
  );
}


/** A number field that keeps what the user types, sends only usable numbers (clamped to min/max) to the store,
 *  and shows the stored value again when the field loses focus. Typing past the limits cannot reach the model. */
export function NumInput({ value, onChange, min, max, step, int = true, allowEmpty = false, className, placeholder, ...rest }: {
  value: number | null | undefined; onChange: (v: number | null) => void; min?: number; max?: number; step?: number | "any";
  int?: boolean; allowEmpty?: boolean; className?: string; placeholder?: string; "aria-label"?: string; disabled?: boolean;
}) {
  const show = (v: number | null | undefined) => (v == null || !Number.isFinite(v) ? "" : String(v));
  const [text, setText] = useState(show(value));
  const [focus, setFocus] = useState(false);
  useEffect(() => { if (!focus) setText(show(value)); }, [value, focus]);
  const clamp = (n: number) => { let x = int ? Math.round(n) : n; if (min != null) x = Math.max(min, x); if (max != null) x = Math.min(max, x); return x; };
  return (
    <input type="number" {...rest} className={className} placeholder={placeholder} value={text} min={min} max={max} step={step}
           onFocus={() => setFocus(true)} onBlur={() => setFocus(false)}
           onChange={(e) => {
             const t = e.target.value; setText(t);
             if (t === "") { if (allowEmpty) onChange(null); return; }
             const n = Number(t); if (Number.isFinite(n)) onChange(clamp(n));
           }} />
  );
}
