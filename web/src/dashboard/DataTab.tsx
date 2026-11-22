import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Region, TaskSpec } from "../protocol";
import { epNumber } from "./Episodes";
import { fmt } from "./util";

/** Data tab: dataset health (GET /api/health) and exports (POST /api/export*). Same selection rules as the review
 *  card and the Episodes tab (audit.select), so every number here matches what an export would write. */

interface HealthRow {
  file: string; success: boolean; duration_s: number; used: boolean; verdict: string | null;
  start_poses: Record<string, [number, number, number]>; created_at: string | null; operator: string | null;
  contact_events: Record<string, number>; safety_events: number; stage_times: (number | null)[];
  task_version: string | null; quality: number | null;
}
interface Health {
  episodes: HealthRow[]; count: number; used: number; success_rate: number; total_s: number; contact_kinds: string[];
  operators: Record<string, { takes: number; success: number; success_rate: number; mean_duration_s: number; takes_per_hour: number | null }>;
  task_versions: Record<string, { task: string | null; count: number }>;
}
type Rules = { include_failures: boolean; max_contact_frac: Record<string, number>; cut_contacts: string[]; contact_margin_s: number;
  trim_idle: boolean; shared_frames: "operator" | "all"; until_stage: number | null; until_stage_tail_s: number; cut_labels: string[] };
type Layout = { cameras: string[] | null; image_size: [number, number]; state: string[]; env_state: boolean; action: "joint" | "ee";
  action_label: "recorded" | "executed"; randomize_visuals: boolean; segments: "takes" | "annotations";
  fps: number | null };  // null = the recording's rate (no form field yet: set it with `xrc export --fps`)
interface Job { state: "running" | "done" | "failed"; done: number; total: number; error: string | null;
  manifest: { episodes: number; steps: number; fps: number; constant_channels: Record<string, string[]>; task_versions: unknown } | null }
interface Exporters { exporters: string[]; layout: Layout; rules: Rules; jobs: Record<string, Job> }
interface Preview { episodes: Record<string, { verdict: string | null; spans: [number, number][]; fps: number }>; segments: number; steps: number; task_versions: string[] }

const STATE_CHANNELS = ["joint_pos", "gripper_width", "joint_vel", "ee_pose", "ee_wrench"];
const kindLabel = (k: string) => k.replace(/_/g, " ");
const pct = (v: number) => `${Math.round(v * 100)}%`;
const q = (dataset: string) => (dataset ? `?dataset=${encodeURIComponent(dataset)}` : "");

// ---- tooltip: one per page, positioned at the pointer; content set as text (labels are data) -------------------
type Tip = { x: number; y: number; rows: [string, string][] } | null;
function useTip() {
  const [tip, setTip] = useState<Tip>(null);
  const show = useCallback((e: React.PointerEvent | React.FocusEvent, rows: [string, string][]) => {
    const r = (e.target as Element).getBoundingClientRect();
    const x = "clientX" in e ? e.clientX : r.left + r.width / 2, y = "clientY" in e ? e.clientY : r.top;
    setTip({ x, y, rows });
  }, []);
  const hide = useCallback(() => setTip(null), []);
  const node = tip && (
    <div className="viz-tip" style={{ left: tip.x + 14, top: tip.y + 14 }} role="status">
      {tip.rows.map(([v, k], i) => <div key={i}><b>{v}</b> <span>{k}</span></div>)}
    </div>
  );
  return { show, hide, node };
}
type TipApi = ReturnType<typeof useTip>;

/** Clean tick values for 0..max. */
function ticks(max: number, n = 4): number[] {
  if (max <= 0) return [0];
  const raw = max / n, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [0];
  while (out[out.length - 1] < max - 1e-9) out.push(+(out[out.length - 1] + step).toFixed(6));  // the top tick covers max
  return out;
}

export function DataTab({ dataset, datasetTitle, task }: { dataset: string; datasetTitle: string; task: TaskSpec | null }) {
  const [health, setHealth] = useState<Health | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const refresh = useCallback(() => {
    setLoading(true);
    fetch(`/api/health${q(dataset)}`, { cache: "no-store" }).then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status}`))))
      .then((h: Health) => { setHealth(h); setErr(null); }).catch((e: Error) => setErr(e.message)).finally(() => setLoading(false));
  }, [dataset]);
  useEffect(refresh, [refresh]);
  const tip = useTip();
  const rows = useMemo(() => (health?.episodes ?? []).slice().sort((a, b) => (a.created_at ?? a.file).localeCompare(b.created_at ?? b.file)), [health]);
  const versions = Object.entries(health?.task_versions ?? {});
  return (
    <div className={`health ${loading && health ? "stale" : ""}`}>
      <div className="card health-head">
        <div className="card-head"><h2>Dataset health · {datasetTitle}</h2><button className="btn ghost small" onClick={refresh}>⟳</button></div>
        {err && <div className="alert warn small">Couldn't load the dataset: {err}</div>}
        {health && health.count === 0 && <p className="muted">No episodes in this folder yet.</p>}
        {health && health.count > 0 && (
          <div className="tiles">
            <div className="tile"><span>Takes</span><b>{health.count.toLocaleString()}</b></div>
            <div className="tile"><span>Success</span><b>{pct(health.success_rate)}</b></div>
            <div className="tile"><span>Used for training</span><b>{health.used.toLocaleString()}</b><small>{pct(health.used / health.count)} of takes</small></div>
            <div className="tile"><span>Recorded</span><b>{fmt(health.total_s)}</b><small>min:s</small></div>
            <div className={`tile ${versions.length > 1 ? "warn" : ""}`}><span>Task versions</span><b>{versions.length}</b>
              {versions.length > 1 && <small>⚠ mixed: exports refuse this unless allowed</small>}</div>
          </div>
        )}
      </div>
      {health && health.count > 0 && <>
        <div className="viz-grid">
          <Coverage rows={rows} task={task} tip={tip} />
          <Durations rows={rows} tip={tip} />
          <SuccessOverTime rows={rows} tip={tip} />
          <Contacts rows={rows} kinds={health.contact_kinds} tip={tip} />
        </div>
        <Operators ops={health.operators} />
        <EpisodeTable rows={rows} />
      </>}
      {tip.node}
    </div>
  );
}

// ---- start-state coverage: one small multiple per object, top-down (far table edge up, operator's left left) ----
function Coverage({ rows, task, tip }: { rows: HealthRow[]; task: TaskSpec | null; tip: TipApi }) {
  const objects = useMemo(() => {
    const seen = new Set<string>();
    for (const r of rows) for (const k of Object.keys(r.start_poses ?? {})) seen.add(k);
    return [...seen];
  }, [rows]);
  if (!objects.length) return null;
  const shown = objects.slice(0, 6);
  return (
    <div className="card viz wide">
      <div className="card-head"><h3>Start positions</h3>
        <span className="viz-legend"><i className="dot-key fill" /> success <i className="dot-key ring" /> failure <i className="rect-key" /> current start region</span>
      </div>
      <p className="muted small">Where each object was at the start of every take, seen from above (far table edge up). Gaps here are layouts a policy never saw.</p>
      <div className="multiples">
        {shown.map((o) => <CoverageCell key={o} name={o} rows={rows} region={task?.spawn?.[o] ?? null}
          label={task?.objects.find((x) => x.name === o)?.label ?? o} tip={tip} />)}
      </div>
      {objects.length > shown.length && <p className="muted small">{objects.length - shown.length} more objects not shown.</p>}
    </div>
  );
}

function CoverageCell({ name, label, rows, region, tip }: { name: string; label: string; rows: HealthRow[]; region: Region | null; tip: TipApi }) {
  const pts = rows.filter((r) => r.start_poses?.[name]).map((r) => ({ r, p: r.start_poses[name] }));
  // Bounds: the points and the start region, padded; same scale on both axes (it's a map of the table).
  const xs = pts.map((d) => d.p[0]), ys = pts.map((d) => d.p[1]);
  if (region) { xs.push(region.center[0] - region.half[0], region.center[0] + region.half[0]); ys.push(region.center[1] - region.half[1], region.center[1] + region.half[1]); }
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  const span = Math.max(0.06, Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) * 1.25;
  const W = 200, H = 200, s = W / span;
  // screen x = -world y (operator's left on the left), screen y = -world x (far edge up)
  const sx = (y: number) => W / 2 - (y - cy) * s, sy = (x: number) => H / 2 - (x - cx) * s;
  return (
    <figure className="cell">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Start positions of ${label}`}>
        <rect x={0} y={0} width={W} height={H} className="plot-bg" />
        {region && <rect className="region" x={sx(region.center[1] + region.half[1])} y={sy(region.center[0] + region.half[0])}
          width={2 * region.half[1] * s} height={2 * region.half[0] * s} />}
        {pts.map(({ r, p }) => (
          <g key={r.file} tabIndex={0} className="pt"
            onPointerMove={(e) => tip.show(e, [[epNumber(r.file), r.success ? "success" : "failure"], [`${(p[0] * 100).toFixed(1)}, ${(p[1] * 100).toFixed(1)} cm`, "x, y"], [`${Math.round(p[2] * 180 / Math.PI)}°`, "yaw"]])}
            onFocus={(e) => tip.show(e, [[epNumber(r.file), r.success ? "success" : "failure"]])} onPointerLeave={tip.hide} onBlur={tip.hide}>
            <circle cx={sx(p[1])} cy={sy(p[0])} r={12} className="hit" />
            <circle cx={sx(p[1])} cy={sy(p[0])} r={4.5} className={r.success ? "mark fill" : "mark ring"} />
          </g>
        ))}
        <text x={6} y={H - 6} className="axis-label">{(span * 100).toFixed(0)} cm wide</text>
      </svg>
      <figcaption>{label}</figcaption>
    </figure>
  );
}

// ---- take durations: histogram ----------------------------------------------------------------------------------
function Durations({ rows, tip }: { rows: HealthRow[]; tip: TipApi }) {
  const d = rows.map((r) => r.duration_s);
  const max = Math.max(...d, 1);
  const raw = max / 12, mag = 10 ** Math.floor(Math.log10(raw));
  const bw = [1, 2, 5, 10].map((m) => m * mag).find((v) => v >= raw) ?? raw;
  const n = Math.max(1, Math.ceil((max + 1e-9) / bw));
  const bins = Array.from({ length: n }, (_, i) => ({ lo: i * bw, hi: (i + 1) * bw, all: 0, ok: 0 }));
  for (const r of rows) { const b = bins[Math.min(n - 1, Math.floor(r.duration_s / bw))]; b.all++; if (r.success) b.ok++; }
  const top = Math.max(...bins.map((b) => b.all));
  const yt = ticks(top, 3);
  const W = 420, H = 190, L = 30, B = 24, T = 8, pw = W - L - 6, ph = H - B - T;
  const slot = pw / n, bwpx = Math.min(24, slot - 2);
  const y = (v: number) => T + ph - (v / yt[yt.length - 1]) * ph;
  const median = d.slice().sort((a, b) => a - b)[Math.floor(d.length / 2)];
  return (
    <div className="card viz">
      <div className="card-head"><h3>Take duration</h3><span className="muted small">median {median.toFixed(1)} s</span></div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Histogram of take durations">
        {yt.map((v) => <g key={v}><line x1={L} x2={W - 6} y1={y(v)} y2={y(v)} className="grid" /><text x={L - 6} y={y(v) + 4} className="tick" textAnchor="end">{v}</text></g>)}
        {bins.map((b, i) => {
          const x = L + i * slot + (slot - bwpx) / 2, h = (b.all / yt[yt.length - 1]) * ph;
          return (
            <g key={i} tabIndex={b.all ? 0 : -1} className="bar"
              onPointerMove={(e) => tip.show(e, [[`${b.all} takes`, `${b.lo.toFixed(bw < 1 ? 1 : 0)}–${b.hi.toFixed(bw < 1 ? 1 : 0)} s`], [`${b.ok}`, "successful"]])}
              onFocus={(e) => tip.show(e, [[`${b.all} takes`, `${b.lo}–${b.hi} s`]])} onPointerLeave={tip.hide} onBlur={tip.hide}>
              <rect x={L + i * slot} y={T} width={slot} height={ph} className="hit" />
              {b.all > 0 && <path className="mark" d={roundTop(x, y(b.all), bwpx, h)} />}
            </g>
          );
        })}
        <line x1={L} x2={W - 6} y1={T + ph} y2={T + ph} className="axis" />
        {bins.map((b, i) => (i % Math.ceil(n / 6) === 0 ? <text key={i} x={L + i * slot} y={H - 6} className="tick">{b.lo.toFixed(bw < 1 ? 1 : 0)}</text> : null))}
        <text x={W - 6} y={H - 6} className="tick" textAnchor="end">s</text>
      </svg>
    </div>
  );
}

/** A column with a 4px rounded top and a square base. */
function roundTop(x: number, y: number, w: number, h: number) {
  const r = Math.min(4, w / 2, h);
  return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`;
}

// ---- success over time: rolling rate per take, chronological ----------------------------------------------------
function SuccessOverTime({ rows, tip }: { rows: HealthRow[]; tip: TipApi }) {
  const win = Math.max(3, Math.min(10, Math.round(rows.length / 4)));
  const pts = rows.map((r, i) => {
    const w = rows.slice(Math.max(0, i - win + 1), i + 1);
    return { r, i, v: w.filter((x) => x.success).length / w.length };
  });
  const W = 420, H = 190, L = 36, B = 24, T = 10, R = 40, pw = W - L - R, ph = H - B - T;
  const x = (i: number) => L + (rows.length > 1 ? (i / (rows.length - 1)) * pw : pw / 2);
  const y = (v: number) => T + ph - v * ph;
  const [hover, setHover] = useState<number | null>(null);
  const svg = useRef<SVGSVGElement>(null);
  const move = (e: React.PointerEvent) => {
    const r = svg.current!.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    const i = Math.max(0, Math.min(rows.length - 1, Math.round(((px - L) / pw) * (rows.length - 1))));
    setHover(i);
    const p = pts[i];
    tip.show(e, [[pct(p.v), `success, last ${Math.min(win, i + 1)} takes`], [epNumber(p.r.file), p.r.success ? "this take: success" : "this take: failure"]]);
  };
  const last = pts[pts.length - 1];
  const d = pts.map((p, k) => `${k ? "L" : "M"}${x(p.i)},${y(p.v)}`).join("");
  return (
    <div className="card viz">
      <div className="card-head"><h3>Success over time</h3><span className="muted small">rolling over {win} takes</span></div>
      <svg ref={svg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Rolling success rate by take" onPointerMove={move} onPointerLeave={() => { setHover(null); tip.hide(); }}>
        {[0, 0.5, 1].map((v) => <g key={v}><line x1={L} x2={L + pw} y1={y(v)} y2={y(v)} className="grid" /><text x={L - 6} y={y(v) + 4} className="tick" textAnchor="end">{pct(v)}</text></g>)}
        <path d={`${d}L${x(last.i)},${y(0)}L${x(0)},${y(0)}Z`} className="area" />
        <path d={d} className="line" />
        {pts.map((p) => <line key={p.i} x1={x(p.i)} x2={x(p.i)} y1={T + ph + 3} y2={T + ph + (p.r.success ? 9 : 5)} className={p.r.success ? "rug ok" : "rug"} />)}
        {hover !== null && <line x1={x(hover)} x2={x(hover)} y1={T} y2={T + ph} className="crosshair" />}
        {hover !== null && <circle cx={x(hover)} cy={y(pts[hover].v)} r={4.5} className="mark fill ringed" />}
        <circle cx={x(last.i)} cy={y(last.v)} r={4.5} className="mark fill ringed" />
        <text x={x(last.i) + 8} y={y(last.v) + 4} className="end-label">{pct(last.v)}</text>
        <text x={L} y={H - 4} className="tick">first take</text>
        <text x={L + pw} y={H - 4} className="tick" textAnchor="end">latest</text>
      </svg>
    </div>
  );
}

// ---- contacts: takes with at least one event, per kind -----------------------------------------------------------
function Contacts({ rows, kinds, tip }: { rows: HealthRow[]; kinds: string[]; tip: TipApi }) {
  const data = kinds.map((k) => ({ k, takes: rows.filter((r) => (r.contact_events?.[k] ?? 0) > 0).length, events: rows.reduce((s, r) => s + (r.contact_events?.[k] ?? 0), 0) }));
  const safety = { takes: rows.filter((r) => r.safety_events > 0).length, events: rows.reduce((s, r) => s + r.safety_events, 0) };
  const all = [...data, { k: "safety filter stepped in", ...safety }];
  const max = Math.max(rows.length, 1);
  const W = 420, rowH = 34, L = 160, H = all.length * rowH + 8, pw = W - L - 50;
  return (
    <div className="card viz">
      <div className="card-head"><h3>Contacts and interventions</h3><span className="muted small">takes with at least one</span></div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Takes with contact events, per kind">
        {all.map((d, i) => {
          const yy = 4 + i * rowH, w = (d.takes / max) * pw, bh = 16;
          return (
            <g key={d.k} tabIndex={0} className="bar"
              onPointerMove={(e) => tip.show(e, [[`${d.takes} of ${rows.length}`, "takes"], [`${d.events}`, "events in total"]])}
              onFocus={(e) => tip.show(e, [[`${d.takes} of ${rows.length}`, kindLabel(d.k)]])} onPointerLeave={tip.hide} onBlur={tip.hide}>
              <rect x={0} y={yy} width={W} height={rowH} className="hit" />
              <text x={L - 10} y={yy + rowH / 2 + 4} className="cat" textAnchor="end">{kindLabel(d.k)}</text>
              <rect x={L} y={yy + (rowH - bh) / 2} width={pw} height={bh} rx={4} className="track" />
              {d.takes > 0 && <path className="mark" d={roundRight(L, yy + (rowH - bh) / 2, Math.max(w, 4), bh)} />}
              <text x={L + Math.max(w, 4) + 8} y={yy + rowH / 2 + 4} className="value">{d.takes}</text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function roundRight(x: number, y: number, w: number, h: number) {
  const r = Math.min(4, w, h / 2);
  return `M${x},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h - r}Q${x + w},${y + h} ${x + w - r},${y + h}H${x}Z`;
}

function Operators({ ops }: { ops: Health["operators"] }) {
  const rows = Object.entries(ops).sort((a, b) => b[1].takes - a[1].takes);
  if (!rows.length) return null;
  return (
    <div className="card">
      <div className="card-head"><h3>Operators</h3></div>
      <table className="data">
        <thead><tr><th>operator</th><th className="num">takes</th><th className="num">success</th><th className="num">mean take</th><th className="num">takes / hour</th></tr></thead>
        <tbody>{rows.map(([name, o]) => (
          <tr key={name}><td>{name}</td><td className="num">{o.takes}</td><td className="num">{pct(o.success_rate)}</td>
            <td className="num">{o.mean_duration_s.toFixed(1)} s</td><td className="num">{o.takes_per_hour ?? "–"}</td></tr>
        ))}</tbody>
      </table>
    </div>
  );
}

/** Every value of the charts above as a table (the charts' accessible view). */
function EpisodeTable({ rows }: { rows: HealthRow[] }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="card">
      <div className="card-head"><h3>All takes</h3><button className="btn ghost small" onClick={() => setOpen(!open)}>{open ? "Hide table" : `Show table (${rows.length})`}</button></div>
      {open && (
        <div className="table-scroll">
          <table className="data">
            <thead><tr><th>take</th><th>recorded</th><th>operator</th><th>outcome</th><th className="num">duration</th><th className="num">contacts</th><th>dataset use</th></tr></thead>
            <tbody>{rows.map((r) => (
              <tr key={r.file}><td>{epNumber(r.file)}</td><td>{r.created_at?.replace("T", " ") ?? "–"}</td><td>{r.operator ?? "–"}</td>
                <td>{r.success ? "✓ success" : "✗ failure"}</td><td className="num">{r.duration_s.toFixed(1)} s</td>
                <td className="num">{Object.values(r.contact_events ?? {}).reduce((a, b) => a + b, 0)}</td>
                <td>{r.used ? "used" : r.verdict}</td></tr>
            ))}</tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// ---- export ------------------------------------------------------------------------------------------------------
export function ExportCard({ dataset, cameras }: { dataset: string; cameras: string[] }) {
  const [info, setInfo] = useState<Exporters | null>(null);
  const [exporter, setExporter] = useState("hdf5");
  const [name, setName] = useState("");
  const [layout, setLayout] = useState<Layout | null>(null);
  const [rules, setRules] = useState<Rules | null>(null);
  const [allowMixed, setAllowMixed] = useState(false);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [job, setJob] = useState<string | null>(null);
  const [startErr, setStartErr] = useState<string | null>(null);
  const load = useCallback(() => fetch("/api/exporters", { cache: "no-store" }).then((r) => r.json()).then((x: Exporters) => {
    setInfo(x);
    setLayout((l) => l ?? x.layout);
    setRules((r) => r ?? x.rules);
    return x;
  }), []);
  useEffect(() => { void load(); }, [load]);
  // The preview follows every change of the rules (debounced): the same selection the export will make.
  useEffect(() => {
    if (!rules) return;
    const t = setTimeout(() => {
      fetch("/api/export/preview", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ dataset, rules }) })
        .then((r) => r.json()).then(setPreview).catch(() => setPreview(null));
    }, 250);
    return () => clearTimeout(t);
  }, [dataset, rules]);
  // Poll the running job.
  const running = job ? info?.jobs[job]?.state === "running" : false;
  useEffect(() => {
    if (!job || (info && info.jobs[job] && !running)) return;
    const t = setInterval(() => void load(), 800);
    return () => clearInterval(t);
  }, [job, running, info, load]);

  if (!info || !layout || !rules) return <div className="card"><p className="muted">Loading exporters…</p></div>;
  const defaultName = `${dataset || "episodes"}_${exporter}`;
  const outName = name.trim() || defaultName;
  const j = job ? info.jobs[job] : undefined;
  const fps = Object.values(preview?.episodes ?? {})[0]?.fps ?? 50;
  const mixed = (preview?.task_versions.length ?? 0) > 1;
  const reasons = Object.values(preview?.episodes ?? {}).reduce<Record<string, number>>((m, e) => {
    if (e.verdict) m[e.verdict.replace(/:.*/, "")] = (m[e.verdict.replace(/:.*/, "")] ?? 0) + 1;
    return m;
  }, {});
  const camList = layout.cameras ?? cameras;
  const toggle = <T,>(xs: T[], x: T) => (xs.includes(x) ? xs.filter((y) => y !== x) : [...xs, x]);
  const start = async () => {
    setStartErr(null);
    const r = await fetch("/api/export", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset, name: outName, exporter, rules, layout, allow_mixed: allowMixed }) });
    if (!r.ok) { setStartErr((await r.json().catch(() => ({}))).detail ?? `${r.status}`); return; }
    setJob((await r.json()).name);
    void load();
  };
  return (
    <div className="card export">
      <div className="card-head"><h2>Export</h2></div>
      <div className="sform">
        <label><span className="muted small">exporter</span>
          <select value={exporter} onChange={(e) => setExporter(e.target.value)}>{info.exporters.map((x) => <option key={x}>{x}</option>)}</select></label>
        <label><span className="muted small">name (exports/…)</span>
          <input type="text" value={name} placeholder={defaultName} onChange={(e) => setName(e.target.value)} /></label>
      </div>

      <h3>Which steps</h3>
      <div className="checks">
        <label><input type="checkbox" checked={rules.include_failures} onChange={(e) => setRules({ ...rules, include_failures: e.target.checked })} /> include failed takes</label>
        <label><input type="checkbox" checked={rules.trim_idle} onChange={(e) => setRules({ ...rules, trim_idle: e.target.checked })} /> trim idle start and end</label>
        <label><input type="checkbox" checked={rules.shared_frames === "operator"} onChange={(e) => setRules({ ...rules, shared_frames: e.target.checked ? "operator" : "all" })} /> shared control: only the operator's steps</label>
      </div>
      <div className="sform">
        <label title="A sub-task dataset: each take ends shortly after this stage (1 = the first) was reached"><span className="muted small">until stage</span>
          <input type="number" min={1} value={rules.until_stage ?? ""} placeholder="whole task"
            onChange={(e) => setRules({ ...rules, until_stage: e.target.value === "" ? null : Math.max(1, Math.floor(+e.target.value)) })} /></label>
        <label><span className="muted small">cut contact margin s</span>
          <input type="number" step={0.05} min={0} value={rules.contact_margin_s} onChange={(e) => setRules({ ...rules, contact_margin_s: Math.max(0, +e.target.value) })} /></label>
      </div>
      <table className="data rules">
        <thead><tr><th>contact</th><th className="num">max share</th><th>cut out</th></tr></thead>
        <tbody>{Object.entries(rules.max_contact_frac).map(([k, v]) => (
          <tr key={k}><td>{kindLabel(k)}</td>
            <td className="num"><input type="number" step={0.01} min={0} max={1} value={v} onChange={(e) => setRules({ ...rules, max_contact_frac: { ...rules.max_contact_frac, [k]: Math.max(0, +e.target.value) } })} /></td>
            <td><input type="checkbox" checked={rules.cut_contacts.includes(k)} onChange={() => setRules({ ...rules, cut_contacts: toggle(rules.cut_contacts, k) })} /></td></tr>
        ))}</tbody>
      </table>
      <div className={`preview ${preview ? "" : "stale"}`}>
        {preview && <>
          <div><b>{preview.segments}</b> segments · <b>{(preview.steps / fps).toFixed(1)} s</b> · {preview.steps.toLocaleString()} steps</div>
          {Object.keys(reasons).length > 0 && <div className="muted small">left out: {Object.entries(reasons).map(([k, n]) => `${k} × ${n}`).join(" · ")}</div>}
          {mixed && <div className="alert warn small">This selection mixes {preview.task_versions.length} task versions.
            <label className="inline"><input type="checkbox" checked={allowMixed} onChange={(e) => setAllowMixed(e.target.checked)} /> allow mixed</label></div>}
        </>}
      </div>

      <h3>What a policy sees</h3>
      <div className="checks">
        {cameras.map((c) => (
          <label key={c}><input type="checkbox" checked={camList.includes(c)}
            onChange={() => { const next = toggle(camList, c); setLayout({ ...layout, cameras: next.length === cameras.length ? null : next }); }} /> {c}</label>
        ))}
      </div>
      <div className="sform">
        <label><span className="muted small">image h × w</span>
          <span className="vec">{[0, 1].map((i) => <input key={i} type="number" min={16} step={16} value={layout.image_size[i]}
            onChange={(e) => { const s = [...layout.image_size] as [number, number]; s[i] = Math.max(16, Math.floor(+e.target.value)); setLayout({ ...layout, image_size: s }); }} />)}</span></label>
        <label><span className="muted small">action</span>
          <select value={layout.action} onChange={(e) => setLayout({ ...layout, action: e.target.value as Layout["action"] })}>
            <option value="joint">joint targets</option><option value="ee">TCP pose</option></select></label>
        <label title="executed: the targets actually sent (with any DART drift)"><span className="muted small">action label</span>
          <select value={layout.action_label} onChange={(e) => setLayout({ ...layout, action_label: e.target.value as Layout["action_label"] })}>
            <option value="executed">executed</option><option value="recorded">recorded</option></select></label>
        <label><span className="muted small">episodes</span>
          <select value={layout.segments} onChange={(e) => setLayout({ ...layout, segments: e.target.value as Layout["segments"] })}>
            <option value="takes">one per take</option><option value="annotations">one per labelled range</option></select></label>
      </div>
      <div className="checks">
        {STATE_CHANNELS.map((c) => <label key={c}><input type="checkbox" checked={layout.state.includes(c)} onChange={() => setLayout({ ...layout, state: toggle(layout.state, c) })} /> {c.replace(/_/g, " ")}</label>)}
        <label title="Object poses: privileged, simulation only"><input type="checkbox" checked={layout.env_state} onChange={(e) => setLayout({ ...layout, env_state: e.target.checked })} /> object poses</label>
        <label><input type="checkbox" checked={layout.randomize_visuals} onChange={(e) => setLayout({ ...layout, randomize_visuals: e.target.checked })} /> randomize visuals</label>
      </div>

      <div className="edit-row">
        <span className="grow" />
        <button className="btn small ok" disabled={running || !preview?.segments || (mixed && !allowMixed) || !layout.state.length} onClick={() => void start()}>
          {running ? "Exporting…" : `Export ${preview?.segments ?? 0} segments`}</button>
      </div>
      {startErr && <div className="alert warn small">{startErr}</div>}
      {j && (
        <div className={`job ${j.state}`}>
          {j.state === "running" && <><div className="meter"><i style={{ width: `${j.total ? (100 * j.done) / j.total : 0}%` }} /></div>
            <span className="muted small">{j.done} / {j.total || "…"} episodes · cameras are rendered from the states now</span></>}
          {j.state === "done" && j.manifest && <div className="alert info small">✓ Wrote {j.manifest.episodes} episodes ({(j.manifest.steps / j.manifest.fps).toFixed(1)} s) to <code>exports/{job}</code></div>}
          {j.state === "done" && j.manifest && Object.keys(j.manifest.constant_channels ?? {}).length > 0 && (
            <div className="alert warn small">Channels that never vary (don't feed them to a policy as they are): {Object.entries(j.manifest.constant_channels).map(([k, v]) => `${k}: ${v.join(", ")}`).join(" · ")}</div>)}
          {j.state === "failed" && <div className="alert warn small">Export failed: {j.error}</div>}
        </div>
      )}
    </div>
  );
}
