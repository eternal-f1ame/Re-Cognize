"use client";

import { useEffect, useState, type ReactNode, type RefObject } from "react";
import data from "../../data/replay.json";
import { Icon } from "../Icon";
import type { Clock } from "./useLoop";

// ---------------------------------------------------------------- data (written by scripts/make_replay.py)
export type Char = "T" | "M" | "A";
export type Match = { match: string; pred: Char; sim: number };
export type Step = {
  q: string;
  truth: Char;
  p1: Match;
  p2: Match;
  p3: { cluster: number; best: number | null; open: boolean };
  p4: Match;
  recast: { pred: Char; sims: Record<Char, number>; commit: Char | null; via: string | null };
};
export type Replay = {
  encoder: string;
  series: string;
  chapter: number;
  tau: number;
  names: Record<Char, string>;
  crops: Record<string, { src: string; char: Char; page: number }>;
  stream: string[];
  gallery: { p1: string[]; seeds: string[] };
  steps: Step[];
};
export const R = data as Replay;
export const CHARS: Char[] = ["T", "M", "A"];

// Frames carry the character, as in the paper's protocols figure; the stream is grey because its labels are unknown.
export const COLOR: Record<Char, string> = { T: "#36679b", M: "#c0662b", A: "#4f8a5b" };
export const GREY = "#9a9083";
export const INK = "#17140f";
export const MUTED = "#6d6152";
export const WRONG = "#b3261e";
export const CARD = "#fffdf8";
export const TONE = {
  blue: { fill: "#dfe8f2", stroke: "#36679b" },
  orange: { fill: "#f6e4d5", stroke: "#c0662b" },
  ink: { fill: "#17140f", stroke: "#17140f" },
};

// Every crop is drawn at TW x TH and scaled, so one clip path serves all of them.
export const TW = 60;
export const TH = 80;

export function ClipDef({ id }: { id: string }) {
  return (
    <defs>
      <clipPath id={id}>
        <rect width={TW} height={TH} rx={7} />
      </clipPath>
    </defs>
  );
}

export function Tile({ crop, clip, x, y, s, frame, dashed = false, opacity = 1, width = 2.2 }: {
  crop: string;
  clip: string;
  x: number;
  y: number;
  s: number;
  frame: string;
  dashed?: boolean;
  opacity?: number;
  width?: number;
}) {
  return (
    <g className="rp-move" style={{ transform: `translate(${x}px, ${y}px) scale(${s})`, opacity }}>
      <rect width={TW} height={TH} rx={7} fill="#fff" />
      <image href={R.crops[crop].src} width={TW} height={TH} preserveAspectRatio="xMidYMid slice" clipPath={`url(#${clip})`} />
      <rect width={TW} height={TH} rx={7} fill="none" stroke={frame} strokeWidth={width}
        strokeDasharray={dashed ? "5 3.5" : undefined} vectorEffect="non-scaling-stroke" />
    </g>
  );
}

// A right/wrong mark, centred on (x, y).
export function Badge({ x, y, ok, show }: { x: number; y: number; ok: boolean; show: boolean }) {
  return (
    <g className="rp-fade" style={{ opacity: show ? 1 : 0 }} transform={`translate(${x} ${y})`}>
      <circle r={10} fill={ok ? "#fff" : WRONG} stroke={ok ? INK : WRONG} strokeWidth={1.8} />
      {ok ? (
        <path d="M-4.5 0.2 L-1.4 3.4 L4.6 -3.4" fill="none" stroke={INK} strokeWidth={2.2} strokeLinecap="round" strokeLinejoin="round" />
      ) : (
        <path d="M-3.6 -3.6 L3.6 3.6 M3.6 -3.6 L-3.6 3.6" fill="none" stroke="#fff" strokeWidth={2.2} strokeLinecap="round" />
      )}
    </g>
  );
}

// A straight connector that draws itself when `drawn` turns true.
export function Link({ x1, y1, x2, y2, drawn, visible = true, color = INK, dashed = false }: {
  x1: number; y1: number; x2: number; y2: number; drawn: boolean; visible?: boolean; color?: string; dashed?: boolean;
}) {
  if (dashed) {
    return (
      <path d={`M${x1} ${y1} L${x2} ${y2}`} className="rp-fade" style={{ opacity: drawn && visible ? 1 : 0 }}
        fill="none" stroke={color} strokeWidth={1.8} strokeDasharray="3 4" strokeLinecap="round" />
    );
  }
  return (
    <path d={`M${x1} ${y1} L${x2} ${y2}`} pathLength={1} className="rp-draw"
      style={{ strokeDashoffset: drawn ? 0 : 1, opacity: visible ? 1 : 0 }}
      fill="none" stroke={color} strokeWidth={1.8} strokeDasharray="1 1" strokeLinecap="round" />
  );
}

export type StreamGeom = { x0: number; dx: number; y: number; s: number; pages: boolean; label: boolean };

export const streamX = (g: StreamGeom, k: number) => g.x0 + k * g.dx;

// The query stream in reading order, identical under every panel, with the playhead on the current query.
export function StreamStrip({ g, clip, clock, at }: {
  g: StreamGeom;
  clip: string;
  clock: Clock;
  at: (k: number) => number;
}) {
  const w = TW * g.s;
  const h = TH * g.s;
  const cur = Math.max(0, clock.beat);
  return (
    <g>
      {g.label && (
        <text x={g.x0} y={g.y - 12} className="rp-mono" fontSize={10.5} letterSpacing="0.12em" fill={MUTED}>
          {`ONE QUERY STREAM · ${R.series.toUpperCase()} CH. ${R.chapter} · READING ORDER →`}
        </text>
      )}
      {R.stream.map((q, k) => {
        const done = at(k) === Infinity;
        return (
          <g key={q}>
            <Tile crop={q} clip={clip} x={streamX(g, k)} y={g.y} s={g.s} frame={GREY} opacity={done ? 0.38 : 1} width={1.6} />
            {g.pages && (
              <text x={streamX(g, k) + w / 2} y={g.y + h + 15} textAnchor="middle" fontSize={11}
                fontWeight={k === clock.beat ? 700 : 400} fill={k === clock.beat ? INK : MUTED}>
                p. {R.crops[q].page}
              </text>
            )}
          </g>
        );
      })}
      <g className="rp-move" style={{ transform: `translate(${streamX(g, cur) - 4}px, ${g.y - 4}px)`, opacity: clock.beat < 0 ? 0 : 1 }}>
        <rect width={w + 8} height={h + 8} rx={8} fill="none" stroke={TONE.orange.stroke} strokeWidth={2.6} />
      </g>
    </g>
  );
}

// A panel's header chip and body, in panel coordinates.
export function PanelFrame({ w, h, tag, title, tone }: { w: number; h: number; tag: string; title: string; tone: keyof typeof TONE }) {
  const t = TONE[tone];
  const dark = tone === "ink";
  return (
    <g>
      <rect x={3} y={45} width={w} height={h - 42} rx={11} fill={INK} />
      <rect x={0} y={42} width={w} height={h - 42} rx={11} fill={CARD} stroke={INK} strokeWidth={2} />
      <rect x={0} y={0} width={w} height={34} rx={9} fill={t.fill} stroke={t.stroke} strokeWidth={2} />
      <text x={12} y={22.5} fontSize={14} fontWeight={800} fill={dark ? "#f6e4d5" : t.stroke} className="rp-display">{tag}</text>
      <text x={12 + tag.length * 9.6 + 8} y={22.5} fontSize={13.5} fontWeight={600} fill={dark ? "#faf7f0" : INK}>{title}</text>
    </g>
  );
}

export function useNarrow(ref: RefObject<HTMLElement | null>, below = 620) {
  const [narrow, setNarrow] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setNarrow(e.contentRect.width < below));
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref, below]);
  return narrow;
}

// The card around a loop: optional tabs (narrow screens show one panel), the stage, the controls and the caption.
export function ReplayCard({ cardRef, tabs, stage, playing, onToggle, onRestart, clock, beats, status, extra, lead, caption }: {
  cardRef: RefObject<HTMLDivElement | null>;
  tabs?: ReactNode;
  stage: ReactNode;
  playing: boolean;
  onToggle: () => void;
  onRestart: () => void;
  clock: Clock;
  beats: number;
  status: string;
  extra?: ReactNode;
  lead: string;
  caption: ReactNode;
}) {
  return (
    <figure className="mx-auto w-full max-w-[1000px]">
      <div ref={cardRef} className="figure-card !p-3 sm:!p-4">
        {tabs}
        {stage}
        <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-2 border-t-2 border-dashed border-[var(--paper-2)] pt-3">
          <button type="button" onClick={onToggle} className="btn !px-3 !py-1.5 !text-sm" aria-label={playing ? "Pause the animation" : "Play the animation"}>
            <Icon name={playing ? "pause" : "play"} size={15} /> {playing ? "Pause" : "Play"}
          </button>
          <button type="button" onClick={onRestart} className="btn !px-3 !py-1.5 !text-sm" aria-label="Restart the animation">
            <Icon name="restart" size={15} /> Restart
          </button>
          <div className="flex items-center gap-1.5" aria-hidden="true">
            {Array.from({ length: beats }, (_, k) => (
              <span key={k} className={`h-2 w-2 rounded-full border border-[var(--line)] transition-colors ${
                k < clock.beat ? "bg-[var(--ink)]" : k === clock.beat ? "bg-[var(--orange)]" : "bg-transparent"}`} />
            ))}
          </div>
          <span className="font-mono text-xs text-[var(--muted)]" aria-live="off">{status}</span>
          {extra && <span className="ml-auto">{extra}</span>}
        </div>
      </div>
      <figcaption className="figure-caption mt-3 px-1">
        <strong>{lead} </strong>
        {caption}
      </figcaption>
    </figure>
  );
}

// Segmented control for the one-panel view on narrow screens.
export function Tabs<T extends string>({ items, value, onChange }: {
  items: { id: T; label: ReactNode }[];
  value: T;
  onChange: (v: T) => void;
}) {
  return (
    <div role="tablist" className="mb-2 flex gap-1.5">
      {items.map((it) => (
        <button key={it.id} type="button" role="tab" aria-selected={value === it.id} onClick={() => onChange(it.id)}
          className={`flex-1 rounded-md border-2 border-[var(--line)] px-2 py-1 text-xs font-bold transition-colors ${
            value === it.id ? "bg-[var(--ink)] text-[var(--paper)]" : "bg-[var(--card)] text-[var(--ink)]"}`}>
          {it.label}
        </button>
      ))}
    </div>
  );
}
