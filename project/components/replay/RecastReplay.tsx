"use client";

import { useState, type ReactNode } from "react";
import { useLoop, type Timeline } from "./useLoop";
import {
  Badge, CHARS, COLOR, ClipDef, GREY, INK, Link, MUTED, PanelFrame, R, ReplayCard, StreamStrip, Tabs, TH, Tile, TW, WRONG,
  streamX, useNarrow, type Char, type StreamGeom,
} from "./parts";

// Same beat as the protocols loop: lift, deal, match, verdict (and Re:Cast's page check), file, settle.
const LIFT = 0, DEAL = 1, MATCH = 2, VERDICT = 3, FILE = 4;
const TIMELINE: Timeline = { beats: R.steps.length, phases: [320, 620, 640, 900, 680, 520], intro: 700, hold: 3000, fade: 500 };

type PanelId = "top1" | "recast";
const PANELS: { id: PanelId; tag: string; title: string; tone: "orange" | "ink" }[] = [
  { id: "top1", tag: "P4", title: "grows by its own top-1", tone: "orange" },
  { id: "recast", tag: "Re:Cast", title: "one entry each, page-checked", tone: "ink" },
];

const PW = 400;
const PH = 470;
const ROW_Y = [68, 138, 208];
const ROW_X = 96;
const ROW_S = 0.62;                      // 37 x 50
const DECK_DX = 13;                      // offset between the crops stacked in a cast-sheet entry
const SLOT = { x: (PW - 0.9 * TW) / 2, y: 282, s: 0.9 };
const TEXT_Y = 382;
const SCORE_Y = PH - 22;

type Layout = { w: number; h: number; stream: StreamGeom; py: number; px: number[]; shown: PanelId[] };
const WIDE: Layout = {
  w: 1000, h: 150 + PH + 6, py: 150, px: [80, 520], shown: ["top1", "recast"],
  stream: { x0: 95, dx: 108, y: 30, s: 0.9, pages: true, label: true },
};
const narrow = (p: PanelId): Layout => ({
  w: 420, h: 80 + PH + 6, py: 80, px: [10], shown: [p],
  stream: { x0: 10, dx: 52, y: 12, s: 0.62, pages: false, label: false },
});

const steps = R.steps;
const name = (c: Char) => R.names[c];
const truth = (k: number) => steps[k].truth;
const seedsOf = (c: Char) => R.gallery.seeds.filter((s) => R.crops[s].char === c);

function rowX(j: number, n: number) {
  const room = PW - 14 - ROW_S * TW - ROW_X;
  const step = n > 1 ? Math.min(46, room / (n - 1)) : 0;
  return ROW_X + j * step;
}

export function RecastReplay() {
  const loop = useLoop(TIMELINE);
  const { clock, at, epoch, fading } = loop;
  const isNarrow = useNarrow(loop.ref);
  const [tab, setTab] = useState<PanelId>("recast");
  const L = isNarrow ? narrow(tab) : WIDE;
  const clip = "rp-clip-recast";

  const inFlight = (k: number, px: number) =>
    at(k) === LIFT
      ? { x: streamX(L.stream, k) - px, y: L.stream.y - L.py, s: L.stream.s }
      : { x: SLOT.x, y: SLOT.y, s: SLOT.s };

  // P4: seeds, then every query filed under its top-1 (all of them); as of now, and as of query k.
  const top1Row = (c: Char) => [...seedsOf(c), ...R.stream.filter((_, j) => at(j) >= FILE && steps[j].p4.pred === c)];
  const top1RowBefore = (c: Char, k: number) => [...seedsOf(c), ...R.stream.filter((_, j) => j < k && steps[j].p4.pred === c)];
  // Re:Cast: the seed, then the queries committed to that character; as of now, and as of query k.
  const deck = (c: Char) => [...seedsOf(c), ...R.stream.filter((_, j) => at(j) >= FILE && steps[j].recast.commit === c)];
  const deckBefore = (c: Char, k: number) => [...seedsOf(c), ...R.stream.filter((_, j) => j < k && steps[j].recast.commit === c)];

  function panel(p: PanelId, px: number): ReactNode {
    const meta = PANELS.find((x) => x.id === p)!;
    const cur = clock.beat;
    const a = cur >= 0 ? at(cur) : -1;
    const els: ReactNode[] = [];

    CHARS.forEach((c, r) => {
      els.push(
        <g key={`lbl-${c}`}>
          <circle cx={18} cy={ROW_Y[r] + 25} r={5.5} fill={COLOR[c]} />
          <text x={29} y={ROW_Y[r] + 29.5} fontSize={13} fontWeight={600} fill={INK}>{name(c)}</text>
        </g>,
      );
      const seeds = seedsOf(c);
      if (p === "top1") {
        const n = top1Row(c).length;
        seeds.forEach((id, j) => els.push(
          <Tile key={`g-${id}`} crop={id} clip={clip} x={rowX(j, n)} y={ROW_Y[r]} s={ROW_S} frame={COLOR[c]} />,
        ));
      } else {
        const d = deck(c);
        seeds.forEach((id, j) => els.push(
          <Tile key={`g-${id}`} crop={id} clip={clip} x={ROW_X + j * DECK_DX} y={ROW_Y[r]} s={ROW_S} frame={COLOR[c]} />,
        ));
        els.push(
          <text key={`mean-${c}`} x={ROW_X + (d.length - 1) * DECK_DX + TW * ROW_S + 12} y={ROW_Y[r] + 29.5} fontSize={12}
            fill={MUTED} className="rp-fade">
            {d.length === 1 ? "the seed alone" : `mean of ${d.length}`}
          </text>,
        );
      }
    });

    R.stream.forEach((q, k) => {
      const ak = at(k);
      if (ak < LIFT) return;
      const s = steps[k];
      const filed = ak >= FILE;
      const frame = ak >= VERDICT ? COLOR[truth(k)] : GREY;
      if (p === "top1") {
        if (!filed) {
          els.push(<Tile key={`q-${q}`} crop={q} clip={clip} {...inFlight(k, px)} frame={frame} />);
          return;
        }
        const row = top1Row(s.p4.pred);
        els.push(<Tile key={`q-${q}`} crop={q} clip={clip} x={rowX(row.indexOf(q), row.length)}
          y={ROW_Y[CHARS.indexOf(s.p4.pred)]} s={ROW_S} frame={COLOR[truth(k)]} dashed />);
        return;
      }
      const c = s.recast.commit;
      if (!filed || !c) {
        if (ak === Infinity) return;                       // held: it leaves the stage
        els.push(<Tile key={`q-${q}`} crop={q} clip={clip} {...inFlight(k, px)} frame={frame} opacity={filed ? 0 : 1} />);
        return;
      }
      const d = deck(c);
      els.push(<Tile key={`q-${q}`} crop={q} clip={clip} x={ROW_X + d.indexOf(q) * DECK_DX} y={ROW_Y[CHARS.indexOf(c)]}
        s={ROW_S} frame={COLOR[truth(k)]} />);
    });

    let text: ReactNode = null;
    if (cur >= 0 && a >= DEAL && a !== Infinity) {
      const s = steps[cur];
      const from = { x: SLOT.x + (TW * SLOT.s) / 2, y: SLOT.y - 2 };
      const page = R.crops[s.q].page;
      if (p === "top1") {
        const m = s.p4;
        const row = top1RowBefore(m.pred, cur);
        const t = { x: rowX(row.indexOf(m.match), row.length), y: ROW_Y[CHARS.indexOf(m.pred)] };
        const ok = m.pred === s.truth;
        els.push(
          <rect key={`halo-${cur}`} x={t.x - 3.5} y={t.y - 3.5} width={TW * ROW_S + 7} height={TH * ROW_S + 7} rx={8} fill="none"
            stroke={ok ? INK : WRONG} strokeWidth={2.6} className="rp-fade" style={{ opacity: a >= MATCH && a < FILE ? 1 : 0 }} />,
          <Link key={`link-${cur}`} x1={from.x} y1={from.y} x2={t.x + (TW * ROW_S) / 2} y2={t.y + TH * ROW_S + 2}
            drawn={a >= MATCH} visible={a < FILE} color={a >= VERDICT && !ok ? WRONG : INK} />,
          <Badge key={`badge-${cur}`} x={SLOT.x + TW * SLOT.s - 2} y={SLOT.y + 2} ok={ok} show={a >= VERDICT && a < FILE} />,
        );
        const msg = a >= FILE ? (ok ? `filed under ${name(m.pred)}` : `filed under ${name(m.pred)}, wrongly`)
          : a >= VERDICT ? (ok ? `matched ${name(m.pred)}` : `matched ${name(m.pred)}, not ${name(s.truth)}`)
          : `top-1 similarity ${m.sim.toFixed(2)}`;
        if (a >= MATCH) {
          text = <text x={PW / 2} y={TEXT_Y} textAnchor="middle" fontSize={13} fontWeight={a >= VERDICT ? 700 : 500}
            fill={a >= VERDICT && !ok ? WRONG : INK}>{msg}</text>;
        }
      } else {
        const rc = s.recast;
        const d = deckBefore(rc.pred, cur);
        const t = { x: ROW_X + (d.length - 1) * DECK_DX, y: ROW_Y[CHARS.indexOf(rc.pred)] };
        const ok = rc.pred === s.truth;
        els.push(
          <rect key={`halo-${cur}`} x={ROW_X - 3.5} y={t.y - 3.5} width={(d.length - 1) * DECK_DX + TW * ROW_S + 7} height={TH * ROW_S + 7}
            rx={8} fill="none" stroke={ok ? INK : WRONG} strokeWidth={2.6} className="rp-fade"
            style={{ opacity: a >= MATCH && a < VERDICT ? 1 : 0 }} />,
          <Link key={`link-${cur}`} x1={from.x} y1={from.y} x2={t.x + (TW * ROW_S) / 2} y2={t.y + TH * ROW_S + 2}
            drawn={a >= MATCH} visible={a < VERDICT} color={INK} />,
          <Badge key={`badge-${cur}`} x={SLOT.x + TW * SLOT.s - 2} y={SLOT.y + 2} ok={ok} show={a >= VERDICT && a < FILE} />,
        );
        if (rc.via) {
          const c = rc.commit as Char;
          const v = deckBefore(c, cur).indexOf(rc.via);
          const vx = ROW_X + v * DECK_DX;
          const vy = ROW_Y[CHARS.indexOf(c)];
          els.push(
            <rect key={`via-${cur}`} x={vx - 3.5} y={vy - 3.5} width={TW * ROW_S + 7} height={TH * ROW_S + 7} rx={8} fill="none"
              stroke={COLOR[c]} strokeWidth={2.6} strokeDasharray="4 3" className="rp-fade"
              style={{ opacity: a >= VERDICT && a < FILE ? 1 : 0 }} />,
            <Link key={`page-${cur}`} x1={from.x} y1={from.y} x2={vx + (TW * ROW_S) / 2} y2={vy + TH * ROW_S + 2}
              drawn={a >= VERDICT} visible={a < FILE} color={COLOR[c]} dashed />,
          );
        }
        const first = a >= VERDICT ? (ok ? `matched ${name(rc.pred)}` : `matched ${name(rc.pred)}, not ${name(s.truth)}`)
          : `nearest entry ${rc.sims[rc.pred].toFixed(2)}`;
        const second = a >= FILE
          ? (rc.commit ? `added to ${name(rc.commit)}` : "held back, not added")
          : rc.via
            ? `page ${page} groups it with ${name(rc.commit as Char)}${rc.via.startsWith("q") ? "" : "’s seed"}`
            : `nothing on page ${page} vouches for it`;
        if (a >= MATCH) {
          text = (
            <>
              <text x={PW / 2} y={TEXT_Y} textAnchor="middle" fontSize={13} fontWeight={a >= VERDICT ? 700 : 500}
                fill={a >= VERDICT && !ok ? WRONG : INK}>{first}</text>
              {a >= VERDICT && (
                <text x={PW / 2} y={TEXT_Y + 21} textAnchor="middle" fontSize={12.5} fontWeight={600}
                  fill={rc.commit ? COLOR[rc.commit] : MUTED}>{second}</text>
              )}
            </>
          );
        }
      }
    }

    const scored = R.stream.map((_, j) => j).filter((j) => at(j) >= VERDICT);
    const added = R.stream.map((_, j) => j).filter((j) => at(j) >= FILE && (p === "top1" || steps[j].recast.commit));
    const pred = (j: number) => (p === "top1" ? steps[j].p4.pred : steps[j].recast.pred);
    const filedAs = (j: number) => (p === "top1" ? steps[j].p4.pred : steps[j].recast.commit);
    const right = scored.filter((j) => pred(j) === truth(j)).length;
    const wrong = added.filter((j) => filedAs(j) !== truth(j)).length;
    let score = scored.length ? `${right} of ${scored.length} right` : "–";
    if (added.length) score += ` · ${added.length} added, ${wrong} wrong`;

    return (
      <>
        <PanelFrame w={PW} h={PH} tag={meta.tag} title={meta.title} tone={meta.tone} />
        {els}
        {text}
        <line x1={16} x2={PW - 16} y1={SCORE_Y - 22} y2={SCORE_Y - 22} stroke="#e6ded0" strokeWidth={1.5} strokeDasharray="3 4" />
        <text x={PW / 2} y={SCORE_Y} textAnchor="middle" fontSize={14} fontWeight={800} fill={wrong ? WRONG : INK}
          className="rp-display">{score}</text>
      </>
    );
  }

  const cur = clock.beat;
  const status = cur < 0 ? "ready" : `query ${cur + 1} of ${steps.length} · page ${R.crops[R.stream[cur]].page}`;

  return (
    <ReplayCard
      cardRef={loop.ref}
      tabs={isNarrow ? (
        <Tabs items={PANELS.map((p) => ({ id: p.id, label: p.id === "top1" ? "P4 · grows by top-1" : "Re:Cast" }))} value={tab} onChange={setTab} />
      ) : undefined}
      stage={(
        <svg viewBox={`0 0 ${L.w} ${L.h}`} className="rp-svg block h-auto w-full" role="img"
          aria-label="The same eight crops against two ways of growing the gallery: filing every query under its top-1 match ends at five of eight with three wrong additions, Re:Cast adds only the two crops their page ties to a seed and ends at seven of eight.">
          <ClipDef id={clip} />
          <g key={`${epoch}-${isNarrow ? tab : "wide"}`} className="rp-stage" style={{ opacity: fading ? 0 : 1 }}>
            <StreamStrip g={L.stream} clip={clip} clock={clock} at={at} />
            {L.shown.map((p, i) => {
              const px = isNarrow ? L.px[0] : L.px[i];
              return <g key={p} transform={`translate(${px} ${L.py})`}>{panel(p, px)}</g>;
            })}
          </g>
        </svg>
      )}
      playing={loop.playing}
      onToggle={loop.toggle}
      onRestart={loop.restart}
      clock={clock}
      beats={steps.length}
      status={status}
      lead="Growth by top-1 against Re:Cast, on the same stream."
      caption={(
        <>
          Both start from one seed per character. P4 files every query under its top-1 match, so the page-29 mistake
          stays in the gallery and is matched again. Re:Cast keeps one averaged entry per character and adds a crop only
          when MagiV2&rsquo;s page grouping ties it to a crop already committed, here the seeds on pages 17 and 50;
          everywhere else it holds.
        </>
      )}
    />
  );
}
