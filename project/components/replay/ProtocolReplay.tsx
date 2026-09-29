"use client";

import { useCallback, useRef, useState, type ReactNode } from "react";
import { ImagePopup } from "../ImagePopup";
import { Icon } from "../Icon";
import { PROTOCOLS_FIGURE } from "../../content";
import { useLoop, type Timeline } from "./useLoop";
import {
  Badge, CHARS, COLOR, ClipDef, GREY, INK, Link, MUTED, PanelFrame, R, ReplayCard, StreamStrip, Tabs, TH, Tile, TONE, TW, WRONG,
  streamX, useNarrow, type Char, type StreamGeom,
} from "./parts";

// One beat per query: it leaves the stream, reaches every panel, is matched, judged, filed, and the score updates.
const LIFT = 0, DEAL = 1, MATCH = 2, VERDICT = 3, FILE = 4;
const TIMELINE: Timeline = { beats: R.steps.length, phases: [320, 620, 640, 600, 660, 520], intro: 700, hold: 3000, fade: 500 };

type PanelId = "p1" | "p2" | "p3" | "p4";
const PANELS: { id: PanelId; tag: string; title: string; tone: "blue" | "orange" }[] = [
  { id: "p1", tag: "P1", title: "full gallery", tone: "blue" },
  { id: "p2", tag: "P2", title: "one seed each", tone: "blue" },
  { id: "p3", tag: "P3", title: "starts empty", tone: "orange" },
  { id: "p4", tag: "P4", title: "grows by top-1", tone: "orange" },
];

// Panel coordinates (a panel is pw x PH; pw depends on the layout). The query waits at the left, beside
// the gallery it is matched against, so a panel is short and the four sit side by side on one screen.
const PH = 250;
const ROW_Y = [44, 92, 140];             // top of each character's row of crops
const DOT_X = 74;                        // the row's colour, named in the legend under the stage
const ROW_X = 86;
const ROW_S = 0.52;                      // gallery crops: 31 x 42
const SLOT = { x: 12, y: 78, s: 0.8 };   // the query as it is decided: 48 x 64
const TEXT_Y = 207;
const SCORE_Y = 239;
const CL = { h: 66, s: 0.4 };            // P3 cluster boxes (two per row) and their members (24 x 32)

// Wide: four panels side by side on a stage about 3.5 times as wide as it is tall, so the loop can take
// the page's full width and still fit a laptop screen's height. Narrow: one panel, chosen by tabs.
type Layout = { w: number; h: number; pw: number; stream: StreamGeom; py: number; px: number[]; shown: PanelId[] };
const WIDE: Layout = {
  w: 1214, h: 96 + PH + 4, pw: 296, py: 96, px: [0, 306, 612, 918], shown: ["p1", "p2", "p3", "p4"],
  stream: { x0: 63.4, dx: 150, y: 18, s: 0.62, pages: true, label: true },
};
const narrow = (p: PanelId): Layout => ({
  w: 264, h: 46 + PH + 4, pw: 244, py: 46, px: [10], shown: [p],
  stream: { x0: 11.5, dx: 31, y: 6, s: 0.4, pages: false, label: false },
});
// The card shares the screen with the section header, its own controls and the caption.
const FIT = { reserve: "17rem" };

const steps = R.steps;
const name = (c: Char) => R.names[c];
const truth = (k: number) => steps[k].truth;

function rowX(j: number, n: number, pw: number) {
  const room = pw - 10 - ROW_S * TW - ROW_X;
  const step = n > 1 ? Math.min(37, room / (n - 1)) : 0;
  return ROW_X + j * step;
}

const clusterW = (pw: number) => (pw - 70 - 12 - 8) / 2;

function clusterBox(c: number, pw: number) {
  return { x: 70 + (c % 2) * (clusterW(pw) + 8), y: 42 + Math.floor(c / 2) * 72 };
}

function clusterMemberX(c: number, m: number, n: number, pw: number) {
  const room = clusterW(pw) - 12 - CL.s * TW;
  const step = n > 1 ? Math.min(20, room / (n - 1)) : 0;
  return clusterBox(c, pw).x + 6 + m * step;
}

const center = (x: number, y: number, s: number) => ({ x: x + (TW * s) / 2, y: y + (TH * s) / 2 });

export function ProtocolReplay() {
  const loop = useLoop(TIMELINE);
  const { clock, at, epoch, fading } = loop;
  const isNarrow = useNarrow(loop.ref);
  const [tab, setTab] = useState<PanelId>("p4");
  const [figure, setFigure] = useState(false);
  const closeFigure = useCallback(() => setFigure(false), []);
  const L = isNarrow ? narrow(tab) : WIDE;
  const PW = L.pw;
  const clip = "rp-clip-protocols";

  // Where a query's copy sits in a panel while it is the current query, before it is filed.
  const inFlight = (k: number, px: number) => {
    const a = at(k);
    if (a === LIFT) return { x: streamX(L.stream, k) - px, y: L.stream.y - L.py, s: L.stream.s };
    return { x: SLOT.x, y: SLOT.y, s: SLOT.s };
  };

  // P4's rows at the current moment: the seeds, then every query already filed there.
  const p4Row = (c: Char) => [
    ...R.gallery.seeds.filter((s) => R.crops[s].char === c),
    ...R.stream.filter((_, j) => at(j) >= FILE && steps[j].p4.pred === c),
  ];
  // The same rows as they stood when query k was scored (its own filing excluded).
  const p4RowBefore = (c: Char, k: number) => [
    ...R.gallery.seeds.filter((s) => R.crops[s].char === c),
    ...R.stream.filter((_, j) => j < k && steps[j].p4.pred === c),
  ];

  const staticRows = (p: PanelId) =>
    CHARS.map((c) => (p === "p1" ? R.gallery.p1 : R.gallery.seeds).filter((s) => R.crops[s].char === c));

  function tilePos(p: PanelId, id: string, k: number) {
    if (p === "p4") {
      const c = id.startsWith("q") ? steps[R.stream.indexOf(id)].p4.pred : R.crops[id].char;
      const row = p4RowBefore(c, k);
      const r = CHARS.indexOf(c);
      return { x: rowX(row.indexOf(id), row.length, PW), y: ROW_Y[r], s: ROW_S };
    }
    const rows = staticRows(p);
    const r = CHARS.indexOf(R.crops[id].char);
    return { x: rowX(rows[r].indexOf(id), rows[r].length, PW), y: ROW_Y[r], s: ROW_S };
  }

  function panel(p: PanelId, px: number): ReactNode {
    const meta = PANELS.find((x) => x.id === p)!;
    const cur = clock.beat;
    const a = cur >= 0 ? at(cur) : -1;
    const els: ReactNode[] = [];

    // character rows (P1, P2, P4)
    if (p !== "p3") {
      CHARS.forEach((c, r) => {
        els.push(<circle key={`lbl-${c}`} cx={DOT_X} cy={ROW_Y[r] + 21} r={4.5} fill={COLOR[c]} />);
      });
      const rows = p === "p4" ? CHARS.map((c) => R.gallery.seeds.filter((s) => R.crops[s].char === c)) : staticRows(p);
      rows.forEach((row, r) => {
        const n = p === "p4" ? p4Row(CHARS[r]).length : row.length;
        row.forEach((id, j) => {
          els.push(<Tile key={`g-${id}`} crop={id} clip={clip} x={rowX(j, n, PW)} y={ROW_Y[r]} s={ROW_S} frame={COLOR[R.crops[id].char]} />);
        });
      });
    } else {
      const n = Math.max(0, ...steps.map((s) => s.p3.cluster)) + 1;
      for (let c = 0; c < n; c++) {
        const opener = steps.findIndex((s) => s.p3.cluster === c);
        const box = clusterBox(c, PW);
        els.push(
          <g key={`cl-${c}`} className="rp-fade" style={{ opacity: at(opener) >= VERDICT ? 1 : 0 }}>
            <rect x={box.x} y={box.y} width={clusterW(PW)} height={CL.h} rx={8} fill="#fff" stroke={TONE.orange.stroke}
              strokeWidth={1.5} strokeDasharray="4 3" />
            <text x={box.x + 6} y={box.y + 13} fontSize={9.5} fontWeight={600} fill={MUTED}>cluster {c + 1}</text>
          </g>,
        );
      }
    }

    // every query that has reached this panel
    R.stream.forEach((q, k) => {
      const ak = at(k);
      if (ak < LIFT) return;
      const s = steps[k];
      const filed = ak >= FILE;
      const reveal = ak >= VERDICT;
      const frame = reveal ? COLOR[truth(k)] : GREY;
      if (p === "p1" || p === "p2") {
        if (ak === Infinity) return;
        const pos = inFlight(k, px);
        els.push(<Tile key={`q-${q}`} crop={q} clip={clip} {...pos} frame={frame} opacity={filed ? 0 : 1} />);
        return;
      }
      if (p === "p4") {
        if (!filed) {
          els.push(<Tile key={`q-${q}`} crop={q} clip={clip} {...inFlight(k, px)} frame={frame} />);
          return;
        }
        const c = s.p4.pred;
        const row = p4Row(c);
        els.push(<Tile key={`q-${q}`} crop={q} clip={clip} x={rowX(row.indexOf(q), row.length, PW)} y={ROW_Y[CHARS.indexOf(c)]}
          s={ROW_S} frame={COLOR[truth(k)]} dashed />);
        return;
      }
      // P3
      if (!filed) {
        els.push(<Tile key={`q-${q}`} crop={q} clip={clip} {...inFlight(k, px)} frame={frame} />);
        return;
      }
      const c = s.p3.cluster;
      const members = R.stream.filter((_, j) => at(j) >= FILE && steps[j].p3.cluster === c);
      const box = clusterBox(c, PW);
      els.push(<Tile key={`q-${q}`} crop={q} clip={clip} x={clusterMemberX(c, members.indexOf(q), members.length, PW)}
        y={box.y + 22} s={CL.s} frame={COLOR[truth(k)]} />);
    });

    // the current query's match, verdict and decision
    let line: ReactNode = null;
    if (cur >= 0 && a >= DEAL && a !== Infinity) {
      const s = steps[cur];
      const from = { x: SLOT.x + TW * SLOT.s + 2, y: SLOT.y + (TH * SLOT.s) / 2 };
      if (p !== "p3") {
        const m = s[p];
        const t = tilePos(p, m.match, cur);
        const c = center(t.x, t.y, t.s);
        els.push(
          <rect key={`halo-${cur}`} x={t.x - 3.5} y={t.y - 3.5} width={TW * t.s + 7} height={TH * t.s + 7} rx={8} fill="none"
            stroke={m.pred === s.truth ? INK : WRONG} strokeWidth={2.6} className="rp-fade"
            style={{ opacity: a >= MATCH && a < FILE ? 1 : 0 }} />,
          <Link key={`link-${cur}`} x1={from.x} y1={from.y} x2={t.x - 2} y2={c.y}
            drawn={a >= MATCH} visible={a < FILE} color={a >= VERDICT && m.pred !== s.truth ? WRONG : INK} />,
          <Badge key={`badge-${cur}`} x={SLOT.x + TW * SLOT.s - 3} y={SLOT.y + 3} ok={m.pred === s.truth} show={a >= VERDICT && a < FILE} />,
        );
        const ok = m.pred === s.truth;
        let msg = a >= VERDICT ? (ok ? `matched ${name(m.pred)}` : `matched ${name(m.pred)}, not ${name(s.truth)}`) : `top-1 similarity ${m.sim.toFixed(2)}`;
        if (p === "p4" && a >= FILE) msg = ok ? `filed under ${name(m.pred)}` : `filed under ${name(m.pred)}, wrongly`;
        line = a >= MATCH ? <text x={PW / 2} y={TEXT_Y} textAnchor="middle" fontSize={11.5} fontWeight={a >= VERDICT ? 700 : 500}
          fill={a >= VERDICT && !ok ? WRONG : INK}>{msg}</text> : null;
      } else {
        const d = s.p3;
        if (!d.open && a >= MATCH && a < FILE) {
          const box = clusterBox(d.cluster, PW);
          els.push(<Link key={`link-${cur}`} x1={from.x} y1={from.y} x2={box.x - 2} y2={box.y + CL.h / 2} drawn={a >= MATCH} visible={a < FILE} />);
        }
        const cmp = d.best === null ? "nothing to compare with yet" : `closest cluster ${d.best.toFixed(2)} ${d.best >= R.tau ? "≥" : "<"} ${R.tau.toFixed(2)}`;
        const msg = a >= VERDICT ? (d.open ? "opens a new cluster" : `joins cluster ${d.cluster + 1}`) : cmp;
        line = a >= MATCH ? <text x={PW / 2} y={TEXT_Y} textAnchor="middle" fontSize={11.5} fontWeight={a >= VERDICT ? 700 : 500}
          fill={INK}>{msg}</text> : null;
      }
    }

    // running score
    const scored = R.stream.map((_, j) => j).filter((j) => at(j) >= VERDICT);
    let score: string;
    if (p === "p3") {
      const open = new Set(scored.map((j) => steps[j].p3.cluster)).size;
      const chars = new Set(scored.map((j) => truth(j))).size;
      score = scored.length ? `${open} cluster${open === 1 ? "" : "s"} · ${chars} character${chars === 1 ? "" : "s"}` : "–";
    } else {
      const right = scored.filter((j) => steps[j][p].pred === truth(j)).length;
      score = scored.length ? `${right} of ${scored.length} right` : "–";
      if (p === "p4") {
        const wrong = R.stream.filter((_, j) => at(j) >= FILE && steps[j].p4.pred !== truth(j)).length;
        if (wrong) score += ` · ${wrong} wrong added`;
      }
    }

    return (
      <>
        <PanelFrame w={PW} h={PH} tag={meta.tag} title={meta.title} tone={meta.tone}
          note={p === "p3" ? `joins at ≥ ${R.tau.toFixed(2)}` : undefined} />
        {els}
        {line}
        <line x1={12} x2={PW - 12} y1={SCORE_Y - 17} y2={SCORE_Y - 17} stroke="#e6ded0" strokeWidth={1.5} strokeDasharray="3 4" />
        <text x={PW / 2} y={SCORE_Y} textAnchor="middle" fontSize={12.5} fontWeight={800} fill={INK} className="rp-display">{score}</text>
      </>
    );
  }

  const cur = clock.beat;
  const status = cur < 0 ? "ready" : `query ${cur + 1} of ${steps.length} · page ${R.crops[R.stream[cur]].page}`;

  return (
    <>
      <ReplayCard
        cardRef={loop.ref}
        tabs={isNarrow ? (
          <Tabs items={PANELS.map((p) => ({ id: p.id, label: p.tag }))} value={tab} onChange={setTab} />
        ) : undefined}
        ratio={L.w / L.h}
        fit={FIT}
        stage={(
          <svg viewBox={`0 0 ${L.w} ${L.h}`} className="rp-svg block h-auto w-full" role="img"
            aria-label="Eight crops of Bakuman chapter 1 answered by the four protocols in reading order: P1 gets all eight right, P2 seven, P4 five after filing one mistake under the wrong character, and P3 opens three clusters.">
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
        extra={(
          <button type="button" onClick={() => setFigure(true)} className="btn btn-sm !gap-1.5 !px-2.5 !py-1 !text-[0.8125rem]">
            <Icon name="zoom" size={14} /> Paper figure
          </button>
        )}
        lead="One stream, four galleries."
        caption={(
          <>
            MagiV2&rsquo;s own decisions (its released encoder) on eight crops of Bakuman chapter 1, against the
            galleries of the paper&rsquo;s protocols figure; the protocols run on whole chapters. The page-29 Mashiro crop
            sits closer to Azuki&rsquo;s seed: P1 still gets it right, P2 does not, and P4 files it under Azuki, where two
            later Mashiro crops match it.
          </>
        )}
      />
      {figure && <ImagePopup image={PROTOCOLS_FIGURE} onClose={closeFigure} />}
    </>
  );
}
