"use client";

import { useState } from "react";
import { Figure } from "./Figure";
import { Icon } from "./Icon";
import { AFFILIATION, AUTHOR_NOTE, AUTHORS, CODE_URL, PAPER_URL, RESULTS_URL, STATS, TEASER, VENUE } from "../content";

export function Hero() {
  const [paperNote, setPaperNote] = useState(false);

  return (
    <header id="top" className="relative overflow-hidden border-b-2 border-[var(--line)]">
      <div className="halftone halftone-fade pointer-events-none absolute inset-x-0 top-0 h-[70%] opacity-[0.08]" aria-hidden="true" />
      {/* Stacked and centred on narrow screens; on wide ones the title block and the numbers sit to the
          left of the teaser, so the first screen uses the whole width. */}
      <div className="page relative py-[clamp(1rem,3.5vh,2.25rem)]">
        <div
          data-fit-row=""
          className="grid items-center gap-x-[3vw] gap-y-5 text-center [grid-template-areas:'text'_'fig'_'stats'] lg:grid-cols-[minmax(0,1fr)_minmax(0,1.5fr)] lg:text-left lg:[grid-template-areas:'text_fig'_'stats_fig']"
        >
          <div className="[grid-area:text] lg:self-end">
            <a
              href={VENUE.url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 rounded-full border-2 border-[var(--line)] bg-[var(--card)] px-3 py-0.5 text-xs font-semibold shadow-[var(--shadow-sm)] sm:text-[0.8125rem]"
            >
              <span className="h-2 w-2 rounded-full bg-[var(--orange)]" aria-hidden="true" />
              {VENUE.label}
            </a>

            <h1 className="mt-3 text-[clamp(2.375rem,4.6vw,4.75rem)] font-extrabold leading-[0.95]">
              Re<span className="text-[var(--orange)]">:</span>Cognize
            </h1>
            <p className="mt-2 font-display text-[clamp(1.0625rem,1.55vw,1.5rem)] font-semibold leading-snug text-[var(--ink-2)]">
              Open-Set Comic Character <span className="whitespace-nowrap">Re-Identification</span>
            </p>

            <p className="mt-3 text-[0.9688rem] font-semibold sm:text-base">
              {AUTHORS.map((a, i) => (
                <span key={a.name} className="whitespace-nowrap">
                  {a.url ? (
                    <a href={a.url} target="_blank" rel="noopener noreferrer" className="underline decoration-[var(--orange)] decoration-2 underline-offset-4 hover:text-[var(--orange)]">
                      {a.name}
                    </a>
                  ) : (
                    a.name
                  )}
                  {a.mark && <sup className="ml-0.5 text-[var(--muted)]">{a.mark}</sup>}
                  {i < AUTHORS.length - 1 && ","}
                </span>
              )).flatMap((el, i) => (i === 0 ? [el] : [" ", el]))}
            </p>
            <p className="mt-1 text-[0.8125rem] text-[var(--muted)]">{AFFILIATION}</p>
            <p className="mt-0.5 text-[0.6875rem] text-[var(--muted)]">
              <sup>{AUTHOR_NOTE.mark}</sup>{AUTHOR_NOTE.text}
            </p>

            <div className="mt-4 flex flex-wrap items-center justify-center gap-2.5 lg:justify-start">
              {PAPER_URL ? (
                <a href={PAPER_URL} target="_blank" rel="noopener noreferrer" className="btn btn-sm">
                  <Icon name="paper" size={17} /> Paper
                </a>
              ) : (
                <span className="relative">
                  <span
                    className="btn btn-sm btn-disabled"
                    aria-disabled="true"
                    tabIndex={0}
                    onMouseEnter={() => setPaperNote(true)}
                    onMouseLeave={() => setPaperNote(false)}
                    onFocus={() => setPaperNote(true)}
                    onBlur={() => setPaperNote(false)}
                  >
                    <Icon name="paper" size={17} /> Paper
                  </span>
                  {paperNote && (
                    <span className="absolute left-1/2 top-full z-10 mt-2 -translate-x-1/2 whitespace-nowrap rounded-md bg-[var(--ink)] px-2.5 py-1 text-xs text-[var(--paper)]">
                      arXiv link coming soon
                    </span>
                  )}
                </span>
              )}
              <a href={CODE_URL} target="_blank" rel="noopener noreferrer" className="btn btn-sm btn-primary">
                <Icon name="github" size={17} /> Code
              </a>
              <a href={RESULTS_URL} target="_blank" rel="noopener noreferrer" className="btn btn-sm">
                <Icon name="package" size={17} /> Results
              </a>
              <a href="#citation" className="btn btn-sm">
                <Icon name="book" size={17} /> BibTeX
              </a>
            </div>
          </div>

          <Figure
            images={[TEASER]}
            lead={TEASER.lead}
            caption={TEASER.caption}
            priority
            sizes="(min-width: 64rem) 58vw, 100vw"
            fit={{ reserve: "24rem", reserveLg: "11rem" }}
            className="text-left [grid-area:fig]"
          />

          <dl className="grid grid-cols-2 gap-3 text-left [grid-area:stats] md:grid-cols-4 lg:grid-cols-2 lg:self-start">
            {STATS.map((s) => (
              <div key={s.label} className="panel-flat px-3.5 py-2.5">
                <dt className="sr-only">{s.label}</dt>
                <dd className="font-display text-[1.75rem] font-extrabold leading-none text-[var(--ink)]">{s.value}</dd>
                <dd className="mt-1 text-[0.8125rem] leading-snug text-[var(--muted)]">{s.label}</dd>
              </div>
            ))}
          </dl>
        </div>
      </div>
    </header>
  );
}
