"use client";

import { useState } from "react";
import { Figure } from "./Figure";
import { Icon } from "./Icon";
import { AFFILIATION, AUTHOR_NOTE, AUTHORS, CODE_URL, PAPER_URL, RESULTS_URL, STATS, TEASER, VENUE } from "../content";

export function Hero() {
  const [paperNote, setPaperNote] = useState(false);

  return (
    <header id="top" className="relative overflow-hidden border-b-2 border-[var(--line)]">
      <div className="halftone halftone-fade pointer-events-none absolute inset-x-0 top-0 h-[420px] opacity-[0.08]" aria-hidden="true" />
      <div className="relative mx-auto max-w-6xl px-4 pb-9 pt-5 text-center sm:px-6 md:pt-6">
        <a
          href={VENUE.url}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-2 rounded-full border-2 border-[var(--line)] bg-[var(--card)] px-3 py-0.5 text-xs font-semibold shadow-[var(--shadow-sm)] sm:text-[0.8125rem]"
        >
          <span className="h-2 w-2 rounded-full bg-[var(--orange)]" aria-hidden="true" />
          {VENUE.label}
        </a>

        <h1 className="mt-3 text-[clamp(2.375rem,5vw,3.625rem)] font-extrabold leading-[0.95]">
          Re<span className="text-[var(--orange)]">:</span>Cognize
        </h1>
        <p className="mt-1.5 font-display text-[clamp(1.0625rem,1.8vw,1.375rem)] font-semibold leading-snug text-[var(--ink-2)]">
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

        <div className="mt-4 flex flex-wrap items-center justify-center gap-2.5">
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

        <Figure
          images={[TEASER]}
          lead={TEASER.lead}
          caption={TEASER.caption}
          priority
          fit={{ max: 880, reserve: 344 }}
          className="mt-5 text-left"
        />

        <dl className="mx-auto mt-8 grid max-w-[880px] grid-cols-2 gap-3 text-left sm:gap-4 md:grid-cols-4">
          {STATS.map((s) => (
            <div key={s.label} className="panel-flat px-4 py-3">
              <dt className="sr-only">{s.label}</dt>
              <dd className="font-display text-3xl font-extrabold leading-none text-[var(--ink)]">{s.value}</dd>
              <dd className="mt-1.5 text-[0.8125rem] leading-snug text-[var(--muted)]">{s.label}</dd>
            </div>
          ))}
        </dl>
      </div>
    </header>
  );
}
