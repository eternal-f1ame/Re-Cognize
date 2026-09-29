"use client";

import { useCallback, useState } from "react";
import Image from "next/image";
import { Section } from "./Section";
import { ImagePopup, type ZoomImage } from "./ImagePopup";
import { fitStyle } from "./fit";
import { HEADROOM_CAPTION, HEADROOM_PANELS } from "../content";

// The four panels share one aspect ratio (190 x 95 pt in the paper).
const PANEL = { width: 1520, height: 760 };

export function Results() {
  const [open, setOpen] = useState<ZoomImage | null>(null);
  const close = useCallback(() => setOpen(null), []);

  return (
    <Section id="results" index={6} kicker="Results" alt title="What the protocols measure, and where the headroom is"
      lead="Five backbones, two corpora: one seed recovers the closed-set mAP, correct growth would add over twenty points, and the commit condition says which changes pay.">
      {/* 2:1 plots. Two by two the grid is about half as tall as it is wide plus the card chrome, and shares
          the screen with the section header; four in a row (xl) it is an eighth as tall, so it takes the
          full width. */}
      <figure className="fit-h" style={fitStyle(2, { reserve: "18rem", ratioXl: 8, reserveXl: "11rem" })}>
        <div data-fit-unit="desktop" data-fit-row="" className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {HEADROOM_PANELS.map((p) => {
            const img = { src: p.src, alt: p.caption, ...PANEL };
            return (
              <div key={p.src} data-fit-unit="phone" className="figure-card flex flex-col">
                <button type="button" onClick={() => setOpen(img)} aria-label={`Enlarge: ${p.caption}`}
                  className="block w-full cursor-zoom-in rounded-md">
                  <Image src={p.src} alt={p.caption} {...PANEL} className="h-auto w-full" unoptimized />
                </button>
                <p className="figure-caption mt-1.5 flex items-center gap-2 truncate border-t-2 border-dashed border-[var(--paper-2)] pt-1.5 !text-[0.8125rem]">
                  <span className="grid h-5 w-5 shrink-0 place-items-center rounded-full bg-[var(--ink)] font-mono text-[0.6875rem] font-bold text-[var(--paper)]">
                    {p.label}
                  </span>
                  {p.caption}
                </p>
              </div>
            );
          })}
        </div>
        <figcaption className="figure-caption mt-3 px-1">
          <strong>What the protocols measure, and where the headroom is. </strong>
          {HEADROOM_CAPTION}
        </figcaption>
      </figure>
      {open && <ImagePopup image={open} onClose={close} />}
    </Section>
  );
}
