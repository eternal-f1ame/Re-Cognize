"use client";

import { useCallback, useState } from "react";
import Image from "next/image";
import { Section } from "./Section";
import { ImagePopup, type ZoomImage } from "./ImagePopup";
import { HEADROOM_CAPTION, HEADROOM_PANELS } from "../content";

// The four panels share one aspect ratio (190 x 95 pt in the paper).
const PANEL = { width: 1520, height: 760 };

export function Results() {
  const [open, setOpen] = useState<ZoomImage | null>(null);
  const close = useCallback(() => setOpen(null), []);

  return (
    <Section id="results" index={6} kicker="Results" alt title="What the protocols measure, and where the headroom is"
      lead="Five backbones, two corpora: one seed recovers the closed-set mAP, correct growth would add over twenty points, and the commit condition says which changes pay.">
      <figure className="mx-auto max-w-[1000px]">
        <div className="grid gap-4 sm:grid-cols-2">
          {HEADROOM_PANELS.map((p) => {
            const img = { src: p.src, alt: p.caption, ...PANEL };
            return (
              <div key={p.src} className="figure-card flex flex-col">
                <button type="button" onClick={() => setOpen(img)} aria-label={`Enlarge: ${p.caption}`}
                  className="block w-full cursor-zoom-in rounded-md">
                  <Image src={p.src} alt={p.caption} {...PANEL} className="h-auto w-full" unoptimized />
                </button>
                <p className="figure-caption mt-2 flex gap-2 border-t-2 border-dashed border-[var(--paper-2)] pt-2">
                  <span className="grid h-5 w-5 shrink-0 place-items-center rounded-full bg-[var(--ink)] font-mono text-[0.6875rem] font-bold text-[var(--paper)]">
                    {p.label}
                  </span>
                  {p.caption}
                </p>
              </div>
            );
          })}
        </div>
        <figcaption className="figure-caption mt-4 px-1">
          <strong>What the protocols measure, and where the headroom is. </strong>
          {HEADROOM_CAPTION}
        </figcaption>
      </figure>
      {open && <ImagePopup image={open} onClose={close} />}
    </Section>
  );
}
