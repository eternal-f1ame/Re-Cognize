"use client";

import { useCallback, useState } from "react";
import Image from "next/image";
import { Section } from "./Section";
import { ImagePopup, type ZoomImage } from "./ImagePopup";
import styles from "./styles/Section.module.css";
import { HEADROOM_CAPTION, HEADROOM_PANELS } from "../content";

// The SVG panels share one aspect ratio (190 x 95 pt in the paper).
const PANEL = { width: 1520, height: 760 };

export function Results() {
  const [open, setOpen] = useState<ZoomImage | null>(null);
  const close = useCallback(() => setOpen(null), []);

  return (
    <Section id="results" title="What the Protocols Measure"
      subtitle="And where the headroom is, across five backbones and two corpora.">
      <figure className={`manga-panel ${styles.figurePanel}`}>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {HEADROOM_PANELS.map((p) => {
            const img = { src: p.src, alt: p.caption, ...PANEL };
            return (
              <div key={p.src}>
                <button type="button" className={styles.figureButton} onClick={() => setOpen(img)}
                  aria-label={`Enlarge: ${p.caption}`}>
                  <Image src={p.src} alt={p.caption} {...PANEL} className={styles.figureImage} unoptimized />
                </button>
                <p className={styles.caption} style={{ marginTop: "0.5rem" }}>
                  <span className={styles.panelLabel}>{p.label}</span>{p.caption}
                </p>
              </div>
            );
          })}
        </div>
        <p className={styles.zoomHint}>Click or tap to enlarge</p>
        <figcaption className={styles.caption}>
          <span className={styles.captionLead}>What the protocols measure, and where the headroom is. </span>
          {HEADROOM_CAPTION}
        </figcaption>
        {open && <ImagePopup image={open} onClose={close} />}
      </figure>
    </Section>
  );
}
