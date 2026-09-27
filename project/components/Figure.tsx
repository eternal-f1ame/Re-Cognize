"use client";

import { useCallback, useState, type ReactNode } from "react";
import Image from "next/image";
import { ImagePopup, type ZoomImage } from "./ImagePopup";
import styles from "./styles/Section.module.css";

// One or more stacked images with a caption; each image opens full screen on click.
export function Figure({ images, lead, caption, priority = false }: {
  images: ZoomImage[];
  lead?: string;
  caption?: ReactNode;
  priority?: boolean;
}) {
  const [open, setOpen] = useState<ZoomImage | null>(null);
  const close = useCallback(() => setOpen(null), []);

  return (
    <figure className={`manga-panel ${styles.figurePanel}`}>
      {images.map((img) => (
        <button key={img.src} type="button" className={styles.figureButton} onClick={() => setOpen(img)}
          aria-label={`Enlarge: ${img.alt}`}>
          <Image
            src={img.src}
            alt={img.alt}
            width={img.width}
            height={img.height}
            sizes="(min-width: 1280px) 1200px, 100vw"
            className={styles.figureImage}
            priority={priority}
            unoptimized={img.src.endsWith(".svg")}
          />
        </button>
      ))}
      <p className={styles.zoomHint}>Click or tap to enlarge</p>
      {(lead || caption) && (
        <figcaption className={styles.caption}>
          {lead && <span className={styles.captionLead}>{lead} </span>}
          {caption}
        </figcaption>
      )}
      {open && <ImagePopup image={open} onClose={close} />}
    </figure>
  );
}
