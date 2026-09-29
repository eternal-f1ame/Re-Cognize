"use client";

import { useCallback, useState, type ReactNode } from "react";
import Image from "next/image";
import { ImagePopup, type ZoomImage } from "./ImagePopup";
import { Icon } from "./Icon";
import { fitStyle, stackRatio, type Fit } from "./fit";

// The card itself: one or more stacked images, each opening full screen on click.
export function FigureCard({ images, priority = false, sizes = "(min-width: 64rem) 70vw, 100vw" }: {
  images: ZoomImage[];
  priority?: boolean;
  sizes?: string;
}) {
  const [open, setOpen] = useState<ZoomImage | null>(null);
  const close = useCallback(() => setOpen(null), []);
  return (
    <div className="figure-card" data-fit-unit="">
      {images.map((img, i) => (
        <button
          key={img.src}
          type="button"
          onClick={() => setOpen(img)}
          aria-label={`Enlarge: ${img.alt}`}
          className={`group relative block w-full cursor-zoom-in rounded-md ${i > 0 ? "mt-2" : ""}`}
        >
          <Image
            src={img.src}
            alt={img.alt}
            width={img.width}
            height={img.height}
            sizes={sizes}
            className="h-auto w-full rounded-md"
            priority={priority}
            unoptimized={img.src.endsWith(".svg")}
          />
          <span className="pointer-events-none absolute right-2 top-2 hidden rounded-full border-2 border-[var(--line)] bg-[var(--card)] p-1.5 opacity-0 transition-opacity group-hover:opacity-100 sm:block">
            <Icon name="zoom" size={14} />
          </span>
        </button>
      ))}
      {open && <ImagePopup image={open} onClose={close} />}
    </div>
  );
}

// A figure: the card, fitted to its column and the screen's height, with an optional caption below.
export function Figure({ images, lead, caption, priority = false, fit, sizes, className = "", captionClassName = "" }: {
  images: ZoomImage[];
  lead?: string;
  caption?: ReactNode;
  priority?: boolean;
  fit: Fit;
  sizes?: string;
  className?: string;
  captionClassName?: string;
}) {
  return (
    <figure className={`fit-h ${className}`} style={fitStyle(stackRatio(images), fit)}>
      <FigureCard images={images} priority={priority} sizes={sizes} />
      {(lead || caption) && (
        <figcaption className={`figure-caption mt-2.5 px-1 ${captionClassName}`}>
          {lead && <strong>{lead} </strong>}
          {caption}
        </figcaption>
      )}
    </figure>
  );
}
