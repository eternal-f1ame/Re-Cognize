"use client";

import { useCallback, useState, type ReactNode } from "react";
import Image from "next/image";
import { ImagePopup, type ZoomImage } from "./ImagePopup";
import { Icon } from "./Icon";

// One or more stacked images in a figure card, with a caption; each image opens full screen on click.
export function Figure({ images, lead, caption, priority = false, max = "max-w-[1000px]", className = "" }: {
  images: ZoomImage[];
  lead?: string;
  caption?: ReactNode;
  priority?: boolean;
  max?: string;
  className?: string;
}) {
  const [open, setOpen] = useState<ZoomImage | null>(null);
  const close = useCallback(() => setOpen(null), []);

  return (
    <figure className={`mx-auto w-full ${max} ${className}`}>
      <div className="figure-card">
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
              sizes="(min-width: 1040px) 1000px, 100vw"
              className="h-auto w-full rounded-md"
              priority={priority}
              unoptimized={img.src.endsWith(".svg")}
            />
            <span className="pointer-events-none absolute right-2 top-2 hidden rounded-full border-2 border-[var(--line)] bg-[var(--card)] p-1.5 opacity-0 transition-opacity group-hover:opacity-100 sm:block">
              <Icon name="zoom" size={14} />
            </span>
          </button>
        ))}
      </div>
      {(lead || caption) && (
        <figcaption className="figure-caption mt-3 px-1">
          {lead && <strong>{lead} </strong>}
          {caption}
        </figcaption>
      )}
      {open && <ImagePopup image={open} onClose={close} />}
    </figure>
  );
}
