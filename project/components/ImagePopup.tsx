"use client";

import { useEffect } from "react";
import Image from "next/image";

export interface ZoomImage {
  src: string;
  alt: string;
  width: number;
  height: number;
}

// Full-screen view of one figure; closes on Escape, the close button or a click outside the image.
export function ImagePopup({ image, onClose }: { image: ZoomImage; onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = overflow;
    };
  }, [onClose]);

  return (
    <div role="dialog" aria-modal="true" aria-label={image.alt} className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/90" onClick={onClose} />
      <button
        onClick={onClose}
        aria-label="Close"
        className="absolute top-6 right-6 z-10 text-white text-3xl font-bold hover:scale-110 transition-transform duration-200"
        autoFocus
      >
        ✕
      </button>
      <div className="relative w-[94vw] h-[84vh] md:w-[88vw]">
        <Image
          src={image.src}
          alt={image.alt}
          fill
          sizes="94vw"
          className="object-contain"
          unoptimized={image.src.endsWith(".svg")}
        />
      </div>
    </div>
  );
}
