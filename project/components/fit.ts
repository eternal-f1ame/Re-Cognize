import type { CSSProperties } from "react";

// How a visual fits the screen: at most `max` px wide, and narrow enough that it plus `reserve` px
// (its card padding and whatever else must share the screen with it) fits under the nav.
// The rule itself is `.fit-h` in app/globals.css; these helpers set its variables.
export type Fit = { max: number; reserve: number };

// The width-to-height ratio of images stacked one above the other at a common width.
export const stackRatio = (images: { width: number; height: number }[]) =>
  1 / images.reduce((h, img) => h + img.height / img.width, 0);

export const fitStyle = (ratio: number, fit: Fit): CSSProperties =>
  ({ "--fit-ratio": ratio, "--fit-reserve": `${fit.reserve}px`, "--fit-max": `${fit.max}px` }) as CSSProperties;
