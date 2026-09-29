import type { CSSProperties } from "react";

// How a visual fits the screen. It is as wide as its column, unless that would make it taller than the
// height under the nav less `reserve` (whatever shares the screen with it: header, card padding,
// controls). Reserves are CSS lengths in rem, so they scale with the type. Rows that reflow into
// columns at lg or xl change the visual's ratio or reserve there. The rule is `.fit-h` in globals.css.
export type Fit = {
  reserve: string;
  reserveLg?: string;
  reserveXl?: string;
  ratioLg?: number;
  ratioXl?: number;
};

// The width-to-height ratio of images stacked one above the other at a common width.
export const stackRatio = (images: { width: number; height: number }[]) =>
  1 / images.reduce((h, img) => h + img.height / img.width, 0);

export const fitStyle = (ratio: number, fit: Fit): CSSProperties => {
  const vars: Record<string, string | number> = { "--fit-ratio": ratio, "--fit-reserve": fit.reserve };
  if (fit.reserveLg) vars["--fit-reserve-lg"] = fit.reserveLg;
  if (fit.reserveXl) vars["--fit-reserve-xl"] = fit.reserveXl;
  if (fit.ratioLg) vars["--fit-ratio-lg"] = fit.ratioLg;
  if (fit.ratioXl) vars["--fit-ratio-xl"] = fit.ratioXl;
  return vars as CSSProperties;
};
