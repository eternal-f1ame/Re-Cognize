import { Section } from "./Section";
import { Figure } from "./Figure";
import type { Fit } from "./fit";
import { RecastReplay } from "./replay/RecastReplay";
import { BINDING_CAPTION, BINDING_FIGURES, RECAST_CAPTION, RECAST_CHANGES, RECAST_FIGURES } from "../content";

// The schematic shares the screen with its card padding and its caption below.
const SCHEMATIC_FIT: Fit = { reserve: "7rem" };
// Stacked, the binding figure sits under its text (about 12rem on a tablet); side by side, only its
// card padding shares the height.
const BINDING_FIT: Fit = { reserve: "15rem", reserveLg: "3rem" };

export function ReCast() {
  return (
    <Section id="recast" index={5} kicker="Re:Cast" title="A cast that reads along"
      lead="A cast sheet of one running average per character, grown only where the page itself vouches for a crop, with nothing fitted on data.">
      <RecastReplay />

      {/* the schematic, with the three changes it draws beside it on wide screens */}
      <div data-fit-row="" className="mt-8 grid items-start gap-x-[2.5vw] gap-y-6 lg:grid-cols-[minmax(0,2.3fr)_minmax(0,1fr)]">
        <Figure images={RECAST_FIGURES} lead="Re:Cast" caption={RECAST_CAPTION} fit={SCHEMATIC_FIT}
          sizes="(min-width: 64rem) 66vw, 100vw" />
        <div className="grid gap-4 md:grid-cols-3 lg:grid-cols-1">
          {RECAST_CHANGES.map((c) => (
            <article key={c.title} className="panel p-4">
              <div className="flex items-center gap-3">
                <span className="grid h-8 w-8 shrink-0 place-items-center rounded-full border-2 border-[var(--line)] bg-[var(--orange)] font-display text-sm font-extrabold text-[var(--paper)]">
                  {c.label}
                </span>
                <h3 className="text-base font-bold leading-snug">{c.title}</h3>
              </div>
              <p className="mt-2 text-sm leading-relaxed text-[var(--ink-2)]">{c.text}</p>
            </article>
          ))}
        </div>
      </div>

      {/* binding: its text beside the figure on wide screens, above it on narrow ones */}
      <div data-fit-unit="card" data-fit-row="" className="mt-12 grid items-center gap-x-[2.5vw] gap-y-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,2.3fr)]">
        <div>
          <p className="kicker">Chronological seeding</p>
          <h3 className="mt-1.5 text-xl font-extrabold">Binding, and pricing the binder</h3>
          <p className="mt-1.5 max-w-[70ch] text-[0.9688rem] leading-relaxed text-[var(--ink-2)]">
            When the seeds are each character&rsquo;s first appearances, binding supplies the reference: page groups
            merged in reading order and named by their first seed, committed only where the commit condition predicts
            a gain.
          </p>
          {/* beside the figure on wide screens; under it (the figure's own caption) on narrow ones */}
          <p className="figure-caption mt-3 hidden max-w-[70ch] lg:block">{BINDING_CAPTION}</p>
        </div>
        <Figure images={BINDING_FIGURES} fit={BINDING_FIT} sizes="(min-width: 64rem) 66vw, 100vw"
          lead="Binding, and pricing the binder." caption={BINDING_CAPTION} captionClassName="lg:hidden" />
      </div>
    </Section>
  );
}
