import { Section } from "./Section";
import { Figure } from "./Figure";
import type { Fit } from "./fit";
import { RecastReplay } from "./replay/RecastReplay";
import { BINDING_CAPTION, BINDING_FIGURES, RECAST_CAPTION, RECAST_CHANGES, RECAST_FIGURES } from "../content";

// the subheader (about 110 px at the column's width) and the card's padding; the caption may run below
const BINDING_FIT: Fit = { max: 1000, reserve: 164 };

export function ReCast() {
  return (
    <Section id="recast" index={5} kicker="Re:Cast" title="A cast that reads along"
      lead="A cast sheet of one running average per character, grown only where the page itself vouches for a crop, with nothing fitted on data.">
      <RecastReplay />
      <Figure images={RECAST_FIGURES} lead="Re:Cast" caption={RECAST_CAPTION} fit={{ max: 1000, reserve: 110 }} className="mt-8" />
      <div className="mx-auto mt-7 grid max-w-[1000px] gap-4 md:grid-cols-3">
        {RECAST_CHANGES.map((c) => (
          <article key={c.title} className="panel p-4">
            <span className="grid h-8 w-8 place-items-center rounded-full border-2 border-[var(--line)] bg-[var(--orange)] font-display text-sm font-extrabold text-[var(--paper)]">
              {c.label}
            </span>
            <h3 className="mt-3 text-base font-bold">{c.title}</h3>
            <p className="mt-1.5 text-sm leading-relaxed text-[var(--ink-2)]">{c.text}</p>
          </article>
        ))}
      </div>

      {/* the subheader and its figure share one screen; the subheader keeps the column's width so its
          height does not grow as the figure narrows */}
      <div data-fit-unit="card" className="mt-12">
        <div className="mx-auto max-w-[1000px]">
          <p className="kicker">Chronological seeding</p>
          <h3 className="mt-1.5 text-xl font-extrabold">Binding, and pricing the binder</h3>
          <p className="mt-1.5 max-w-4xl text-[0.9688rem] leading-relaxed text-[var(--ink-2)]">
            When the seeds are each character&rsquo;s first appearances, binding supplies the reference: page groups
            merged in reading order and named by their first seed, committed only where the commit condition predicts
            a gain.
          </p>
        </div>
        <Figure images={BINDING_FIGURES} lead="Binding, and pricing the binder." caption={BINDING_CAPTION}
          fit={BINDING_FIT} className="mt-4" />
      </div>
    </Section>
  );
}
