import { Section } from "./Section";
import { Figure } from "./Figure";
import { RecastReplay } from "./replay/RecastReplay";
import { BINDING_CAPTION, BINDING_FIGURES, RECAST_CAPTION, RECAST_CHANGES, RECAST_FIGURES } from "../content";

export function ReCast() {
  return (
    <Section id="recast" index={5} kicker="Re:Cast" title="A cast that reads along"
      lead="A cast sheet of one running average per character, grown only where the page itself vouches for a crop, with nothing fitted on data.">
      <RecastReplay />
      <Figure images={RECAST_FIGURES} lead="Re:Cast" caption={RECAST_CAPTION} className="mt-10" />
      <div className="mx-auto mt-10 grid max-w-[1000px] gap-5 md:grid-cols-3">
        {RECAST_CHANGES.map((c) => (
          <article key={c.title} className="panel p-5">
            <span className="grid h-8 w-8 place-items-center rounded-full border-2 border-[var(--line)] bg-[var(--orange)] font-display text-sm font-extrabold text-[var(--paper)]">
              {c.label}
            </span>
            <h3 className="mt-3 text-base font-bold">{c.title}</h3>
            <p className="mt-1.5 text-sm leading-relaxed text-[var(--ink-2)]">{c.text}</p>
          </article>
        ))}
      </div>

      <div className="mx-auto mt-16 max-w-[1000px]">
        <p className="kicker">Chronological seeding</p>
        <h3 className="mt-2 text-2xl font-extrabold">Binding, and pricing the binder</h3>
        <p className="mt-2 max-w-3xl text-[1.0625rem] leading-relaxed text-[var(--ink-2)]">
          When the seeds are each character&rsquo;s first appearances, binding supplies the reference: page groups
          merged in reading order and named by their first seed, committed only where the commit condition predicts
          a gain.
        </p>
      </div>
      <Figure images={BINDING_FIGURES} lead="Binding, and pricing the binder." caption={BINDING_CAPTION} className="mt-6" />
    </Section>
  );
}
