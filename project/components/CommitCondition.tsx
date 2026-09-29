import { Section } from "./Section";
import { COMMIT_INTRO, COMMIT_OUTRO, COMMIT_TERMS } from "../content";

export function CommitCondition() {
  return (
    <Section id="commit-condition" index={4} kicker="The commit condition" alt title="When a change to the gallery pays"
      lead="The bottleneck is acceptance, not vision, and one comparison decides it.">
      <div>
        <div data-fit-row="" className="panel px-5 py-5 text-center md:px-8 md:py-6">
          <p className="mx-auto max-w-[80ch] text-[0.9688rem] leading-relaxed text-[var(--ink-2)]">{COMMIT_INTRO}</p>
          <p className="math my-3 text-[clamp(1.75rem,3.4vw,2.5rem)] leading-tight" aria-label="Delta equals c times p-eff minus a-plus">
            &Delta; = <i>c</i>&thinsp;(<i>p</i><sub className="text-[0.55em]">eff</sub> &minus; <i>a</i><sup className="text-[0.55em]">+</sup>)
          </p>
          <div className="grid gap-3 text-left md:grid-cols-3">
            {COMMIT_TERMS.map((t) => (
              <div key={t.name} className="rounded-lg border-2 border-dashed border-[var(--line)] bg-[var(--paper)] px-4 py-3">
                <p className="math text-xl leading-none">{t.symbol}</p>
                <p className="mt-1 text-sm font-bold">{t.name}</p>
                <p className="mt-1 text-[0.8125rem] leading-relaxed text-[var(--ink-2)]">{t.text}</p>
              </div>
            ))}
          </div>
          <p className="mx-auto mt-4 max-w-[80ch] text-[0.9688rem] leading-relaxed text-[var(--ink-2)]">{COMMIT_OUTRO}</p>
        </div>
      </div>
    </Section>
  );
}
