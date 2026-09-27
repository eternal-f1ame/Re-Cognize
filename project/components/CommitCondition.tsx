import { Section } from "./Section";
import { COMMIT_INTRO, COMMIT_OUTRO, COMMIT_TERMS } from "../content";

export function CommitCondition() {
  return (
    <Section id="commit-condition" index={4} kicker="The commit condition" alt title="When a change to the gallery pays"
      lead="The bottleneck is acceptance, not vision, and one comparison decides it.">
      <div className="mx-auto max-w-[1000px]">
        <div className="panel px-5 py-8 text-center md:px-10">
          <p className="mx-auto max-w-2xl text-[1.0625rem] leading-relaxed text-[var(--ink-2)]">{COMMIT_INTRO}</p>
          <p className="math my-6 text-[clamp(2rem,5vw,3.25rem)]" aria-label="Delta equals c times p-eff minus a-plus">
            &Delta; = <i>c</i>&thinsp;(<i>p</i><sub className="text-[0.55em]">eff</sub> &minus; <i>a</i><sup className="text-[0.55em]">+</sup>)
          </p>
          <div className="grid gap-4 text-left md:grid-cols-3">
            {COMMIT_TERMS.map((t) => (
              <div key={t.name} className="rounded-lg border-2 border-dashed border-[var(--line)] bg-[var(--paper)] p-4">
                <p className="math text-2xl">{t.symbol}</p>
                <p className="mt-1 text-sm font-bold">{t.name}</p>
                <p className="mt-1 text-sm leading-relaxed text-[var(--ink-2)]">{t.text}</p>
              </div>
            ))}
          </div>
          <p className="mx-auto mt-6 max-w-2xl text-[1.0625rem] leading-relaxed text-[var(--ink-2)]">{COMMIT_OUTRO}</p>
        </div>
      </div>
    </Section>
  );
}
