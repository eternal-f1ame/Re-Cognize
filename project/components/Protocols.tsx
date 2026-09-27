import { Section } from "./Section";
import { Figure } from "./Figure";
import { PROTOCOLS, PROTOCOLS_FIGURE } from "../content";

const TONE = {
  blue: { chip: "bg-[var(--blue-soft)] text-[var(--blue)]", bar: "bg-[var(--blue)]" },
  orange: { chip: "bg-[var(--orange-soft)] text-[var(--orange)]", bar: "bg-[var(--orange)]" },
};

export function Protocols() {
  return (
    <Section id="protocols" index={3} kicker="The framework" title="Four protocols, one stream"
      lead="Every protocol answers the same stream of query crops in reading order; they differ only in the gallery and whether it may change.">
      <Figure images={[PROTOCOLS_FIGURE]} lead="The four Re:Cognize protocols." caption={PROTOCOLS_FIGURE.caption} />
      <div className="mx-auto mt-10 grid max-w-[1000px] gap-5 sm:grid-cols-2 lg:grid-cols-4">
        {PROTOCOLS.map((p) => {
          const tone = TONE[p.tone as keyof typeof TONE];
          return (
            <article key={p.id} className="panel relative overflow-hidden p-5 pt-6">
              <span className={`absolute inset-x-0 top-0 h-1.5 ${tone.bar}`} aria-hidden="true" />
              <span className={`inline-block rounded-md border-2 border-[var(--line)] px-2 py-0.5 font-display text-base font-extrabold ${tone.chip}`}>
                {p.id}
              </span>
              <h3 className="mt-3 text-base font-bold">{p.name}</h3>
              <p className="mt-1.5 text-sm leading-relaxed text-[var(--ink-2)]">{p.text}</p>
            </article>
          );
        })}
      </div>
    </Section>
  );
}
