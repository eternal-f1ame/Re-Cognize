import { Section } from "./Section";
import { ABSTRACT } from "../content";

export function Abstract() {
  const [first, ...rest] = ABSTRACT.slice(0, -1);
  const scope = ABSTRACT[ABSTRACT.length - 1];
  return (
    <Section id="abstract" index={1} kicker="Abstract" title="Reading along, not matching against a cast list">
      <div className="grid gap-6 lg:grid-cols-12">
        <div className="space-y-3.5 text-[0.9688rem] leading-[1.7] text-[var(--ink-2)] lg:col-span-8">
          <p className="first-letter:float-left first-letter:mr-2 first-letter:font-display first-letter:text-[2.9rem] first-letter:font-extrabold first-letter:leading-[0.85] first-letter:text-[var(--orange)]">
            {first}
          </p>
          {rest.map((p) => (
            <p key={p.slice(0, 32)}>{p}</p>
          ))}
        </div>
        <aside className="lg:col-span-4">
          <div className="panel p-4 lg:sticky lg:top-[calc(var(--nav-h)+1.5rem)]">
            <p className="kicker">Scope of the claims</p>
            <p className="mt-2 text-[0.875rem] leading-relaxed text-[var(--ink-2)]">{scope}</p>
          </div>
        </aside>
      </div>
    </Section>
  );
}
