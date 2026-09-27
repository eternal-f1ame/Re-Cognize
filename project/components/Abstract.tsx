import { Section } from "./Section";
import { ABSTRACT } from "../content";

export function Abstract() {
  const [first, ...rest] = ABSTRACT.slice(0, -1);
  const scope = ABSTRACT[ABSTRACT.length - 1];
  return (
    <Section id="abstract" index={1} kicker="Abstract" title="Reading along, not matching against a cast list">
      <div className="grid gap-8 lg:grid-cols-12">
        <div className="space-y-5 text-[1.0625rem] leading-[1.8] text-[var(--ink-2)] lg:col-span-8">
          <p className="first-letter:float-left first-letter:mr-2 first-letter:font-display first-letter:text-[3.4rem] first-letter:font-extrabold first-letter:leading-[0.85] first-letter:text-[var(--orange)]">
            {first}
          </p>
          {rest.map((p) => (
            <p key={p.slice(0, 32)}>{p}</p>
          ))}
        </div>
        <aside className="lg:col-span-4">
          <div className="panel p-5 lg:sticky lg:top-[calc(var(--nav-h)+1.5rem)]">
            <p className="kicker">Scope of the claims</p>
            <p className="mt-3 text-[0.9375rem] leading-relaxed text-[var(--ink-2)]">{scope}</p>
          </div>
        </aside>
      </div>
    </Section>
  );
}
