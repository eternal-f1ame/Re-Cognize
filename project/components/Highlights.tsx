import { Section } from "./Section";
import { Icon, type IconName } from "./Icon";
import { HIGHLIGHTS, QUOTE } from "../content";

export function Highlights() {
  return (
    <Section id="findings" index={2} kicker="Key findings" alt
      title="Where models fail" lead="Recognising a character is close to solved. Deciding which of your own matches to believe is not.">
      <div data-fit-row="" className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {HIGHLIGHTS.map((h, i) => (
          <article key={h.title} className="panel flex flex-col p-4">
            <div className="flex items-center justify-between">
              <span className="grid h-9 w-9 place-items-center rounded-full border-2 border-[var(--line)] bg-[var(--orange-soft)]">
                <Icon name={h.icon as IconName} size={17} />
              </span>
              <span className="font-mono text-xs font-semibold text-[var(--muted)]">{String(i + 1).padStart(2, "0")}</span>
            </div>
            <h3 className="mt-3 text-base font-bold leading-snug">{h.title}</h3>
            <p className="mt-1.5 text-sm leading-relaxed text-[var(--ink-2)]">{h.text}</p>
          </article>
        ))}
      </div>
      <blockquote className="relative mx-auto mt-8 max-w-3xl rounded-[18px] border-2 border-[var(--line)] bg-[var(--card)] px-6 py-4 text-center shadow-[var(--shadow)]">
        <p className="font-display text-lg font-semibold leading-snug md:text-xl">&ldquo;{QUOTE}&rdquo;</p>
        <span aria-hidden="true" className="absolute -bottom-[14px] left-12 h-6 w-6 rotate-45 border-b-2 border-r-2 border-[var(--line)] bg-[var(--card)]" />
      </blockquote>
    </Section>
  );
}
