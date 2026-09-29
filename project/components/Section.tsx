import type { ReactNode } from "react";

// A numbered section: kicker, then the title with its lead beside it on wide screens, content below.
// `alt` shades the band. Kept short so a section's header and its first visual fit one screen.
export function Section({ id, index, kicker, title, lead, alt = false, children }: {
  id: string;
  index: number;
  kicker: string;
  title: string;
  lead?: ReactNode;
  alt?: boolean;
  children: ReactNode;
}) {
  return (
    <section id={id} className={`section-anchor ${alt ? "border-y-2 border-[var(--line)] bg-[var(--paper-2)]" : ""}`}>
      <div className="page section-pad">
        <header className="mb-5">
          <p className="kicker">{String(index).padStart(2, "0")} · {kicker}</p>
          <div className={`mt-2 ${lead ? "grid gap-2 lg:grid-cols-[minmax(0,1.05fr)_minmax(0,1fr)] lg:items-end lg:gap-10" : ""}`}>
            <h2 className="max-w-[30ch] text-[clamp(1.5rem,2.3vw,2.125rem)] font-extrabold leading-tight">{title}</h2>
            {lead && <p className="max-w-[62ch] text-[0.9688rem] leading-relaxed text-[var(--ink-2)] lg:pb-1">{lead}</p>}
          </div>
        </header>
        {children}
      </div>
    </section>
  );
}
