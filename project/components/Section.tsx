import type { ReactNode } from "react";

// A numbered section: kicker, title and lead on the left, content below. `alt` shades the band.
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
    <section id={id} className={alt ? "border-y-2 border-[var(--line)] bg-[var(--paper-2)]" : ""}>
      <div className="mx-auto max-w-6xl px-4 py-14 sm:px-6 md:py-20">
        <header className="mb-8 max-w-3xl md:mb-10">
          <p className="kicker">{String(index).padStart(2, "0")} · {kicker}</p>
          <h2 className="mt-3 text-[clamp(1.75rem,3.2vw,2.5rem)] font-extrabold leading-tight">{title}</h2>
          {lead && <p className="mt-3 text-lg leading-relaxed text-[var(--ink-2)]">{lead}</p>}
        </header>
        {children}
      </div>
    </section>
  );
}
