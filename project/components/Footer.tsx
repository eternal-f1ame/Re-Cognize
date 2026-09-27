import Image from "next/image";
import { CODE_URL, INSTITUTIONS, RESULTS_URL } from "../content";

export function Footer() {
  return (
    <footer className="border-t-2 border-[var(--line)]">
      <div className="mx-auto flex max-w-6xl flex-col items-center gap-6 px-4 py-10 sm:px-6 md:flex-row md:justify-between">
        <div className="flex flex-wrap items-center justify-center gap-6">
          {INSTITUTIONS.map((inst) => (
            <a key={inst.name} href={inst.url} target="_blank" rel="noopener noreferrer"
              className="flex items-center gap-3 text-sm font-semibold text-[var(--ink-2)] hover:text-[var(--ink)]">
              <Image src={inst.logo} alt="" width={400} height={400} className="h-10 w-10 object-contain" />
              {inst.name}
            </a>
          ))}
        </div>
        <p className="text-center text-sm text-[var(--muted)] md:text-right">
          Re<span className="text-[var(--orange)]">:</span>Cognize · NeurIPS 2026 ·{" "}
          <a href={CODE_URL} target="_blank" rel="noopener noreferrer" className="underline underline-offset-4 hover:text-[var(--ink)]">Code</a> ·{" "}
          <a href={RESULTS_URL} target="_blank" rel="noopener noreferrer" className="underline underline-offset-4 hover:text-[var(--ink)]">Results</a>
        </p>
      </div>
    </footer>
  );
}
