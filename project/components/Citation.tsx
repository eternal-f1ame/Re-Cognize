"use client";

import { useState } from "react";
import { Section } from "./Section";
import { Icon } from "./Icon";
import { BIBTEX } from "../content";

export function Citation() {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(BIBTEX);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  };

  return (
    <Section id="citation" index={8} kicker="Citation" alt title="Cite Re:Cognize">
      <div data-fit-row="" className="relative overflow-hidden rounded-[10px] border-2 border-[var(--line)] bg-[var(--ink)] shadow-[4px_4px_0_var(--orange)]">
        <div className="flex items-center justify-between border-b border-white/15 px-4 py-2.5">
          <span className="font-mono text-xs uppercase tracking-[0.14em] text-[var(--paper-2)]/70">BibTeX</span>
          <button type="button" onClick={copy}
            className="inline-flex items-center gap-1.5 rounded-full border border-white/25 px-3 py-1 text-xs font-semibold text-[var(--paper)] transition-colors hover:bg-white/10">
            <Icon name={copied ? "check" : "copy"} size={14} /> {copied ? "Copied" : "Copy"}
          </button>
        </div>
        <pre className="overflow-x-auto whitespace-pre-wrap break-words px-4 py-4 font-mono text-[0.8125rem] leading-relaxed text-[var(--paper)] md:whitespace-pre md:px-6 md:text-sm">
          {BIBTEX}
        </pre>
      </div>
    </Section>
  );
}
