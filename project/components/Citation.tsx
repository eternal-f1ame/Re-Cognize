"use client";

import { useState } from "react";
import { Section } from "./Section";
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
    <Section id="citation" title="Citation" subtitle="If you use Re:Cognize, please cite the paper:">
      <div className="manga-panel p-6 md:p-8">
        <div className="flex items-center justify-between mb-6 gap-4">
          <h3 className="text-xl md:text-2xl font-bold" style={{ color: "var(--manga-black)" }}>📝 BibTeX</h3>
          <button
            type="button"
            onClick={copy}
            className="inline-flex items-center px-4 py-2 rounded-lg font-medium transition-all duration-300 manga-panel hover:scale-105"
            style={{ color: copied ? "#16a34a" : "var(--manga-black)" }}
          >
            {copied ? "Copied!" : "Copy"}
          </button>
        </div>
        <div className="rounded-lg p-4 md:p-6 border-2 border-dashed overflow-x-auto"
          style={{ backgroundColor: "var(--manga-cream)", borderColor: "var(--manga-brown)" }}>
          <pre className="text-xs md:text-sm font-mono whitespace-pre-wrap break-words leading-relaxed" style={{ color: "var(--manga-black)" }}>
            {BIBTEX}
          </pre>
        </div>
      </div>
    </Section>
  );
}
