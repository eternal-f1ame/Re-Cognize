"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import { Icon } from "./Icon";
import { CODE_URL, NAV } from "../content";

// Sticky bar with the section links; the section in view is underlined.
export function Nav() {
  const [active, setActive] = useState<string>("");

  useEffect(() => {
    const seen = new Map<string, number>();
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) seen.set(e.target.id, e.isIntersecting ? e.intersectionRatio : 0);
        const best = [...seen.entries()].sort((a, b) => b[1] - a[1])[0];
        if (best && best[1] > 0) setActive(best[0]);
      },
      { rootMargin: "-20% 0px -55% 0px", threshold: [0, 0.1, 0.25, 0.5] },
    );
    NAV.forEach(({ id }) => {
      const el = document.getElementById(id);
      if (el) io.observe(el);
    });
    return () => io.disconnect();
  }, []);

  return (
    <nav className="sticky top-0 z-40 h-[var(--nav-h)] border-b-2 border-[var(--line)] bg-[var(--paper)]/90 backdrop-blur">
      <div className="mx-auto flex h-full max-w-6xl items-center gap-6 px-4 sm:px-6">
        <a href="#top" className="flex shrink-0 items-center gap-2 font-display text-lg font-extrabold">
          <Image src="/comic-icons/icon6.png" alt="" width={28} height={28} className="h-7 w-7" />
          <span>Re<span className="text-[var(--orange)]">:</span>Cognize</span>
        </a>
        <ul className="hidden flex-1 items-center justify-center gap-5 text-sm font-medium lg:flex">
          {NAV.map(({ id, label }) => (
            <li key={id}>
              <a
                href={`#${id}`}
                className={`border-b-2 pb-0.5 transition-colors ${
                  active === id ? "border-[var(--orange)] text-[var(--ink)]" : "border-transparent text-[var(--muted)] hover:text-[var(--ink)]"
                }`}
              >
                {label}
              </a>
            </li>
          ))}
        </ul>
        <a href={CODE_URL} target="_blank" rel="noopener noreferrer" className="btn btn-primary ml-auto !py-1.5 !text-sm lg:ml-0">
          <Icon name="github" size={16} /> Code
        </a>
      </div>
    </nav>
  );
}
