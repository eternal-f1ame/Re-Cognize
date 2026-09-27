import Image from "next/image";
import { Hero } from "../components/Hero";
import { Abstract } from "../components/Abstract";
import { Highlights } from "../components/Highlights";
import { Protocols } from "../components/Protocols";
import { CommitCondition } from "../components/CommitCondition";
import { ReCast } from "../components/ReCast";
import { Results } from "../components/Results";
import { Resources } from "../components/Resources";
import { Citation } from "../components/Citation";
import { Team } from "../components/Team";

// Floating comic icons behind the content: [left %, top %, icon, delay s, duration s]
const FLOATERS: [number, number, number, number, number][] = [
  [15, 20, 1, 0, 3], [85, 15, 2, 1, 4], [10, 70, 3, 2, 3.5], [90, 80, 4, 0.5, 4.5], [25, 85, 5, 1.5, 3],
  [75, 5, 6, 2.5, 4], [5, 45, 7, 0.8, 3.5], [95, 40, 1, 1.8, 3], [45, 10, 2, 2.2, 4.5], [65, 95, 3, 0.3, 3.5],
  [35, 25, 4, 1.2, 4], [80, 60, 5, 2.8, 3], [20, 50, 6, 0.7, 4.5], [55, 75, 7, 1.7, 3.5], [70, 30, 1, 2.3, 4],
];

export default function Home() {
  return (
    <div className="min-h-screen relative overflow-x-hidden" style={{ background: "var(--manga-cream)" }}>
      <div className="fixed inset-0 z-0 overflow-hidden" aria-hidden="true">
        <div className="absolute inset-0 opacity-15" style={{ backgroundColor: "var(--manga-brown)" }}></div>
        <div className="absolute inset-0 bg-grid-pattern opacity-30"></div>
        <div className="absolute inset-0 manga-action-lines opacity-5"></div>
        <div className="absolute inset-0 manga-vignette"></div>
        <div className="absolute inset-0">
          {FLOATERS.map(([left, top, icon, delay, duration], i) => (
            <div
              key={i}
              className="absolute opacity-30 animate-float"
              style={{ left: `${left}%`, top: `${top}%`, animationDelay: `${delay}s`, animationDuration: `${duration}s` }}
            >
              <Image
                src={`/comic-icons/icon${icon}.png`}
                alt=""
                width={48}
                height={48}
                className="w-8 h-8 md:w-10 md:h-10 lg:w-12 lg:h-12 object-contain"
                style={{ filter: "sepia(50%) saturate(80%) brightness(0.8) contrast(1.2)" }}
              />
            </div>
          ))}
        </div>
      </div>

      <main className="relative z-10">
        <Hero />
        <Abstract />
        <Highlights />
        <Protocols />
        <CommitCondition />
        <ReCast />
        <Results />
        <Resources />
        <Citation />
        <Team />
      </main>
    </div>
  );
}
