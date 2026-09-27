import Image from "next/image";
import { INSTITUTIONS } from "../content";

export function Team() {
  return (
    <section className="py-16 px-4 md:px-8 lg:px-12">
      <div className="max-w-4xl mx-auto grid grid-cols-1 sm:grid-cols-2 gap-8">
        {INSTITUTIONS.map((inst) => (
          <a
            key={inst.name}
            href={inst.url}
            target="_blank"
            rel="noopener noreferrer"
            className="block p-6 text-center hover:scale-105 transition-transform duration-300"
            style={{ border: "3px solid var(--manga-black)", borderRadius: "8px", backgroundColor: "var(--manga-cream)" }}
          >
            <Image src={inst.logo} alt={`${inst.name} logo`} width={400} height={400} className="w-24 h-24 object-contain mx-auto mb-4" />
            <h3 className="text-lg font-bold" style={{ color: "var(--manga-black)" }}>{inst.name}</h3>
          </a>
        ))}
      </div>
    </section>
  );
}
