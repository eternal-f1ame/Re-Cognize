import { Fragment } from "react";
import { Section } from "./Section";
import { Icon, type IconName } from "./Icon";
import { RESOURCES } from "../content";

export function Resources() {
  const [code, results, ...corpora] = RESOURCES;
  return (
    <Section id="resources" index={7} kicker="Code & data" title="Rerun it, or regenerate every table"
      lead="The harness, the per-tuple results and the three corpora. With the results archive, every table and figure of the paper regenerates without a GPU.">
      <div data-fit-row="" className="grid gap-4 md:grid-cols-2">
        {[code, results].map((r, i) => (
          <ResourceCard key={r.title} {...r} primary={i === 0} />
        ))}
      </div>
      <div className="mt-4 grid gap-4 md:grid-cols-3">
        {corpora.map((r) => (
          <ResourceCard key={r.title} {...r} />
        ))}
      </div>
    </Section>
  );
}

function ResourceCard({ icon, title, text, href, label, primary = false }: {
  icon: string;
  title: string;
  text: string;
  href: string;
  label: string;
  primary?: boolean;
}) {
  return (
    <a href={href} target="_blank" rel="noopener noreferrer"
      className={`panel lift group flex flex-col p-4 no-underline ${primary ? "bg-[var(--ink)] text-[var(--paper)] shadow-[4px_4px_0_var(--orange)] hover:shadow-[6px_7px_0_var(--orange)]" : ""}`}>
      <div className="flex items-center justify-between">
        <span className={`grid h-9 w-9 place-items-center rounded-full border-2 ${primary ? "border-[var(--paper)] bg-transparent" : "border-[var(--line)] bg-[var(--blue-soft)]"}`}>
          <Icon name={(icon === "code" ? "github" : icon) as IconName} size={16} />
        </span>
        <Icon name="external" size={18} className="opacity-60 transition-transform group-hover:-translate-y-0.5 group-hover:translate-x-0.5" />
      </div>
      <h3 className="mt-2.5 text-base font-bold">{title}</h3>
      <p className={`mt-1 flex-1 text-[0.8125rem] leading-relaxed ${primary ? "text-[var(--paper-2)]" : "text-[var(--ink-2)]"}`}>{text}</p>
      <span className={`mt-2.5 font-mono text-xs [overflow-wrap:anywhere] ${primary ? "text-[var(--orange-soft)]" : "text-[var(--orange)]"}`}>
        {label.split("/").map((part, i) => (
          <Fragment key={i}>{i > 0 && <>/<wbr /></>}{part}</Fragment>
        ))}
      </span>
    </a>
  );
}
