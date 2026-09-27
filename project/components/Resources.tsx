import { Section } from "./Section";
import styles from "./styles/Section.module.css";
import { RESOURCES } from "../content";

export function Resources() {
  return (
    <Section id="resources" title="Code, Results and Data"
      subtitle="Everything needed to rerun an evaluation, or to regenerate every table and figure of the paper without a GPU.">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 lg:gap-8">
        {RESOURCES.map((r) => (
          <a key={r.title} href={r.href} target="_blank" rel="noopener noreferrer"
            className="manga-panel manga-universal-card" style={{ textDecoration: "none", display: "block" }}>
            <div className="manga-card-icon" aria-hidden="true">{r.icon}</div>
            <h3 className="manga-card-title">{r.title}</h3>
            <p className={`manga-card-description ${styles.cardBody}`}>{r.text}</p>
            <span className={styles.linkLabel}>{r.label}</span>
          </a>
        ))}
      </div>
    </Section>
  );
}
