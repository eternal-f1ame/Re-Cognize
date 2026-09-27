import { Section } from "./Section";
import { Figure } from "./Figure";
import styles from "./styles/Section.module.css";
import { BINDING_CAPTION, BINDING_FIGURES, RECAST_CAPTION, RECAST_CHANGES, RECAST_FIGURES } from "../content";

export function ReCast() {
  return (
    <Section id="recast" title="Re:Cast"
      subtitle="A cast sheet of one running average per character, grown only where the page itself vouches for a crop, with nothing fitted on data.">
      <Figure images={RECAST_FIGURES} lead="Re:Cast" caption={RECAST_CAPTION} />
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 lg:gap-8" style={{ marginTop: "2.5rem" }}>
        {RECAST_CHANGES.map((c) => (
          <div key={c.title} className="manga-panel manga-universal-card">
            <div className="manga-card-icon" aria-hidden="true">{c.icon}</div>
            <h3 className="manga-card-title">{c.title}</h3>
            <p className={`manga-card-description ${styles.cardBody}`}>{c.text}</p>
          </div>
        ))}
      </div>

      <h3 className={styles.subheading}>Binding under chronological seeding</h3>
      <p className={styles.subtitle} style={{ marginBottom: "2rem" }}>
        When the seeds are each character&rsquo;s first appearances, binding supplies the reference: page groups
        merged in reading order and named by their first seed.
      </p>
      <Figure images={BINDING_FIGURES} lead="Binding, and pricing the binder." caption={BINDING_CAPTION} />
    </Section>
  );
}
