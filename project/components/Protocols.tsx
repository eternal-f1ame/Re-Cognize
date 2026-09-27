import { Section } from "./Section";
import { Figure } from "./Figure";
import styles from "./styles/Section.module.css";
import { PROTOCOLS, PROTOCOLS_FIGURE } from "../content";

export function Protocols() {
  return (
    <Section id="protocols" title="Four Protocols, One Stream"
      subtitle="Every protocol answers the same stream of query crops in reading order; they differ only in the gallery and whether it may change.">
      <Figure images={[PROTOCOLS_FIGURE]} lead="The four Re:Cognize protocols." caption={PROTOCOLS_FIGURE.caption} />
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 lg:gap-8" style={{ marginTop: "2.5rem" }}>
        {PROTOCOLS.map((p) => (
          <div key={p.id} className="manga-panel manga-universal-card">
            <span className={styles.protocolId}>{p.id}</span>
            <h3 className="manga-card-title">{p.name}</h3>
            <p className={`manga-card-description ${styles.cardBody}`}>{p.text}</p>
          </div>
        ))}
      </div>
    </Section>
  );
}
