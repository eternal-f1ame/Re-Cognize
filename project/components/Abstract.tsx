import { Section } from "./Section";
import styles from "./styles/Section.module.css";
import { ABSTRACT } from "../content";

export function Abstract() {
  const body = ABSTRACT.slice(0, -1);
  const scope = ABSTRACT[ABSTRACT.length - 1];
  return (
    <Section id="abstract" title="Abstract">
      <div className="manga-panel" style={{ padding: "2rem", marginBottom: "2rem" }}>
        <div className={styles.prose}>
          {body.map((p) => (
            <p key={p.slice(0, 32)}>{p}</p>
          ))}
        </div>
      </div>
      <div className="manga-speech-bubble" style={{ maxWidth: "56rem", margin: "0 auto" }}>
        <p style={{ margin: 0, lineHeight: 1.65 }}>{scope}</p>
      </div>
    </Section>
  );
}
