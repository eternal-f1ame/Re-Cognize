import { Section } from "./Section";
import styles from "./styles/Section.module.css";
import { HIGHLIGHTS, QUOTE } from "../content";

export function Highlights() {
  return (
    <Section id="highlights" title="Key Findings" subtitle="Where models fail when the cast is assembled as the story is read">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 lg:gap-8">
        {HIGHLIGHTS.map((h) => (
          <div key={h.title} className="manga-panel manga-universal-card">
            <div className="manga-card-icon" aria-hidden="true">{h.icon}</div>
            <h3 className="manga-card-title">{h.title}</h3>
            <p className={`manga-card-description ${styles.cardBody}`}>{h.text}</p>
          </div>
        ))}
      </div>
      <div className="text-center" style={{ marginTop: "3.5rem" }}>
        <div className="manga-speech-bubble" style={{ display: "inline-block", maxWidth: "56rem" }}>
          <p style={{ margin: 0, fontSize: "1.125rem", fontStyle: "italic", fontWeight: 500 }}>&ldquo;{QUOTE}&rdquo;</p>
        </div>
      </div>
    </Section>
  );
}
