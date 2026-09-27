import { Section } from "./Section";
import styles from "./styles/Section.module.css";
import { COMMIT_CONDITION } from "../content";

export function CommitCondition() {
  return (
    <Section id="commit-condition" title="When a Change to the Gallery Pays"
      subtitle="The bottleneck is acceptance, not vision, and one comparison decides it.">
      <div className="manga-panel" style={{ padding: "2rem", maxWidth: "56rem", margin: "0 auto" }}>
        <div className={styles.prose}>
          <p>{COMMIT_CONDITION.intro}</p>
        </div>
        <p className={styles.equation} aria-label="Delta equals c times p-eff minus a-plus">
          &Delta; = <i>c</i>&thinsp;(<i>p</i><sub>eff</sub> &minus; <i>a</i><sup>+</sup>)
        </p>
        <div className={styles.prose}>
          <p>{COMMIT_CONDITION.outro}</p>
        </div>
      </div>
    </Section>
  );
}
