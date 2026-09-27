"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import styles from "./styles/Hero.module.css";
import { AFFILIATIONS, AUTHORS, CODE_URL, PAPER_URL, RESULTS_URL, TAGLINE, TEASER, VENUE } from "../content";

const buttonStyle = {
  padding: "0.75rem 1.5rem",
  fontSize: "0.875rem",
  fontWeight: 700,
  color: "var(--manga-black)",
  textDecoration: "none",
  transition: "transform 300ms",
  display: "inline-block",
} as const;

export function Hero() {
  const [isVisible, setIsVisible] = useState(false);
  const [showPaperNote, setShowPaperNote] = useState(false);

  useEffect(() => {
    setIsVisible(true);
  }, []);

  return (
    <section className={styles.hero}>
      <div className={`${styles.container} ${isVisible ? styles.containerVisible : styles.containerHidden}`}>
        <div className={styles.logoTitleContainer}>
          <div className={styles.logoContainer}>
            <div className={styles.logoWrapper}>
              <div className={styles.logoGlow}></div>
              <div className={styles.logoImage}>
                <Image src="/comic-icons/icon6.png" alt="" width={120} height={120} className={styles.logo} />
              </div>
            </div>
          </div>
          <h1 className={styles.title}>
            <span className={styles.titleMain}>Re:Cognize</span>
            <span className={styles.titleSub}>Open-Set Comic Character Re-Identification</span>
          </h1>
        </div>

        <div className={styles.conferenceInfo}>
          <p className={styles.conferenceTitle}>
            <a href={VENUE.url} target="_blank" rel="noopener noreferrer">{VENUE.name}</a>
          </p>
          <p className={styles.conferenceSubTitle}>
            <a href={VENUE.trackUrl} target="_blank" rel="noopener noreferrer">{VENUE.track}</a>
          </p>
          <p className={styles.conferenceDetails}>{VENUE.details}</p>
        </div>

        <div className={styles.authorsContainer}>
          <p className={styles.authorsMain}>
            {AUTHORS.map((a, i) => (
              <span key={a.name}>
                {a.url ? (
                  <a href={a.url} target="_blank" rel="noopener noreferrer" className={styles.authorName}>{a.name}</a>
                ) : (
                  <span className={styles.authorName}>{a.name}</span>
                )}
                <sup>{a.affiliations.join(",")}</sup>
                {i < AUTHORS.length - 1 ? ", " : ""}
              </span>
            ))}
          </p>
          <div className={styles.affiliations}>
            <p className={styles.affiliationsText}>
              {AFFILIATIONS.map((name, i) => (
                <span key={name}>
                  <sup>{i + 1}</sup>{name}{i < AFFILIATIONS.length - 1 ? "  ||  " : ""}
                </span>
              ))}
            </p>
          </div>
        </div>

        <div className={styles.buttonsContainer} style={{ marginBottom: "2.5rem" }}>
          <div className={styles.readPaperWrapper}>
            {PAPER_URL ? (
              <a href={PAPER_URL} target="_blank" rel="noopener noreferrer" className="manga-panel" style={buttonStyle}>
                📄 Paper
              </a>
            ) : (
              <span
                className={`manga-panel ${styles.buttonDisabled}`}
                style={buttonStyle}
                aria-disabled="true"
                tabIndex={0}
                onMouseEnter={() => setShowPaperNote(true)}
                onMouseLeave={() => setShowPaperNote(false)}
                onFocus={() => setShowPaperNote(true)}
                onBlur={() => setShowPaperNote(false)}
              >
                📄 Paper
              </span>
            )}
            {!PAPER_URL && showPaperNote && <div className={styles.comingSoonPopup}>arXiv link coming soon</div>}
          </div>
          <a href={CODE_URL} target="_blank" rel="noopener noreferrer" className="manga-panel" style={buttonStyle}>
            💻 Code
          </a>
          <a href={RESULTS_URL} target="_blank" rel="noopener noreferrer" className="manga-panel" style={buttonStyle}>
            📦 Results
          </a>
        </div>

        <div className="manga-panel" style={{ padding: "1rem", width: "100%", maxWidth: "80rem", margin: "0 auto 2rem" }}>
          <p className={styles.descriptionText}>{TAGLINE}</p>
          <div className={styles.imageContainer}>
            <Image
              src={TEASER.src}
              alt={TEASER.alt}
              width={TEASER.width}
              height={TEASER.height}
              sizes="(min-width: 1280px) 1200px, 100vw"
              className={styles.teaserImage}
              priority
            />
          </div>
          <p className={styles.teaserCaption}>{TEASER.caption}</p>
        </div>
      </div>
    </section>
  );
}
