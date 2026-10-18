// Everything the page says and links to. The wording follows the paper; edit it here, not in the components.

import type { ReactNode } from "react";

export const PAPER_URL = "https://arxiv.org/abs/2609.34032"; // arXiv link; the Paper button stays disabled while this is empty
export const CODE_URL = "https://github.com/eternal-f1ame/Re-Cognize";
export const RESULTS_URL = "https://github.com/eternal-f1ame/Re-Cognize/releases/tag/neurips-2026";

export const TITLE = "Re:Cognize - A Framework For Open-Set Comic Character Re-Identification";
export const DESCRIPTION =
  "Re:Cognize evaluates comic character re-identification as the story is read: four protocols on one query " +
  "stream, from closed-set retrieval to a cast the model must build and grow itself. NeurIPS 2026, Evaluations & " +
  "Datasets Track.";

export const VENUE = {
  label: "NeurIPS 2026 · Evaluations & Datasets Track",
  url: "https://neurips.cc/Conferences/2026",
};

// Author order and affiliation as in the paper; `mark` points to AUTHOR_NOTE.
export const AUTHORS: { name: string; url?: string; mark?: string }[] = [
  { name: "Aaditya Baranwal", url: "https://sochastic.me" },
  { name: "Madhav Kataria", mark: "*" },
  { name: "Yogesh S. Rawat" },
  { name: "Shruti Vyas" },
];

export const AFFILIATION = "Institute of Artificial Intelligence, University of Central Florida";

export const AUTHOR_NOTE = { mark: "*", text: "Work done as an intern at the University of Central Florida." };

export const INSTITUTIONS = [
  { unit: "Institute of Artificial Intelligence", name: "University of Central Florida", logo: "/UCF-logo.png", url: "https://ai.ucf.edu" },
];

export const NAV = [
  { id: "abstract", label: "Abstract" },
  { id: "findings", label: "Findings" },
  { id: "protocols", label: "Protocols" },
  { id: "commit-condition", label: "Commit condition" },
  { id: "vouch", label: "Vouch" },
  { id: "results", label: "Results" },
  { id: "resources", label: "Code & data" },
  { id: "citation", label: "Cite" },
];

export const TEASER = {
  src: "/figures/teaser.png",
  width: 2400,
  height: 800,
  alt: "A reader's question, closed-set Re-ID with a gallery built in advance, and Re:Cognize's four protocols on one query stream",
  lead: "Recognising characters while the story is read.",
  caption:
    "On the left, a reader's question such as where Takagi asks Mashiro depends on who appears on which page. In " +
    "the centre, standard Re-ID matches each query against a gallery built before reading starts, so a new " +
    "character finds no match. On the right, Re:Cognize streams crops in reading order against four galleries that " +
    "hold every character (P1), a few labelled examples of each (P2), none (P3) or a few that grow as it reads (P4), " +
    "and Vouch grows the gallery only where the page vouches for a crop.",
};

export const STATS = [
  { value: "4", label: "protocols on one query stream" },
  { value: "5", label: "Re-ID backbones evaluated" },
  { value: "3", label: "corpora: POPCharacters, Manga109, Re:Verse" },
  { value: "20+", label: "points of top-1 that correct labels would add" },
];

// The abstract of the paper, split at its natural breaks; the last entry is its scope statement.
export const ABSTRACT = [
  "Comics and manga tell stories through characters whose appearances change from panel to panel, yet readers " +
    "recognise them hundreds of pages later without ever consulting a cast list. Character re-identification models " +
    "rarely face the same challenge: existing evaluations give them a fixed set of identities before the story " +
    "begins. Re:Cognize instead asks models to recognise characters as the story unfolds and learn who is who along " +
    "the way.",
  "Across three manga corpora, we evaluate five Re-ID backbones and MeCha, a memory-augmented baseline we " +
    "introduce, and find that the central difficulty lies somewhere unexpected. A gallery with one example of each " +
    "character nearly matches the full one, but growing it remains difficult. When models grow it from their own " +
    "predictions, their performance deteriorates, even though correct identity labels would improve it by over " +
    "twenty points. This raises a simple question: when should a model trust its own match?",
  "We derive a single comparison that determines when gallery updates improve recognition and use it to design " +
    "Vouch, which updates its memory only when the page provides supporting evidence. Without fitting additional " +
    "parameters, Vouch recovers between one-third and two-thirds of the improvement available from perfect labels. " +
    "Re:Cognize measures whether a model can recognise characters as it reads; Vouch shows how to remember them " +
    "reliably.",
  "We focus our methodological claims on identity maintenance for previously encountered characters, while " +
    "separately evaluating new-character emergence as a diagnostic under a fixed reference rule to distinguish " +
    "recognition from discovery.",
];

export const HIGHLIGHTS = [
  {
    icon: "eye",
    title: "A small gallery nearly matches the full one",
    text: "One reference image per character already ranks as well as a gallery built in advance.",
  },
  {
    icon: "alert",
    title: "Growing it remains difficult",
    text:
      "A model that adds its own matches makes its cast worse, while the same growth with correct labels would " +
      "gain over twenty points of top-1 accuracy.",
  },
  {
    icon: "scale",
    title: "One comparison decides it",
    text:
      "An addition pays exactly when it is right more often than the cast already was on the queries it takes " +
      "over. The comparison has nothing to fit, and measured on half of a new corpus it calls the other half correctly " +
      "wherever the change matters.",
  },
  {
    icon: "cast",
    title: "Vouch narrows the gap",
    text:
      "One running average per character, grown only where the page itself vouches for a crop, recovers a third " +
      "to two thirds of what perfect labels would.",
  },
];

export const QUOTE = "Re:Cognize measures the gap. Vouch shows how to narrow it.";

export const PROTOCOLS_FIGURE: { src: string; width: number; height: number; alt: string; caption: ReactNode } = {
  src: "/figures/protocols.png",
  width: 2400,
  height: 800,
  alt: "The four Re:Cognize protocols on one query stream in reading order",
  caption: (
    <>
      All four answer one stream of query crops in reading order and differ in the gallery and whether it may
      change. <i>k</i> is the number of seed crops per character, <i>&tau;</i><sub>nov</sub> the novelty threshold of
      P3 and <i>B</i><sub>max</sub> the cap on crops P4 adds per character; frames are coloured by character and
      dashed where the model added the crop.
    </>
  ),
};

export const PROTOCOLS = [
  {
    id: "P1",
    tone: "blue",
    name: "Closed-set retrieval",
    text:
      "A fifth of each character's crops form the gallery and the rest are queries, ranked by cosine similarity " +
      "and scored by mAP and Rank-k. The closed-set ceiling.",
  },
  {
    id: "P2",
    tone: "blue",
    name: "Seeded static gallery",
    text:
      "k labelled crops per character, drawn at random (Seq-R) or as the character's first k appearances " +
      "(Seq-T), as a reader meets them; every other crop is a query.",
  },
  {
    id: "P3",
    tone: "orange",
    name: "Online clustering",
    text:
      "Crops arrive unlabelled into an empty gallery and join the nearest cluster above a novelty threshold or " +
      "open a new one. A diagnostic of emergence: one fixed threshold compares every representation.",
  },
  {
    id: "P4",
    tone: "orange",
    name: "Seeded gallery that grows",
    text:
      "P2's gallery, but in reading order each query is added under its top-1 match. Static, predicted and oracle " +
      "policies separate what growth gives, scored by identity Rank-1.",
  },
];

export const COMMIT_INTRO =
  "A change to the gallery can alter the answer only for a query whose nearest entry it supplied: the queries it " +
  "captures. Over the stream, the change in accuracy is exactly";

export const COMMIT_TERMS: { symbol: ReactNode; name: string; text: string }[] = [
  { symbol: <i>c</i>, name: "Capture rate", text: "The share of queries whose nearest entry the change supplied." },
  {
    symbol: <><i>p</i><sub>eff</sub></>,
    name: "Precision where it acts",
    text: "The share of captured queries whose capturing entry carries their own identity.",
  },
  {
    symbol: <><i>a</i><sup>+</sup></>,
    name: "The gallery it replaces",
    text: "The accuracy the unchanged gallery would have had on those same queries.",
  },
];

export const COMMIT_OUTRO: ReactNode = (
  <>
    So a change pays exactly when <i>p</i><sub>eff</sub> &gt; <i>a</i><sup>+</sup>: when its additions are right
    more often than the gallery already was on the queries they take over, not on average. A crop filed under the
    right character can still take over other characters&rsquo; queries, which is why neither term is the
    intuitive one.
  </>
);

export const VOUCH_FIGURES = [
  { src: "/figures/vouch_schematic_a.png", width: 2400, height: 800, alt: "Vouch: cast sheet, commitment and seed expansion" },
  { src: "/figures/vouch_schematic_b.png", width: 2400, height: 600, alt: "Over six pages the cast sheet is updated only on the pages that name the character" },
];

export const VOUCH_CAPTION =
  "on Re:Zero crops from Re:Verse, with Rom as identity A. Top: the three changes. A page names a character when " +
  "one of its crops is already committed to it, and commitment then adds that crop's page-group sibling. Bottom: " +
  "over six pages the cast sheet is updated only on the four that name Rom; elsewhere the rule abstains.";

export const VOUCH_CHANGES = [
  {
    label: "a",
    title: "A cast sheet, not a bag of crops",
    text:
      "Each character is one ℓ2-normalised average of the crops filed under it, so a new crop refines its " +
      "character's entry instead of becoming a rival entry that takes over other characters' queries.",
  },
  {
    label: "b",
    title: "Commitment under the page constraint",
    text:
      "A crop joins a character only when another crop in its page group is already committed to that character. " +
      "The rule consults only crops already read, and abstains wherever the page offers no evidence.",
  },
  {
    label: "c",
    title: "Seed expansion",
    text:
      "Before the stream starts, the crops on a seed's page that share its character join it: 1.43 per seed on " +
      "average, 90.6 % of them correct.",
  },
];

export const BINDING_FIGURES = [
  { src: "/figures/vouch_binding.png", width: 2400, height: 800, alt: "Binding: page groups merged in reading order and named by their first seed" },
  { src: "/figures/vouch_commit_flow.png", width: 2400, height: 600, alt: "Pricing the binder: crops are committed when the commit condition predicts a gain" },
];

export const BINDING_CAPTION =
  "Top, on Re:Verse's Re:Zero annotations: (a) a page's crops are grouped above one similarity threshold, (b) page " +
  "groups merge in reading order into the best earlier match above a second, or start new ones, and (c) a group's " +
  "first seed names it. Bottom: any frozen encoder can bind, with threshold τ; its crops are committed when the " +
  "commit condition predicts a positive Δ, and τ is chosen by that prediction, never by the measured gain.";

export const HEADROOM_PANELS = [
  { src: "/figures/ceiling_recovery.svg", label: "a", caption: "What one seed recovers, by metric" },
  { src: "/figures/p4_closes_gap.svg", label: "b", caption: "What correct growth adds, k = 1" },
  { src: "/figures/adaptation.svg", label: "c", caption: "Encoder-side adaptation" },
  { src: "/figures/one_breakeven.svg", label: "d", caption: "Every change to the gallery, k = 5" },
];

export const HEADROOM_CAPTION =
  "(a) One seed per character recovers the closed-set mAP but only part of its Rank-1. (b) Growth by the model's " +
  "own top-1 matches stays below a static gallery, while correct labels would add over twenty points (labels: " +
  "oracle minus static). (a, b): memory-block configuration, one random seed per character, three training runs. " +
  "(c) Encoder-side adaptation helps most on the backbones weakest in this domain. (d) Each point is one change to " +
  "the gallery on one backbone and corpus: adding under the page constraint, adding by top-1 match, or restricting " +
  "the candidate characters. The stronger the gallery already is, the less any change adds, and the commit " +
  "condition predicts which points fall below zero.";

export const RESOURCES = [
  {
    icon: "code",
    title: "Code",
    text: "The evaluation harness with all four protocols, Vouch, the memory-block baseline, and the analyses behind every number in the paper.",
    href: CODE_URL,
    label: "github.com/eternal-f1ame/Re-Cognize",
  },
  {
    icon: "package",
    title: "Results",
    text: "Per-tuple results behind every table and figure (23 MB). scripts/fetch_results.py downloads and verifies them; the tables then regenerate without a GPU.",
    href: RESULTS_URL,
    label: "recognize-results.tar.xz",
  },
  {
    icon: "database",
    title: "POPCharacters",
    text: "The public PopCharacters subset of PopManga: 23 series, with 4,058 test crops of 70 characters over 8 held-out series.",
    href: "https://huggingface.co/datasets/ragavsachdeva/popmanga_test",
    label: "huggingface.co/datasets/ragavsachdeva/popmanga_test",
  },
  {
    icon: "library",
    title: "Manga109",
    text: "27 held-out volumes (784 characters, 29,315 crops), evaluated zero-shot.",
    href: "https://manga109.github.io/manga109-project-website/en/index.html",
    label: "www.manga109.org",
  },
  {
    icon: "book",
    title: "Re:Verse",
    text: "The Re:Zero annotations behind the Vouch and binding figures.",
    href: "https://re-verse.vercel.app",
    label: "re-verse.vercel.app",
  },
];

export const BIBTEX = `@inproceedings{baranwal2026recognize,
  title     = {Re:Cognize - A Framework For Open-Set Comic Character Re-Identification},
  author    = {Baranwal, Aaditya and Kataria, Madhav and Rawat, Yogesh S. and Vyas, Shruti},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Evaluations and Datasets Track},
  year      = {2026},
  eprint    = {2609.34032},
  archivePrefix = {arXiv}
}`;
