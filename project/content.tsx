// Everything the page says and links to. The wording follows the paper; edit it here, not in the components.

import type { ReactNode } from "react";

export const PAPER_URL = ""; // arXiv link; the Paper button stays disabled while this is empty
export const CODE_URL = "https://github.com/eternal-f1ame/Re-Cognize";
export const RESULTS_URL = "https://github.com/eternal-f1ame/Re-Cognize/releases/tag/neurips-2026";

export const TITLE = "Re:Cognize: Open-Set Comic Character Re-Identification";
export const TAGLINE = "Recognising characters while the story is read";
export const DESCRIPTION =
  "Re:Cognize evaluates comic character re-identification as the story is read: four protocols on one query " +
  "stream, from closed-set retrieval to a cast the model must build and grow itself. NeurIPS 2026, Evaluations & " +
  "Datasets Track.";

export const VENUE = {
  name: "Neural Information Processing Systems (NeurIPS 2026)",
  url: "https://neurips.cc/Conferences/2026",
  track: "Evaluations & Datasets Track",
  trackUrl: "https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets",
  details: "December 2026",
};

export const AUTHORS: { name: string; affiliations: number[]; url?: string }[] = [
  { name: "Aaditya Baranwal", affiliations: [1], url: "https://sochastic.me" },
  { name: "Madhav Kataria", affiliations: [1, 2] },
  { name: "Shruti Vyas", affiliations: [1] },
  { name: "Yogesh S. Rawat", affiliations: [1] },
];

export const AFFILIATIONS = ["University of Central Florida", "Indian Institute of Technology Jodhpur"];

export const INSTITUTIONS = [
  { name: "University of Central Florida", logo: "/UCF-logo.png", url: "https://www.ucf.edu" },
  { name: "Indian Institute of Technology Jodhpur", logo: "/IITJ-logo.png", url: "https://iitj.ac.in" },
];

export const TEASER = {
  src: "/figures/teaser.png",
  width: 2400,
  height: 800,
  alt: "A reader's question, closed-set Re-ID with a gallery built in advance, and Re:Cognize's four protocols on one query stream",
  caption:
    "Left: a reader's question, such as where Takagi asks Mashiro, depends on who appears on which page. Centre: " +
    "standard Re-ID matches each query against a gallery built before reading starts, so a new character finds no " +
    "match. Right: Re:Cognize streams crops in reading order against four galleries: every character (P1), a few " +
    "labelled examples of each (P2), none (P3), or a few that grow as it reads (P4). Re:Cast grows the gallery only " +
    "where additions are right more often than the gallery on the queries they take over.",
};

// The abstract of the paper, split at its natural breaks.
export const ABSTRACT = [
  "A manga reader meets a character on one page and knows them on sight a hundred pages later, without ever being " +
    "handed a cast list. Re-identifying comic characters demands the same, open-set and sequential: pages arrive as " +
    "a stream in reading order, new faces appear before anyone names them, and the cast is assembled as the story " +
    "is read. Re:Cognize evaluates recognition as the story is read, not against a cast handed over in advance: four " +
    "protocols on one query stream, from closed-set retrieval to a cast the model must build and grow itself.",
  "The surprise is where models fail. Recognising is close to solved: one reference image per character already " +
    "ranks as well as a gallery built in advance. Knowing what to believe is not: a model that adds its own matches " +
    "makes its cast worse, while the same growth with correct labels would gain over twenty points of top-1 " +
    "accuracy. The bottleneck is acceptance, not vision, and one comparison decides it: an addition pays exactly " +
    "when it is right more often than the cast already was on the queries it takes over. The comparison has nothing " +
    "to fit, and measured on half of a new corpus it calls the other half correctly.",
  "Re:Cast puts it to work with nothing fitted on data: a cast sheet of one running average per character, grown " +
    "only where the page itself vouches for a crop. It recovers a third to two thirds of what perfect labels would, " +
    "depending on whether the cast starts from random examples or from first appearances. Re:Cognize measures " +
    "whether a model can read along; Re:Cast is a cast that does.",
  "Our claims are on identity maintenance, recognising characters already met; the emergence of new ones is " +
    "measured as a diagnostic under a fixed reference rule, and we propose no method for it.",
];

export const HIGHLIGHTS = [
  {
    icon: "🎯",
    title: "Recognising is close to solved",
    text: "One reference image per character already ranks as well as a gallery built in advance.",
  },
  {
    icon: "🧭",
    title: "Knowing what to believe is not",
    text:
      "A model that adds its own matches makes its cast worse, while the same growth with correct labels would " +
      "gain over twenty points of top-1 accuracy.",
  },
  {
    icon: "⚖️",
    title: "One comparison decides it",
    text:
      "An addition pays exactly when it is right more often than the cast already was on the queries it takes " +
      "over. The comparison has nothing to fit, and measured on half of a new corpus it calls the other half correctly.",
  },
  {
    icon: "📇",
    title: "Re:Cast, a cast that reads along",
    text:
      "One running average per character, grown only where the page itself vouches for a crop, recovers a third " +
      "to two thirds of what perfect labels would.",
  },
];

export const QUOTE = "Re:Cognize measures whether a model can read along; Re:Cast is a cast that does.";

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
    name: "Closed-set retrieval",
    text:
      "A fifth of each character's crops form the gallery and the rest are queries, ranked by cosine similarity " +
      "and scored by mAP and Rank-k. P1 is the closed-set ceiling.",
  },
  {
    id: "P2",
    name: "Seeded static gallery",
    text:
      "k labelled crops per character, drawn at random (Seq-R) or as the character's first k appearances " +
      "(Seq-T), as a reader meets them; every other crop is a query.",
  },
  {
    id: "P3",
    name: "Online clustering, a diagnostic",
    text:
      "Crops arrive unlabelled into an empty gallery and join the nearest cluster above a novelty threshold or " +
      "open a new one. P3 is a measurement instrument, not a method: one fixed threshold compares every " +
      "representation under one criterion.",
  },
  {
    id: "P4",
    name: "Seeded gallery that grows",
    text:
      "P2's gallery, but in reading order each query is added under the character of its top-1 match. Static, " +
      "predicted and oracle policies separate what growth gives, scored by identity Rank-1.",
  },
];

export const COMMIT_CONDITION: { intro: ReactNode; outro: ReactNode } = {
  intro: (
    <>
      A change to the gallery can alter the answer only for a query whose nearest entry it supplied. With{" "}
      <i>c</i> the share of queries it captures, <i>p</i><sub>eff</sub> the share of captured queries whose capturing
      entry carries their own identity, and <i>a</i><sup>+</sup> the accuracy the unchanged gallery would have had on
      them, the change in accuracy is exactly
    </>
  ),
  outro: (
    <>
      so a change pays exactly when <i>p</i><sub>eff</sub> &gt; <i>a</i><sup>+</sup>. Both terms differ from the
      intuition, which compares how often the added crops are correct with the gallery&rsquo;s average accuracy: a
      crop filed under the right character can still take over other characters&rsquo; queries, and what it must beat
      is the gallery&rsquo;s accuracy on the queries it captures, not its average.
    </>
  ),
};

export const RECAST_FIGURES = [
  { src: "/figures/recast_schematic_a.png", width: 2400, height: 800, alt: "Re:Cast: cast sheet, commitment and seed expansion" },
  { src: "/figures/recast_schematic_b.png", width: 2400, height: 600, alt: "Over six pages the cast sheet is updated only on the pages that name the character" },
];

// Follows the bold figure name, as in the paper: "Re:Cast on Re:Zero crops ..."
export const RECAST_CAPTION =
  "on Re:Zero crops from Re:Verse, with Rom as identity A. Top: the three changes. A page names a " +
  "character when one of its crops is already committed to it, and commitment then adds that crop's page-group " +
  "sibling. Bottom: over six pages the cast sheet is updated only on the four that name Rom; elsewhere the rule " +
  "abstains.";

export const RECAST_CHANGES = [
  {
    icon: "📇",
    title: "A cast sheet instead of a bag of crops",
    text:
      "Each character is one ℓ2-normalised average of the crops filed under it, so a new crop refines its " +
      "character's entry instead of becoming a rival entry that takes over other characters' queries.",
  },
  {
    icon: "📄",
    title: "Commitment under the page constraint",
    text:
      "A crop joins a character only when another crop in its page group is already committed to that character. " +
      "The rule consults only crops already read, and abstains wherever the page offers no evidence.",
  },
  {
    icon: "🌱",
    title: "Seed expansion",
    text:
      "Before the stream starts, the crops on a seed's page that share its character join it: 1.43 per seed on " +
      "average, 90.6 % of them correct.",
  },
];

export const BINDING_FIGURES = [
  { src: "/figures/recast_binding.png", width: 2400, height: 800, alt: "Binding: page groups merged in reading order and named by their first seed" },
  { src: "/figures/recast_commit_flow.png", width: 2400, height: 600, alt: "Pricing the binder: crops are committed when the commit condition predicts a gain" },
];

export const BINDING_CAPTION =
  "Top, on Re:Verse's Re:Zero annotations: (a) a page's crops are grouped above one similarity threshold, (b) page " +
  "groups merge in reading order into the best earlier match above a second, or start new ones, and (c) a group's " +
  "first seed names it. Bottom: any frozen encoder can bind, with threshold τ; its crops are committed when the " +
  "commit condition predicts a positive Δ, and τ is chosen by that prediction, never by the measured gain.";

export const HEADROOM_PANELS = [
  { src: "/figures/ceiling_recovery.svg", label: "(a)", caption: "What one seed recovers, by metric." },
  { src: "/figures/p4_closes_gap.svg", label: "(b)", caption: "What correct growth would add, k = 1." },
  { src: "/figures/adaptation.svg", label: "(c)", caption: "Encoder-side adaptation, three training runs." },
  { src: "/figures/one_breakeven.svg", label: "(d)", caption: "Every gallery operation at k = 5, against the gallery it starts from." },
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
    icon: "💻",
    title: "Code",
    text: "The evaluation harness with all four protocols, Re:Cast, the memory-block baseline, and the analyses behind every number in the paper.",
    href: CODE_URL,
    label: "github.com/eternal-f1ame/Re-Cognize",
  },
  {
    icon: "📦",
    title: "Results",
    text: "Per-tuple results behind every table and figure (23 MB). scripts/fetch_results.py downloads and verifies them; the tables then regenerate without a GPU.",
    href: RESULTS_URL,
    label: "recognize-results.tar.xz",
  },
  {
    icon: "📚",
    title: "POPCharacters",
    text: "The public PopCharacters subset of PopManga: 23 series, 4,058 test crops of 70 characters over 8 held-out series.",
    href: "https://huggingface.co/datasets/ragavsachdeva/popmanga_test",
    label: "huggingface.co/datasets/ragavsachdeva/popmanga_test",
  },
  {
    icon: "🗾",
    title: "Manga109",
    text: "27 held-out volumes (784 characters, 29,315 crops), evaluated zero-shot.",
    href: "https://manga109.github.io/manga109-project-website/en/index.html",
    label: "manga109.org",
  },
  {
    icon: "📖",
    title: "Re:Verse",
    text: "The Re:Zero annotations behind the Re:Cast and binding figures.",
    href: "https://re-verse.vercel.app",
    label: "re-verse.vercel.app",
  },
];

export const BIBTEX = `@inproceedings{baranwal2026recognize,
  title     = {Re:Cognize: Open-Set Comic Character Re-Identification},
  author    = {Baranwal, Aaditya and Kataria, Madhav and Vyas, Shruti and Rawat, Yogesh S.},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Evaluations and Datasets Track},
  year      = {2026}
}`;
