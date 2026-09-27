# Re:Cognize project page

A static Next.js page. Every word and link on it lives in `content.tsx`; the figures in
`public/figures/` are exported from the paper's PDFs (plots as SVG, figures with manga crops as
2x PNG).

The two animated loops (`components/replay/`) replay decisions computed by
`scripts/make_replay.py`: MagiV2's released encoder and the repository's own protocol rules on
eight crops of Bakuman chapter 1. From the repository root, with the dataset and the results
archive in place:

```bash
python project/scripts/make_replay.py   # writes project/data/replay.json and public/replay/*.webp
```

```bash
npm install
npm run dev        # http://localhost:3000
npm run build      # the production build Vercel runs
```

Deployment: import this repository in Vercel with Root Directory `project` and the Next.js
preset; every push to `main` then redeploys. The Paper button stays disabled until `PAPER_URL` in
`content.tsx` is set.
