# Re:Cognize project page

A static Next.js page. Every word and link on it lives in `content.tsx`; the figures in
`public/figures/` are exported from the paper's PDFs (plots as SVG, figures with manga crops as
2x PNG).

```bash
npm install
npm run dev        # http://localhost:3000
npm run build      # the production build Vercel runs
```

Deployment: import this repository in Vercel with Root Directory `project` and the Next.js
preset; every push to `main` then redeploys. The Paper button stays disabled until `PAPER_URL` in
`content.tsx` is set.
