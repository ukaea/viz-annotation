# JOSS Paper

This folder contains the [Journal of Open Source Software](https://joss.theoj.org/) paper for this project (`paper.md`, `paper.bib`, and associated figures).

## Building

The paper is built locally using the [Open Journals `inara`](https://github.com/openjournals/inara) Docker image, which compiles `paper.md` into a PDF and JATS XML.

From this directory, run:

```bash
docker run --rm \
    --volume $PWD:/data \
    --user $(id -u):$(id -g) \
    --env JOURNAL=joss \
    openjournals/inara
```

This produces `paper.pdf` and `jats/paper.jats` (along with copies of the figures) in this folder.

## Files

- `paper.md` — paper source (Markdown with YAML front matter)
- `paper.bib` — bibliography
- `paper.pdf` — compiled output
- `jats/` — compiled JATS XML output
