## Plato Jowett Corpus

Repo-root public-domain Plato training corpus imported from the sibling
`memory-engine-llm` repository into `reciprocator` under `corpora/`.

- Source: Project Gutenberg texts in Benjamin Jowett translation
- Cleaning: Project Gutenberg header/footer removed upstream
- Bundled assets:
  - `plato_jowett_combined.txt`: combined training text
  - `sources.tsv`: ebook ids, titles, and source URLs

Included works:

- The Republic
- Symposium
- Phaedrus
- Phaedo
- Apology
- Gorgias
- Laws
- Theaetetus
- Timaeus
- Meno
- Euthyphro
- Critias
- Protagoras
- Sophist

Access from Python:

```python
from reciprocator import read_corpus_text

text = read_corpus_text("plato_jowett")
```
