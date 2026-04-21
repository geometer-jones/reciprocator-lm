## Greek Philosophy Classics Corpus

Repo-root public-domain Greek philosophy training corpus imported from the sibling
`memory-engine-llm` repository into `reciprocator` under `corpora/`.

- Source family: Project Gutenberg English translations
- Cleaning: Project Gutenberg header/footer removed upstream
- Bundled assets:
  - `greek_philosophy_classics_combined.txt`: combined training text
  - `sources.tsv`: author, ebook id, title, source URL, and original filename

Authors included:

- Plato
- Aristotle
- Xenophon

Included works:

- Plato: The Republic, Symposium, Phaedrus, Phaedo, Apology, Gorgias, Laws, Theaetetus, Timaeus, Meno, Euthyphro, Critias, Protagoras, Sophist
- Aristotle: The Ethics of Aristotle, Politics: A Treatise on Government, The Athenian Constitution
- Xenophon: The Memorabilia, The Apology, The Symposium, The Economist

Access from Python:

```python
from reciprocator import read_corpus_text

text = read_corpus_text("greek_philosophy_classics")
```
