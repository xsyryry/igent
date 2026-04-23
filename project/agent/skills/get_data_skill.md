# get_data_skill

This skill guides operator-controlled data collection for the local corpus.

Important rule: IELTS official scoring standards, band descriptors, rubrics,
and assessment criteria are not collected into RAG. Put scoring policy in
`writing_review_skill.md` instead.

## Collectable Categories

- `official_questions`: official public sample questions and practice tests.
- `lecture_notes`: legal public teaching notes and study materials.
- `news_corpus`: external reading corpus and magazine/news materials.

## Storage

Root: `D:\Afile\igent\data\`

- official questions: `official_questions\<module>`
- lecture notes: `lecture_notes`
- external corpus: `news_corpus`
- raw downloaded input: `raw`
- exports: `exports`

## Rules

1. Do not crawl or store scoring standards for RAG.
2. Question collection must extract structured question records, not save noisy webpages as final output.
3. Remove navigation, ads, comments, recommendations, and unrelated page text.
4. Use operator parameters for source, module, task type, time, and quantity.
5. If legal public questions cannot be found, store only official public samples/practice tests.
