What this notebook does (big picture)

It teaches an AI model (IndicBERT) to recognize the grammar role (Part-of-Speech) of Sanskrit words — e.g., is gacchati a verb? Is eṣa a pronoun? It uses real Sanskrit data from the Bhagavata Purana JSON files in this repo.

---
Cell-by-Cell Breakdown

Cell 0 — Title / Overview (markdown)

Just documentation. Explains the model being used (ai4bharat/indic-bert), the task (POS tagging), the 5 grammar categories being predicted (NOUN, VERB, PRON, PART, IND), and 4 improvements made over a first version.

Cell 1 — Section heading (markdown)

"Step 1 — Install dependencies"

Cell 2 — Install Python packages

!pip install transformers datasets accelerate indic-transliteration scikit-learn
Installs the libraries needed: HuggingFace for the AI model, indic-transliteration to convert IAST ↔ Devanagari script, scikit-learn for evaluation metrics.

Cell 3 — Instructions (markdown)

Tells the user to zip their json/ folder and upload it to Google Colab.

Cell 4 — Upload & unzip the JSON dataset

Prompts a file-upload dialog in Colab, unzips the uploaded file, and counts how many chapter JSON files were found.

Cell 5 — Section heading (markdown)

"Step 3a — Inspect raw lexical categories"

Cell 6 — Section heading only

"Step 3b — Load and parse data" (no code, just a label)

Cell 7 — Load & label the data (main data loader)

This is the core data preparation cell. It does several things:

1. Cleans verse text — strips verse numbers (॥१.२॥ style markers)
2. Converts IAST → Devanagari — e.g. dharma → धर्म
3. Maps grammar categories to POS labels — e.g. "substantive noun, masculine" → NOUN
4. Disambiguates ambiguous words using smart rules:
  - If the word's base form is in a known pronoun list (tad, eṣa, aham…) → PRON
  - If it's in a known indeclinable list (ca, eva, na…) → IND
  - If it ends in -tvā or -tum (Sanskrit verbal suffixes) → PART
  - If the grammar analyzer says it's a finite verb with person (1st/2nd/3rd) and there's no competing noun reading → VERB
  - Otherwise, trust the analyzer's first candidate
5. Builds a list of examples, each with: the word (Devanagari), the word (IAST), the full verse, and the POS label
6. Prints statistics — total examples, how many files were skipped, label distribution

Cell 8 — Duplicate of Cell 7

Identical data loader, kept so the notebook works regardless of which loader cell you run.

Cell 9 — Section heading (markdown)

"Step 4 — Explore the label distribution"

Cell 10 — Count labels & build mappings

Prints a table showing how many examples exist per label (e.g. NOUN: 13,984 / 77%). Also builds two lookup tables: label → number and number → label (needed by the model internally).

Cell 11 — Train/validation split

Shuffles all examples randomly (with a fixed seed for reproducibility), then puts 90% into training and 10% into validation.

Cell 12 — Section heading (markdown)

"Step 6 — Tokenise with verse context" — explains that instead of feeding just the word, the model now sees the full verse + the word together, which helps it disambiguate.

Cell 13 — Tokenize with verse context (improved version)

Loads the IndicBERT tokenizer. Converts each (verse, word) pair into numbers the model can read, using a 128-token window. The verse goes in as "sentence A" and the word as "sentence B" — the model can see context when classifying.

Cell 14 — Section heading (markdown)

"Step 6 (old version)" — tokenization without verse context.

Cell 15 — Tokenize without verse context (older version)

Same as Cell 13 but simpler: only feeds the word itself (no verse). Uses 32 tokens instead of 128. This is the v1 approach.

Cell 16 — Section heading (markdown)

"Step 8 — Define metrics" — explains macro F1 vs weighted F1.

Cell 17 — Define evaluation metrics

Defines compute_metrics which, given the model's predictions, calculates:
- Accuracy — overall % correct
- Weighted F1 — performance weighted by class size (inflated by NOUN)
- Macro F1 — honest metric, treats all classes equally (the real test)

Cell 18 — Section heading (markdown)

"Step 9 — Training with all four fixes" — explains class-weighted loss, label smoothing, and using macro F1 to pick the best checkpoint.

Cell 19 — Train the model (improved version)

The main training cell:
1. Computes class weights — rare classes (PRON: ~144 examples) get ~97× higher weight than common ones (NOUN: ~14k examples)
2. Defines WeightedTrainer — a custom training loop that applies those weights so the model is penalized harder for getting rare classes wrong
3. Sets hyperparameters — 5 epochs, batch size 32, learning rate 2e-5, label smoothing 0.1
4. Starts training — runs through the data 5 times, saving the checkpoint with the best macro F1

Cell 20 — Section heading (markdown)

"Step 9 (old version)" — training notes for the simpler v1 approach.

Cell 21 — Evaluate (improved version)

After training, runs evaluation on the validation set and prints:
- Accuracy, macro F1, weighted F1
- Comparison against v1 baseline numbers
- Per-class breakdown (how well the model did on each of the 5 POS categories)

Cell 22 — Section heading (markdown)

"Step 10 — Evaluate" (for old version)

Cell 23 — Evaluate (older version)

Same evaluation but simpler — just accuracy and weighted F1, no macro F1 or v1 comparison.

Cell 24 — Section heading (markdown)

"Step 12 — Inference" — explains the predict_pos function with and without context.

Cell 25 — Run inference (improved version)

Defines predict_pos(word, verse) that takes a Sanskrit word and optional verse, runs the trained model, and returns the predicted POS + confidence score. Then tests it on a handful of real examples like gacchati (expected: VERB), eṣa (expected: PRON), ca (expected: IND).

Cell 26 — Section heading (markdown)

"Step 12 (old version)" inference section.

Cell 27 — Run inference (older version)

Simpler predict_pos(word) that only takes the word, no context. Tests on a list of Sanskrit words.

Cell 28 — Section heading (markdown)

"Step 13 — Save the model"

Cell 29 — Save & download the model

Saves the trained model + tokenizer + label mappings to a folder, zips it, and downloads it to your local machine as indicbert_sanskrit_pos.zip.

---
Summary

┌─────────────────────────────────┬───────┐
│              Phase              │ Cells │
├─────────────────────────────────┼───────┤
│ Setup & upload data             │ 1–6   │
├─────────────────────────────────┼───────┤
│ Parse & label data              │ 7–8   │
├─────────────────────────────────┼───────┤
│ Explore labels, split train/val │ 9–11  │
├─────────────────────────────────┼───────┤
│ Tokenize for model input        │ 12–15 │
├─────────────────────────────────┼───────┤
│ Define metrics                  │ 16–17 │
├─────────────────────────────────┼───────┤
│ Train                           │ 18–19 │
├─────────────────────────────────┼───────┤
│ Evaluate                        │ 20–23 │
├─────────────────────────────────┼───────┤
│ Inference / test predictions    │ 24–27 │
├─────────────────────────────────┼───────┤
│ Save model                      │ 28–29 │
└─────────────────────────────────┴───────┘

The notebook has two versions of most steps — an improved v2 (with verse context, class weights, label smoothing) and a simpler v1 (word only, no reweighting). The v2 cells are the ones you should actually run.