# Three-minute demo plan

This outline is timed to the HW1 submission requirement. Record the live application rather than slides.

## 0:00-0:25 — Data and purpose

“This is Debate Persuasion Explorer. It uses the Debate.org DDO dataset: debate metadata and arguments plus voters' recorded agreement before and after each debate. This viewer is the foundation for a later model of individualized stance change; HW1 itself is descriptive and does not make causal claims.”

Show the title, dataset count in the footer, and debate list.

## 0:25-1:20 — Core viewer

1. Search for a topic or choose a category.
2. Select a debate with several switchers.
3. Point out PRO, CON, and the valid transition counts.
4. Toggle **Before** and **After** so several gold-ringed voter nodes change color.
5. Zoom with the wheel, drag to pan, and enable **Only switchers**.

## 1:20-2:05 — Inspection and text

1. Click a voter node.
2. Show its before/after transition and optional non-sensitive profile summary.
3. Scroll to the argument viewer.
4. Change rounds and explain that these texts will later be Transformer inputs.

## 2:05-2:45 — Extra-credit features

1. Verify the header says **SQLite connected**.
2. Select a node, choose **Notable switch**, add a short note, and save.
3. Refresh or change debates and return to demonstrate persistence.
4. Mention FastAPI + SQLite and the browser-storage fallback.

## 2:45-3:00 — Close

“The application therefore satisfies the interactive viewer objective and both extra-credit categories: graph-node annotation and backend/database integration. The next semester step is a leakage-aware prediction baseline before any deep model.”
