# Project overview

## Central question

How are users' prior positions and the content of a debate associated with whether their expressed position differs after the debate?

HW1 stays descriptive: **How are audience positions distributed before and after individual online debates?** The interface says “observed position change,” not “persuasion,” because the records do not establish causality.

## HW1 product

Debate Persuasion Explorer is a React viewer over a compact sample produced from the DDO raw data. A user can:

1. search, filter, sort, and select a debate;
2. view the PRO and CON participants and audience nodes;
3. toggle each voter's recorded stance before and after;
4. zoom and pan the network and filter to switchers;
5. click a voter to inspect the transition and non-sensitive profile metadata;
6. read the argument text by round;
7. label a voter and write a research note; and
8. persist annotations in SQLite through FastAPI (with browser storage as a fallback).

## Strict stance definition

For each agreement field, the parser inspects the PRO participant, CON participant, and `Tied` flag. A strict transition requires exactly one of PRO or CON both before and after. Ties, empty selections, and contradictory selections are kept as separate labels and excluded from the switch-rate denominator.

`switch = 1` only when a strict transition is `PRO -> CON` or `CON -> PRO`.

## Architecture

```text
DDO debates.json + users.json
            |
            v
streaming Python audit/preprocessing
            |
            +--> outputs/audit_summary.json
            |
            v
compact index + 51 debate detail JSON files
            |
            v
React/Vite interactive viewer
            |
            v
FastAPI --> SQLite node annotations
```

The 1.23 GB raw debate file is never loaded in the browser and is not committed to GitHub.

## Semester direction

- Baseline: logistic regression using pre-debate user activity and debate metadata.
- Text model: Transformer representations of PRO and CON argument text.
- Personalized model: combine debate text with a user-history representation.
- Advanced extension: add friendship/interactions through a graph neural network.

The future prediction target is whether a strict pre-debate PRO/CON position changes. Post-debate fields and current-debate outcomes must never be used as model inputs. Evaluation should address the observed 6.39% positive class rate and use debate-, user-, or time-aware splits to limit leakage.
