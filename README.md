# Debate Persuasion Explorer

Debate Persuasion Explorer is an AI-enabled web application for examining and predicting recorded stance change in Debate.org debates. The HW1 viewer remains the main interface; HW2 adds a trained PyTorch multi-layer perceptron, live inference, interactive retraining, and a real-time training monitor. The model combines debate structure, argument text, voter profiles, and **belief features**: how closely the voter's stances on Debate.org's 48 "big issues" match each debater's.

The application uses observational before/after agreement records. A changed record is an **observed stance transition**, not proof that a debate caused persuasion.

## HW2 prediction task

The primary roadmap task is binary classification for a voter-debate pair:

- `STAY` (`PRO -> PRO` or `CON -> CON`)
- `SWITCH` (`PRO -> CON` or `CON -> PRO`)

Only voters whose recorded stance is PRO or CON both before and after the debate are in scope. Voters with a tie or unknown stance on either side have no Stay/Switch outcome, so the app shows no prediction for them. Given the recorded before stance, the app converts the binary prediction back into an interpretable predicted transition, reports Stay/Switch probabilities, and visibly states whether the prediction matches the recorded outcome.

### Model inputs

The model receives 188 input features:

- **Initial stance:** one binary feature for the recorded PRO/CON stance before the debate
- **Debate structure:** category, number of rounds, PRO length, CON length, and argument-length balance
- **Debate text:** separate 32-dimensional signed feature-hash vectors for the concatenated PRO and CON arguments
- **Conventional user profile:** self-reported political ideology and religious ideology
- **User activity:** number of debates as a participant, number of votes cast, Debate.org Elo, and a profile-availability indicator
- **Big-issue beliefs:** the voter's stance on each of the 48 big issues (+1 Pro, −1 Con, 0 undecided or unanswered)
- **Voter–debater belief similarity:** agreement with the debater the voter initially backed and with the opposing debater (mean agreement over issues both answered, plus the share of issues they share), the voter's issue coverage, and **relative dissonance** = agreement with the opposing debater − agreement with the initially backed debater
- **Debater strength:** Elo and number of debates for the initially backed and the opposing debater

The exact breakdown is 13 normalized numeric features, 24 debate-category indicators, 16 political-ideology indicators, 17 religious-ideology indicators, 48 big-issue stances, 6 belief-similarity features, and 64 argument-text hash features (32 per side). Each categorical group includes an `Other` fallback learned from the training set.

The belief features follow the idea behind the dataset's source paper, which links stance change to *relative dissonance* between a person's beliefs and the two sides. This is still an HW2-scale `M1`-style model with hand-built belief features; learned belief embeddings (`M2`) and the combined `M3` comparison remain later roadmap stages.

The structured logistic-regression benchmark uses every input except the argument-text features; the MLP adds the 64 argument-text features. The after stance is used only as the training target. Post-debate judging fields, including "Made more convincing arguments," are excluded from the input. All voters from the same debate remain in one split so identical debate text cannot cross boundaries. In addition, every debate visible in the application is excluded from both training and validation.

## Neural network and training

The active model is a feed-forward MLP implemented in PyTorch (24,450 trainable parameters):

```text
188 input features
  -> Linear(188, 128) -> Tanh -> Dropout(0.30)
  -> Linear(128, 2)
  -> Softmax Stay/Switch probabilities
```

It was trained with learning rate 0.001, 60 epochs, batch size 128, and weight decay 0.01. These hyperparameters are saved together with the weights, so the active model is reproducible and self-describing.

Training uses inverse-frequency class-weighted cross-entropy, automatic backpropagation, and the AdamW gradient-descent optimizer (Adam with decoupled weight decay). The epoch with the best validation Switch PR-AUC is kept. The test set is never used for any choice.

**Decision threshold.** The 51 showcase debates were chosen among debates with at least five strict voters and at least one switcher, which doubles their switch rate (12.4% vs. 6.2%). The decision threshold is therefore tuned for balanced accuracy on the 195 validation debates (1,710 voter-debate pairs) that pass the same selection rule. This uses only validation labels and lets the threshold transfer to the showcase test set. The selected threshold is 0.435. The logistic baseline is calibrated the same way.

The saved artifact contains trained weights, feature normalization values, categorical vocabularies, network configuration, threshold, and evaluation metrics.

The 51-debate browser sample is the fixed final test set: 51 debates, 1,863 strict voter transitions, and 231 switchers. The model is fitted on a compact background cache made from every other DDO debate with a strict transition. A deterministic debate-level split places 22,156 background debates (54,938 voter-debate pairs) in training and 3,910 background debates (9,496 pairs) in validation. Thus a voter prediction shown anywhere in the browser is always a genuine test prediction.

### Active-model results

**Switch PR-AUC is the primary metric.** It measures how well the model ranks true switchers above stayers, independent of the decision threshold; a random ranking scores the switch rate (12.4% on test). It is also the model-selection criterion on validation, and the app shows it first.

| Measure | Validation | Test |
|---|---:|---:|
| MLP accuracy | 74.28% | **62.27%** |
| MLP balanced accuracy | 71.42% | **68.61%** |
| MLP balanced error rate | 28.58% | **31.39%** |
| Switch precision | | **21.50%** |
| Switch recall | | **77.06%** |
| Switch F1 | 25.46% | **33.62%** |
| Switch PR-AUC | 30.14% | **40.21%** |
| Logistic baseline accuracy | | 51.10% |
| Logistic baseline balanced accuracy | | 64.29% |
| Logistic baseline balanced error rate | | 35.71% |
| Logistic baseline Switch PR-AUC | 19.59% | 29.56% |
| Always-Stay accuracy | | 87.60% |
| Always-Stay Switch recall | | 0.00% |

The MLP beats the shallow logistic-regression benchmark on balanced error rate (31.39% vs. 35.71%), raw accuracy, and PR-AUC. The always-Stay rule has higher raw accuracy because only 12.4% of the test observations are switchers, but it finds zero switchers. The MLP finds 178 of the 231 test switchers; the cost is 650 Stay voters flagged as likely switchers, which is why raw accuracy is lower than balanced accuracy. Full results, the confusion matrix, per-class scores, class weights, decision threshold, and every epoch of the loss history are in [`outputs/training_metrics.json`](outputs/training_metrics.json).

### What changed, and what was tried

All variants below use the same splits. Variants were compared on **validation** Switch PR-AUC only; test numbers were not used to choose.

| Variant | Validation PR-AUC | Test PR-AUC | Test balanced accuracy |
|---|---:|---:|---:|
| Previous model: no belief features, 1×128 Tanh, dropout 0.1, 150 epochs | 0.208 | 0.372 | 68.74% |
| + belief features (same training) | 0.281 | 0.373 | 66.58%¹ |
| + dropout 0.3, weight decay 0.01, 60 epochs (**active model**) | **0.301** | **0.402** | 68.61% |
| + 16-d MiniLM sentence embeddings per side | 0.290 | 0.388 | 70.45% |
| + 32-d MiniLM sentence embeddings per side | 0.257 | 0.373 | 69.90% |
| + 64-d MiniLM sentence embeddings per side | 0.239 | 0.367 | 68.70% |

¹ Before the showcase-rule threshold calibration was added; the threshold tuned on all validation debates was too low for the showcase test set.

The belief features gave the largest gain (validation PR-AUC +45% relative), and regularization added more. Pretrained sentence embeddings of the argument text lowered validation PR-AUC, and more dimensions hurt more: every voter in a debate shares the same text, so richer text features let the network memorize debates instead of generalizing. They are kept as an optional, documented experiment (see below) but are off in the active model.

## Application features

### HW1 foundation

- Search, category filtering, and debate sorting
- Interactive debate/debater/voter network
- Before/after toggle using real recorded stances
- Voter details, transition counts, and argument text by round
- Research annotations stored in SQLite or browser storage

### HW2 required features

- PyTorch MLP with one 128-unit hidden layer in the active model (layers and units are configurable)
- Forward propagation, cross-entropy loss, backpropagation, and AdamW updates
- Saved trained weights at `backend/model/transition_mlp.pt`
- Training/validation loss history and train/validation/test metrics
- FastAPI prediction endpoint called directly from the React interface
- Per-voter Switch/Stay prediction, two class probabilities, and an implied transition
- A "Model inputs used for this prediction" panel listing every input group for the selected voter, including big-issue coverage, belief agreement with each debater, and relative dissonance

### Optional challenge 1 interactive training

The in-app training lab lets the user change the learning rate, number of epochs, hidden units, hidden layers, activation function (`ReLU`, `Tanh`, or `GELU`), batch size, dropout, and weight decay. Starting a run launches training in a backend worker thread using only background train/validation data. A validation quality gate promotes the candidate only when its validation balanced accuracy, Switch F1, and Switch PR-AUC are all at least as strong as the active model; otherwise the experiment is displayed but the stronger saved model remains active.

The saved model file stores the hyperparameters it was trained with. When the page loads, the hyperparameter form is filled from the active model, so after a promoted run and a page refresh the form shows the settings of the best model rather than fixed defaults.

### Optional challenge 2 real-time training monitor

While training runs, the interface displays the current epoch, live training and validation loss curves, test accuracy, balanced accuracy, Switch F1, Switch PR-AUC, the logistic-regression comparison, and updated prediction results after the new model becomes active.

## Run locally

Prerequisites: Node.js 20+ and Python 3.10+.

### 1. Create the Python environment

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.10,<3.0"
.\.venv\Scripts\python -m pip install -r .\backend\requirements.txt
```

### 2. Start the FastAPI backend

```powershell
.\.venv\Scripts\python -m uvicorn backend.app:app --reload --port 8000
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to inspect the model, prediction, training, and annotation APIs.

### 3. Start the React frontend

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

On Windows PowerShell, if script execution is blocked, use `npm.cmd install` and `npm.cmd run dev`.

Open [http://localhost:5173](http://localhost:5173). The header should show **SQLite connected**.

### Docker alternative

```powershell
docker compose up --build
```

Then open [http://localhost:5173](http://localhost:5173).

## Reproduce training

The committed compact background cache and 51-debate test set are sufficient to reproduce the active model; the default arguments are the active model's hyperparameters:

```powershell
.\.venv\Scripts\python -m training.train
```

This rewrites `backend/model/transition_mlp.pt` (trained weights and preprocessing state) and `outputs/training_metrics.json` (curves, metrics, split sizes, baselines, and leakage controls). Other settings can be passed explicitly, for example `--hidden-units 64 --hidden-layers 2 --activation relu --dropout 0.3 --weight-decay 0.01 --epochs 60`.

To rebuild the training cache and refresh the test-set profiles (including big-issue stances) from the raw DDO files:

```powershell
.\.venv\Scripts\python .\scripts\preprocess_training_data.py --raw-dir "..\DDO_dataset\01_rawdata"
.\.venv\Scripts\python .\scripts\preprocess_hw1.py --raw-dir "..\DDO_dataset\01_rawdata" --profiles-only
```

Running `preprocess_hw1.py` without `--profiles-only` re-runs the full audit and re-selects the showcase sample. The raw 1.23 GB `debates.json` and 244 MB `users.json` files are not committed.

### Optional: sentence-embedding text features

The MiniLM experiment in the comparison table can be reproduced. It needs `sentence-transformers` only during preprocessing; the backend never loads the language model.

```powershell
.\.venv\Scripts\python -m pip install -r .\training\requirements-embeddings.txt
.\.venv\Scripts\python .\scripts\preprocess_training_data.py --raw-dir "..\DDO_dataset\01_rawdata" --text-output tmp\background_text.jsonl.gz
.\.venv\Scripts\python -m training.embeddings encode --text tmp\background_text.jsonl.gz
.\.venv\Scripts\python -m training.embeddings reduce --dimensions 16
.\.venv\Scripts\python -m training.train --text-embeddings --model-path tmp\embedding_model.pt --metrics-path tmp\embedding_metrics.json
```

Encoding about 52,000 PRO/CON texts takes roughly 40 minutes on a CPU. The PCA projection is fitted on training-split debates only.

## Test and validate

```powershell
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python .\scripts\validate_outputs.py
cd frontend
npm run build
```

The tests cover stance parsing, streaming JSON, leakage-safe group splits, feature dimensions, model output shape, big-issue encoding, relative dissonance, the showcase-rule calibration subset, saved-model inference, processed-data consistency, and the trained model's improvement over the logistic benchmark.

## Project layout

```text
backend/                 FastAPI prediction, training, and annotation services
  model/                 Saved PyTorch weights
frontend/                React/Vite application and compact debate sample
training/                Feature pipeline, MLP definition, training, metrics, optional embeddings
outputs/                 Data audit and training results
scripts/                 Raw-data preprocessing and validation
tests/                   Data and model pipeline tests
docs/                    Demo script and screenshots
```

## Data

The project uses the DDO dataset collected from Debate.org by Esin Durmus and Claire Cardie. The supplied files contain debate metadata, argument rounds, audience votes, and user profiles, including each user's stance on 48 big issues. The data were downloaded from the Figshare record [Dataset for "Neural embedding of beliefs reveals the role of relative dissonance in human decision-making"](https://doi.org/10.6084/m9.figshare.28326899), which credits the original DDO dataset.

The full preprocessing audit found 78,376 debates, 199,210 vote records, 66,297 strict before/after transitions, and 4,239 observed switchers. See [`outputs/audit_summary.json`](outputs/audit_summary.json).

## Limitations

- Debate.org users and voters are self-selected and are not representative of the public.
- Before/after agreement is self-reported and does not establish causal persuasion.
- Profiles, including big-issue stances, Elo, and activity totals, are snapshots from when the data were collected, not from the time of each debate. They may postdate the vote, so belief features describe a user's overall observed beliefs rather than beliefs held before the debate.
- The fixed browser test set was selected among debates with at least five strict voters and at least one switcher; its 12.4% switch rate is double the background corpus's 6.2%, so it is a demanding showcase test rather than a prevalence-representative random sample.
- With 231 switchers clustered in 51 debates, test metrics are noisy; the 3,910-debate validation set gives more stable comparisons.
- Switch is rare, so accuracy alone is misleading; class-sensitive metrics and the confusion matrix must be reported.
- The model is a small HW2 demonstration, not a claim of deployable persuasion prediction.

## AI-assisted development disclosure

Codex and Claude Code were used to help design, implement, debug, test, and document the neural-network pipeline, the belief features, the embedding experiment, and the web integration. The project owner is responsible for reviewing the code and being able to explain the model, forward pass, cross-entropy loss, backpropagation, optimizer updates, held-out-debate split, threshold calibration, metrics, and leakage controls without AI assistance.
