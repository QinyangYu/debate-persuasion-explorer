# Debate Persuasion Explorer

Debate Persuasion Explorer is an AI-enabled web application for examining and predicting recorded stance change in Debate.org debates. The HW1 viewer remains the main interface; HW2 adds a trained PyTorch multi-layer perceptron, live inference, interactive retraining, and a real-time training monitor.

The application uses observational before/after agreement records. A changed record is an **observed stance transition**, not proof that a debate caused persuasion.

## HW2 prediction task

The primary roadmap task is binary classification for a voter-debate pair:

- `STAY` (`PRO -> PRO` or `CON -> CON`)
- `SWITCH` (`PRO -> CON` or `CON -> PRO`)

Given the recorded before stance, the app converts the binary prediction back into an interpretable predicted transition. It reports calibrated Stay/Switch probabilities and visibly states whether the prediction matches the recorded outcome.

### Model inputs

The 130-dimensional input follows the project roadmap's proposed baseline and text-model stages:

- **Initial stance:** one binary feature for the recorded PRO/CON stance before the debate
- **Debate structure:** category, number of rounds, PRO length, CON length, and argument-length balance
- **Debate text:** separate 32-dimensional signed feature-hash vectors for the concatenated PRO and CON arguments
- **Conventional user profile:** self-reported political ideology and religious ideology
- **User activity:** number of debates as a participant, number of votes cast, Debate.org Elo, and a profile-availability indicator

The exact dimensional breakdown is 9 normalized numeric features, 24 debate-category indicators, 16 political-ideology indicators, 17 religious-ideology indicators, and 64 argument-text hash features (32 per side), totaling 130. Each categorical group includes an `Other` fallback learned from the background training set.

The structured logistic-regression benchmark uses the initial-stance, debate-structure, and profile/activity features. The MLP adds the 64 argument-text features. This is an HW2-scale `M1`-style model: learned belief representations (`M2`) and the combined `M3` comparison remain later roadmap stages. Activity totals are described as overall observed activity, not chronological “prior” activity, because reliable temporal ordering has not been established.

The after stance is used only as the training target. Post-debate judging fields, including “Made more convincing arguments,” are excluded from the input. All voters from the same debate remain in one split so identical debate text cannot cross boundaries. In addition, every debate visible in the application is excluded from both training and validation.

## Neural network and training

The default model is a feed-forward MLP implemented in PyTorch:

```text
130 input features
  -> Linear(130, 64) -> Tanh -> Dropout(0.10)
  -> Linear(64, 2)
  -> Softmax Stay/Switch probabilities
```

Training uses inverse-frequency class-weighted cross-entropy, automatic backpropagation, and the Adam gradient-descent optimizer. Model selection uses validation PR-AUC, and the binary decision threshold is selected using validation balanced accuracy. The test set is never used for either choice. The saved artifact contains trained weights, feature normalization values, categorical vocabularies, network configuration, threshold, and evaluation metrics.

The 51-debate browser sample is the fixed final test set: 51 debates, 1,863 strict voter transitions, and 231 switchers. The model is fitted on a compact background cache made from every other DDO debate with a strict transition. A deterministic debate-level split places 22,156 background debates (54,938 voter-debate pairs) in training and 3,910 background debates (9,496 pairs) in validation. Thus a voter prediction shown anywhere in the browser is always a genuine test prediction.

### Default trained-model results

| Measure | Test result |
|---|---:|
| MLP accuracy | **70.48%** |
| MLP balanced accuracy | **68.66%** |
| MLP balanced error rate | **31.34%** |
| Switch precision | **24.48%** |
| Switch recall | **66.23%** |
| Switch F1 | **35.75%** |
| Switch PR-AUC | **34.86%** |
| Logistic baseline accuracy | 47.61% |
| Logistic baseline balanced accuracy | 57.46% |
| Logistic baseline balanced error rate | 42.54% |
| Always-Stay accuracy | 87.60% |
| Always-Stay Switch recall | 0.00% |

The MLP beats the roadmap's shallow structured logistic-regression benchmark on both raw and balanced accuracy. The always-Stay rule has higher raw accuracy because only 12.4% of the fixed test observations are switchers, but it finds zero switchers. The MLP correctly finds 153 of the 231 test switchers. Full results, the confusion matrix, per-class scores, class weights, decision threshold, and every epoch of the loss history are in [`outputs/training_metrics.json`](outputs/training_metrics.json).

## Application features

### HW1 foundation

- Search, category filtering, and debate sorting
- Interactive debate/debater/voter network
- Before/after toggle using real recorded stances
- Voter details, transition counts, and argument text by round
- Research annotations stored in SQLite or browser storage

### HW2 required features

- PyTorch MLP with one configurable hidden layer by default
- Forward propagation, cross-entropy loss, backpropagation, and Adam updates
- Saved trained weights at `backend/model/transition_mlp.pt`
- Training/validation loss history and train/validation/test metrics
- FastAPI prediction endpoint called directly from the React interface
- Per-voter Switch/Stay prediction, two class probabilities, and an implied transition

### Optional challenge 1 interactive training

The in-app training lab lets the user change the learning rate, number of epochs, hidden units, hidden layers, activation function (`ReLU`, `Tanh`, or `GELU`), and batch size. Starting a run launches training in a backend worker thread using only background train/validation data. A validation quality gate promotes the candidate only when its validation balanced accuracy, Switch F1, and Switch PR-AUC are all at least as strong as the active model; otherwise the experiment is displayed but the stronger saved model remains active.

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

Open [http://localhost:5173](http://localhost:5173). The header should show **SQLite connected**.

### Docker alternative

```powershell
docker compose up --build
```

Then open [http://localhost:5173](http://localhost:5173).

## Reproduce training

The committed compact background cache and 51-debate test set are sufficient to reproduce the default model:

```powershell
.\.venv\Scripts\python -m training.train `
  --epochs 150 `
  --learning-rate 0.001 `
  --hidden-units 64 `
  --hidden-layers 1 `
  --activation tanh `
  --batch-size 128
```

This rewrites `backend/model/transition_mlp.pt` (trained weights and preprocessing state) and `outputs/training_metrics.json` (curves, metrics, split sizes, baselines, and leakage controls).

To rebuild both the browser test set and the non-showcase training cache from the raw DDO files:

```powershell
.\.venv\Scripts\python .\scripts\preprocess_hw1.py --raw-dir "..\DDO_dataset\01_rawdata"
.\.venv\Scripts\python .\scripts\preprocess_training_data.py --raw-dir "..\DDO_dataset\01_rawdata"
```

The raw 1.23 GB `debates.json` and 244 MB `users.json` files are not committed.

## Test and validate

```powershell
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python .\scripts\validate_outputs.py
cd frontend
npm run build
```

The tests cover stance parsing, streaming JSON, leakage-safe group splits, feature dimensions, model output shape, saved-model inference, processed-data consistency, and the trained model's improvement over the majority benchmark.

## Project layout

```text
backend/                 FastAPI prediction, training, and annotation services
  model/                 Saved PyTorch weights
frontend/                React/Vite application and compact debate sample
training/                Feature pipeline, MLP definition, training, metrics
outputs/                 Data audit and training results
scripts/                 Raw-data preprocessing and validation
tests/                   Data and model pipeline tests
docs/                    Demo script and screenshots
```

## Data

The project uses the DDO dataset collected from Debate.org by Esin Durmus and Claire Cardie. The supplied files contain debate metadata, argument rounds, audience votes, and user profiles. The data were downloaded from the Figshare record [Dataset for “Neural embedding of beliefs reveals the role of relative dissonance in human decision-making”](https://doi.org/10.6084/m9.figshare.28326899), which credits the original DDO dataset.

The full preprocessing audit found 78,376 debates, 199,210 vote records, 66,297 strict before/after transitions, and 4,239 observed switchers. See [`outputs/audit_summary.json`](outputs/audit_summary.json).

## Limitations

- Debate.org users and voters are self-selected and are not representative of the public.
- Before/after agreement is self-reported and does not establish causal persuasion.
- The fixed browser test set intentionally emphasizes debates with useful transitions; its 12.4% switch rate is higher than the background corpus's 6.2% rate, so it is a demanding showcase test rather than a prevalence-representative random sample.
- Switch is rare, so accuracy alone is misleading; class-sensitive metrics and the confusion matrix must be reported.
- The model is a small HW2 demonstration, not a claim of deployable persuasion prediction.

## AI-assisted development disclosure

Codex was used to help design, implement, debug, test, and document the HW2 neural-network pipeline and web integration. The project owner is responsible for reviewing the code and being able to explain the model, forward pass, cross-entropy loss, backpropagation, optimizer updates, held-out-debate split, metrics, and leakage controls without AI assistance.
