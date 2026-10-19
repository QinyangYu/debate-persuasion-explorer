# Homework 2 Submission Checklist

## Before recording

- Run the backend and frontend from the README instructions (on Windows PowerShell, use `npm.cmd run dev` if script execution is blocked).
- Confirm the header says `SQLite connected`.
- Confirm the training-lab form shows the active model's settings: 60 epochs, 128 hidden units, 1 layer, Tanh, batch size 128, dropout 0.3, weight decay 0.01.
- Confirm the test cards show 62.3% accuracy, 68.6% balanced accuracy, 33.6% Switch F1, 40.2% Switch PR-AUC, and a 64.3% logistic baseline.
- Turn on `Only switchers`, click several gold-outlined voters, and verify that at least some predictions display `Matches the recorded Switch/Stay outcome`.
- Expand `Model inputs used for this prediction` and check that big-issue coverage, belief agreement with each debater, and relative dissonance appear.
- Click a tie voter and confirm the panel says it is not predicted.
- Run one 10-epoch interactive training demonstration.
- Confirm the loss curves move and the run reaches `Training complete`.
- Keep the final video at or below three minutes.

## Video must show

- Which model was selected and why, including the belief features
- Test performance and limitations, including why raw accuracy is misleading
- A prediction initiated from the graphical interface
- Interactive hyperparameter controls
- Real-time loss and metric monitoring

## Repository must contain

- Source code for the neural network and training loop
- Saved weights in `backend/model/transition_mlp.pt`
- Training results in `outputs/training_metrics.json`
- Training cache in `training/data/` (debates, profiles with big-issue stances, manifest)
- FastAPI inference and interactive-training endpoints
- React prediction and training-monitor interface
- README setup, feature documentation, and the model comparison table
- AI-assisted development disclosure

## Final checks

- Run `python -m unittest discover -s tests -v` (10 tests).
- Run `python scripts/validate_outputs.py`.
- Run `npm run build` inside `frontend`.
- Push the repository and verify the submitted GitHub link opens.
- Upload the video to the Canvas assignment Dropbox.
