# Homework 2 Submission Checklist

## Before recording

- Run the backend and frontend from the README instructions.
- Confirm the header says `SQLite connected`.
- Turn on `Only switchers`, click several gold-outlined voters, and verify that at least some predictions display `Matches the recorded Switch/Stay outcome`.
- Run one 10-epoch interactive training demonstration.
- Confirm the loss curves move and the run reaches `Training complete`.
- Keep the final video at or below three minutes.

## Video must show

- Which model was selected and why
- Test performance and limitations
- A prediction initiated from the graphical interface
- Interactive hyperparameter controls
- Real-time loss and metric monitoring

## Repository must contain

- Source code for the neural network and training loop
- Saved weights in `backend/model/transition_mlp.pt`
- Training results in `outputs/training_metrics.json`
- FastAPI inference and interactive-training endpoints
- React prediction and training-monitor interface
- README setup and feature documentation
- AI-assisted development disclosure

## Final checks

- Run `python -m unittest discover -s tests -v`.
- Run `python scripts/validate_outputs.py`.
- Run `npm run build` inside `frontend`.
- Push the repository and verify the submitted GitHub link opens.
- Upload the video to the Canvas assignment Dropbox.
