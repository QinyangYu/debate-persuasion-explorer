# Homework 2 Three Minute Demo Script

Target length: 2 minutes 40 seconds to 2 minutes 55 seconds.

## 0:00 to 0:20 Application and task

"This is Debate Persuasion Explorer. In Homework 1 it visualized recorded voter positions before and after Debate.org debates. For Homework 2, I added a PyTorch neural network for the project's primary binary task: predicting whether a voter stays with or switches away from their initial stance. Only voters with a PRO or CON stance both before and after a debate are in scope; ties have no Stay/Switch outcome, so the app does not predict them."

Show the debate selector, network, and before/after toggle.

## 0:20 to 0:55 Model selection and rationale

"The model is a small multi-layer perceptron: 188 input features, one 128-unit Tanh hidden layer with dropout, and a two-unit Stay/Switch output. Besides the before stance, debate structure, PRO and CON argument-text features, and profile and activity information, it uses belief features. Every Debate.org user states a position on 48 big issues, so I measure how much the voter agrees with the debater they initially backed and with the opposing debater. Their difference is the relative dissonance idea from the dataset's source paper. Class-weighted cross-entropy handles the rare Switch class, and AdamW with weight decay limits overfitting. I also tried pretrained MiniLM sentence embeddings of the arguments, but they lowered validation performance, so the final model does not use them."

Show the training-lab architecture controls.

## 0:55 to 1:30 Performance

"All 51 debates in the browser are the fixed test set and never take part in fitting or model selection. Training and validation use 26,066 other debates, split at the debate level, and every model choice was made on validation. Because the showcase debates were selected for having switchers, the decision threshold is tuned on validation debates chosen by the same rule. On the 1,863 test voter-debate pairs, the MLP reaches 68.6 percent balanced accuracy and 40.2 percent Switch PR-AUC, compared with 64.3 percent balanced accuracy and 29.6 percent PR-AUC for the logistic-regression baseline. It finds 178 of 231 switchers, 77 percent recall. Raw accuracy is 62.3 percent, lower than always predicting Stay, because only 12.4 percent of voters switch: always-Stay scores 87.6 percent but finds no switchers, so balanced accuracy and PR-AUC are the meaningful measures."

Show the test metric cards and the loss chart.

## 1:30 to 2:05 Live prediction

Select a debate and click a voter node.

"Every selectable voter here is a test example. The frontend sends the selected debate and voter to FastAPI. The backend loads the saved PyTorch weights, rebuilds the same features used in training, performs a forward pass, and returns Stay and Switch probabilities. The app shows the predicted result and implied transition next to the observed transition. Expanding this section lists every input group, including how many big issues the voter answered, their belief agreement with each debater, and the relative dissonance. The after stance is excluded."

Point to the prediction panel and expand the model inputs. Click one tie voter to show that it is not predicted.

## 2:05 to 2:40 Both optional challenges

"I implemented both optional challenges. Interactive training lets me change the learning rate, epoch count, hidden units, hidden layers, activation function, batch size, dropout, and weight decay; the form starts from the active model's saved settings. The real-time monitor streams the current epoch, training and validation loss, and the final train, validation, and test metrics. I will start a short run now."

Set epochs to 10 and click **Start a new training run**. Show the progress indicator and moving curves.

Mention the validation quality gate: a weaker short run is shown as an experiment but cannot replace the stronger active model.

## 2:40 to 2:55 Close

"The complete pipeline is data, neural-network training, saved weights, FastAPI inference, and an interactive React prediction. The README documents setup, architecture, the feature and embedding comparison, evaluation, leakage controls, tests, and the AI-assistance disclosure."
