# Homework 2 Three Minute Demo Script

Target length: 2 minutes 40 seconds to 2 minutes 55 seconds.

## 0:00 to 0:25 Application and task

“This is Debate Persuasion Explorer. In Homework 1 it visualized recorded voter positions before and after Debate.org debates. For Homework 2, I added a PyTorch neural network that performs the project roadmap's primary binary task: predicting whether a voter stays with or switches away from their initial stance.”

Show the debate selector, network, and before/after toggle.

## 0:25 to 0:55 Model selection and rationale

“I used a small multi-layer perceptron with 130 input features, one 64-unit hidden layer, a Tanh activation, dropout, and a two-unit Stay/Switch output. The inputs include the before stance, debate structure, separate PRO and CON argument-text features, and conventional profile and activity information. The after stance, post-debate judging fields, and post-debate aggregate counts are excluded. Class-weighted cross-entropy addresses the rare Switch class.”

Show the training-lab architecture controls.

## 0:55 to 1:30 Performance

“The data boundary is visible and simple: all 51 debates in the browser are the fixed test set and never participate in fitting or model selection. Training and validation use 26,066 different background debates, split at the debate level. Across the 1,863 displayed test voter-debate pairs, the MLP reaches 70.48 percent accuracy and 68.66 percent balanced accuracy. It beats the structured logistic-regression baseline's 57.46 percent balanced accuracy. Because switching is uncommon, raw accuracy alone is misleading: the always-Stay rule finds no switchers, while the MLP finds 153 of 231, for 66.23 percent recall.”

Show the set-aside metric cards and live loss chart.

## 1:30 to 2:05 Live prediction

Select a debate and click a voter node.

“Every selectable voter here is a test example. The frontend sends the selected debate and voter to FastAPI. The backend loads the saved PyTorch weights, reconstructs the same features used in training, performs a forward pass, and returns Stay and Switch probabilities. Here the app displays the predicted result and implied transition alongside the observed transition. Expanding this section shows every input group and confirms that the after stance is excluded.”

Point to the prediction panel. Toggle before/after once to connect the prediction to the visualization.

## 2:05 to 2:40 Both optional challenges

“I implemented both optional challenges. First, interactive training lets me change the learning rate, epoch count, hidden units, hidden layers, activation function, and batch size. Second, the real-time monitor streams the current epoch, training and validation loss, and final train, validation, and test metrics. I will start a short run now.”

Set epochs to 10 and click **Start a new training run**. Show the progress indicator and moving curves. If time permits, show that predictions refresh after completion.

Mention the validation quality gate: a weaker short demonstration run is shown as an experiment but cannot overwrite the stronger committed model.

## 2:40 to 2:55 Close

“The complete pipeline is data to neural-network training to saved weights to FastAPI inference to an interactive React prediction. The README includes setup, architecture, evaluation, leakage controls, tests, and the AI-assistance disclosure.”

