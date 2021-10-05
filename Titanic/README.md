 # predict titanic survivors

- DACON과 다르게 Kaggle에서는 예측값을 확률로 제출하면 0점이다. 0과 1로 예측하여 제출한 것만 정상적으로 채점된다. 
- DACON에서는 0.8113이 최고점수 였으나 Kaggle에서는 현재 VC(Voting Classifier)를 사용하여 제출한 0.75358점이 최고 점수이다.
- 06/oct/2021 LogisticRegression 사용하여 시도. GridSearchCV 사용하여 HyperParameteTuning 시도. RF, XGB, LR조합하여 VC시도. XGB와 LR로 VC하여 기록 경신. (0.7775)
