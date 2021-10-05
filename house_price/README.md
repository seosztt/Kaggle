# House Prices - Advanced Regression Techniques

## EDA / Data Preprocessing

### Feature classification
- 80개의 Feature들을 아래 기준에 따라 6가지 카테고리로 분류.
- 분류 후, 각 카테고리별로 알맞는 방법으로 전처리를 할 예정.
- 결측치 / 이상치(Outlier)처리 방법. 특히 이상치를 어떻게 처리할지에 대한 고민.

#### Numeric data (숫자형 데이터)
- numeric(raw) : real valued data (ex. area, length...)
  - Scaling을 해주고 넣어 줄 예정
  - 넓이 데이터들의 경우 겹치는 데이터 / 서로 depend하는 데이터가 많은데 처리 할 방법 고민.

- discrete : discrete values (ex. number of rooms)
  - 년도 등의 값은 scaling을 해주고 정수 값으로 넣어 줄 예정
  - 아래 순서를 가진 범주형 데이터와 마찬가지로 one-hot을 하는게 더 좋을지?

#### Categorical data (범주형 데이터)
- feature_map : 순서를 가진 데이터 (Ex. Quality : Excellent->Good->Fair->Poor)
  - 순서 혹은 가격과의 상관관계를 보고 0,1,2,3,4 등 정수값으로 매핑해 줄 예정
  - 그냥 one-hot을 하는게 성능이 더 좋을지? 에 대한 고민.

- feature_onehot : 데이터간의 순서/관계를 찾기 힘든 데이터 (ex. SaleType)
  - One-hot encoding을 할 예정
  - 이때 개수가 적은 값들의 경우 drop. (one-hot column에 1의 값이 아주 적은 경우)

#### Others
- extra : data that need discussion
  - 어느 분류에 들어갈지 모호하거나, 같은 분류내의 데이터들과 다른 방식의 전처리가 필요하다고 생각되는 feature

- delete : need to delete
  - 값의 개수가 적거나 / 결측치가 많거나 / 가격과의 상관관계가 매우 낮거나 / 등등의 이유로 지워 줄 feature

## Modeling

### Model
We will use Scikit-learn (maybe)
- RandomForestRegressor
  - 첫 시도에 0.1670 기록
  - feature_importance가 적은 feature drop / 가격에 log함수를 써서 skewness를 줄여주는 방법을 통해 0.1584까지 기록
- XGBoost 
  - RandomForestRegressor와 같은 조건에서 돌려서 0.1373 기록
  - Hyperparameter tuning을 통해 좀 더 나은 결과를 얻을 수 있을거라고 생각함
- 18/sep/2021, sklearn의 OLS(Ordinary Least Squares) 모듈을 사용하여 기록경신 (0.13450)
- 22/sep/2021, BO(Bayesian Optimization)과 GridSearchCV를 사용하여 Hyperparameter tuning을 시도하였으나 기록경신에는 실패.
- 24/sep/2021, PCA(Principal component analysis)를 사용하여 차원 축소하여 OLS 시도.
- 24/sep/2021, Ridge와 EN(ElasticNet), GBR(Gradient Boosting Regression), LGB(LightGBM)를 사용하여 시도. EN으로 기록 경신 `Best Score:0.13433`
- 24/sep/2021, VR(VotingRegressor)를 사용하여 RF, XGB, LGBM, Ridge, EN, GBR, LGB를 조합하여 시도했으나 기록경신에는 실패.
- 25/sep/2021, scaling할 때와 PCA 모듈을 사용할 때 test data를 transform하는데 오류가 있음을 확인하고 전처리를 다시하여 submission파일 제출. 기록경신에는 실패.
- 03/oct/2021, DNN(Deep Neural Network)를 사용하여 예측 시도.
- 06/oct/2021, ElasticNet을 하이퍼파라미터 튜닝하여 제출하였으나 기록 경신 실패.

## Evaluating
- Evaluation standard : RMSLE
- Use K-fold validation to evaluate our model

### Ensemble
- XGBoost / LightGBM 등을 각각 K-fold validation을 통해 fit하고 예측값을 얻을 것임.

- 그 후 각 예측값들을 합쳐서(ensemble) 더 좋은 예측값을 기대.

  

