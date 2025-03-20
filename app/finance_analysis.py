# import pandas as pd
# import yfinance as yf
# import numpy as np
# import os
# import statsmodels.api as sm
# import matplotlib.pyplot as plt
# import seaborn as sns
# from econml.dml import CausalForestDML
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LassoCV
# from sklearn.linear_model import RidgeCV
# from dowhy import CausalModel
# from fastapi import FastAPI, Query
# import joblib
# import uvicorn

# # 📌 데이터 저장 경로 설정
# data_dir = "/app/data"
# os.makedirs(data_dir, exist_ok=True)

# # ✅ [1] S&P 500 데이터 로드 및 변동성 계산
# print("\n🔍 S&P 500 데이터 로드 중...")
# sp500 = pd.read_csv(f"{data_dir}/SP500.csv", index_col="observation_date", parse_dates=True)

# # ✅ Time 컬럼 추가 (시간 흐름 고려한 회귀 분석용)
# sp500["Time"] = range(len(sp500))

# # ✅ 변동성 계산 (20일 이동 표준편차 적용)
# sp500["Daily Return"] = sp500["SP500"].pct_change().fillna(0)
# sp500["Volatility"] = sp500["Daily Return"].rolling(window=20).std().fillna(0)

# # ✅ 로그 수익률 추가
# sp500["Log_Return"] = np.log(sp500["SP500"] / sp500["SP500"].shift(1)).fillna(0)

# # ✅ 거래량 변화율 추가
# sp500["Volume_Change"] = sp500["SP500"].pct_change().fillna(0)

# # 📌 [2] 산업별 변동성 계산
# print("\n🔍 산업별 변동성 데이터 추가 중...")
# tickers = ["QQQ", "XLF", "XLE"]
# data = yf.download(tickers, start="2015-01-01", end="2025-01-01", auto_adjust=False)

# # ✅ 데이터 변환
# data = data.swaplevel(axis=1).sort_index(axis=1)

# # ✅ 산업별 변동성 추가
# sp500["Tech_Volatility"] = data["QQQ"]["Close"].pct_change().rolling(20).std()
# sp500["Finance_Volatility"] = data["XLF"]["Close"].pct_change().rolling(20).std()
# sp500["Energy_Volatility"] = data["XLE"]["Close"].pct_change().rolling(20).std()

# # 📌 [3] 주요 경제 지표 데이터 병합
# print("\n🔍 경제 지표 데이터 로드 중...")
# fed_rate = pd.read_csv(f"{data_dir}/DFF.csv", index_col="observation_date", parse_dates=True).rename(columns={"DFF": "FED_Rate"})
# unemployment_rate = pd.read_csv(f"{data_dir}/UNRATE.csv", index_col="observation_date", parse_dates=True).rename(columns={"UNRATE": "Unemployment_Rate"})
# wti_oil = pd.read_csv(f"{data_dir}/DCOILWTICO.csv", index_col="observation_date", parse_dates=True).rename(columns={"DCOILWTICO": "WTI_Oil_Price"})
# core_cpi = pd.read_csv(f"{data_dir}/CORESTICKM159SFRBATL.csv", index_col="observation_date", parse_dates=True).rename(columns={"CORESTICKM159SFRBATL": "CPI"})

# # ✅ 데이터 병합
# sp500 = sp500.join([fed_rate, unemployment_rate, wti_oil, core_cpi], how="left")

# # ✅ 결측값 처리
# sp500.ffill(inplace=True)
# sp500.bfill(inplace=True)
# sp500.dropna(inplace=True)

# # 📌 [4] 이벤트 변수 추가
# print("\n🔍 이벤트 변수 추가 중...")
# sp500["Vaccine_Announced"] = (sp500.index >= "2020-11-09").astype(int)
# sp500["Covid_Announced"] = (sp500.index >= "2020-02-27").astype(int)
# sp500["Fed_Rate_Cut"] = (sp500.index == "2020-03-15").astype(int)
# sp500["Ukraine_War"] = (sp500.index >= "2022-02-24").astype(int)

# sp500["Fed_Hike"] = (sp500.index >= "2022-03-16").astype(int)
# sp500["Oil_Surge"] = (sp500.index == "2022-03-08").astype(int)
# sp500["Oil_Crash"] = (sp500.index == "2020-04-20").astype(int)

# # ✅ 데이터 저장
# sp500.to_csv(f"{data_dir}/sp500_processed.csv")
# print(f"✅ 데이터 저장 완료: {data_dir}/sp500_processed.csv")

# # ✅ FastAPI 설정
# app = FastAPI()

# # ✅ [5] Causal AI 기반 금융 분석
# def analyze_financial_question(question):
#     print(f"📌 입력된 질문: {question}")  # 디버깅 로그 추가

#     if "금리" in question or "FED" in question:
#         X = sp500[["Unemployment_Rate", "CPI"]]
#         y = sp500["Volatility"]
#         treatment = sp500["FED_Rate"]
#         factor = "FED_Rate"
#     else:
#         print("🚨 관련 변수를 찾을 수 없습니다.")  # 디버깅 로그 추가
#         return {"error": "관련 변수를 찾을 수 없습니다."}

#     try:
#         # scaler = StandardScaler()
#         # X_scaled = scaler.fit_transform(X)

#         # # ✅ 1. Causal Forest DML 모델 적용
#         # cf_dml = CausalForestDML(
#         #     model_y=RandomForestRegressor(n_estimators=100, random_state=42),
#         #     model_t=RandomForestRegressor(n_estimators=100, random_state=42),
#         #     n_estimators=100,
#         #     random_state=42
#         # )
#         # cf_dml.fit(y, treatment, X=X_scaled)
#         # treatment_effects = cf_dml.effect(X_scaled)
#         # avg_effect = np.mean(treatment_effects)

#         # 🚀 DoWhy 모델 설정
#         model = CausalModel(
#             data=sp500, 
#             treatment=factor, 
#             outcome=y.name, 
#             common_causes=X.columns.tolist()
#         )

#         identified_estimand = model.identify_effect()
#         print(f"📌 Identified estimand 정보: {identified_estimand}")

#         # 🚀 인과 효과 추정
#         estimate = model.estimate_effect(
#             identified_estimand, 
#             method_name="backdoor.linear_regression"
#         )
        
#         # ✅ 결과 값 추출
#         dowhy_effect = estimate.value if hasattr(estimate, "value") else estimate
#         print(f"📌 DoWhy 결과값: {dowhy_effect}")

#         # ✅ Counterfactual 분석 적용
#         try:
#             intervention_value = float(sp500[factor].median())  # 🔥 float 변환 추가
#             print(f"📌 Median {factor}: {intervention_value}")

#             if np.isnan(intervention_value):
#                 print("🚨 Median value is NaN! Setting to 0.")
#                 intervention_value = 0  # NaN 방지

#             # ✅ 올바른 데이터 형식 전달
#             intervention_dict = {factor: intervention_value}
#             print(f"📌 DoWhy Intervention Dict: {intervention_dict}")

#             # 🔥 `identified_estimand`을 첫 번째 인자로 전달
#             counterfactual_result = model.do(identified_estimand, intervention_dict)
#             print("[2]-----------------------------")

#             # ✅ Counterfactual 결과 처리
#             if isinstance(counterfactual_result, pd.DataFrame):
#                 counterfactual_effect = counterfactual_result[y.name].mean()
#             elif isinstance(counterfactual_result, (float, int, np.float64)):
#                 counterfactual_effect = float(counterfactual_result)  # 🔥 float 변환 추가
#             else:
#                 counterfactual_effect = np.nan

#         except Exception as e:
#             print(f"🚨 `model.do()` 실행 중 오류 발생: {e}")
#             counterfactual_effect = np.nan  # 오류 방지

#         return {
#             "question": question,
#             "DoWhy_result": dowhy_effect,
#             "Counterfactual_result": counterfactual_effect,
#         }

#     except Exception as e:
#         print(f"🚨 분석 중 오류 발생: {e}")
#         return {"error": str(e)}



# # ✅ [6] FastAPI 엔드포인트 정의
# @app.get("/")
# def root():
#     return {"message": "Causal ML Server is Running 🚀"}

# @app.get("/analyze")
# def analyze_question(question: str = Query(..., title="Economic Question")):
#     try:
#         response = analyze_financial_question(question)
#         return response
#     except Exception as e:
#         return {"error": str(e)}

# # ✅ 실행
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)


# import pandas as pd
# import yfinance as yf
# import numpy as np
# import os
# import statsmodels.api as sm
# import matplotlib.pyplot as plt
# from dowhy import CausalModel
# from fastapi import FastAPI, Query
# import uvicorn
# import math

# # 📌 데이터 저장 경로 설정
# data_dir = "/app/data"
# os.makedirs(data_dir, exist_ok=True)

# # ✅ [1] S&P 500 데이터 로드 및 변동성 계산
# print("\n🔍 S&P 500 데이터 로드 중...")
# sp500 = pd.read_csv(f"{data_dir}/SP500.csv", index_col="observation_date", parse_dates=True)

# # ✅ Time 컬럼 추가 (시간 흐름 고려한 회귀 분석용)
# sp500["Time"] = range(len(sp500))

# # ✅ 변동성 계산 (20일 이동 표준편차 적용)
# sp500["Daily Return"] = sp500["SP500"].pct_change(fill_method=None).fillna(0)
# sp500["Volatility"] = sp500["Daily Return"].rolling(window=20).std().fillna(0)

# # ✅ 로그 수익률 추가
# sp500["Log_Return"] = np.log(sp500["SP500"] / sp500["SP500"].shift(1)).fillna(0)

# # ✅ 거래량 변화율 추가
# sp500["Volume_Change"] = sp500["SP500"].pct_change(fill_method=None).fillna(0)

# # 📌 [2] 산업별 변동성 계산
# print("\n🔍 산업별 변동성 데이터 추가 중...")
# tickers = ["QQQ", "XLF", "XLE"]
# data = yf.download(tickers, start="2015-01-01", end="2025-01-01", auto_adjust=False)

# # ✅ 데이터 변환
# data = data.swaplevel(axis=1).sort_index(axis=1)

# # ✅ 산업별 변동성 추가
# sp500["Tech_Volatility"] = data["QQQ"]["Close"].pct_change(fill_method=None).rolling(20).std()
# sp500["Finance_Volatility"] = data["XLF"]["Close"].pct_change(fill_method=None).rolling(20).std()
# sp500["Energy_Volatility"] = data["XLE"]["Close"].pct_change(fill_method=None).rolling(20).std()

# # 📌 [3] 주요 경제 지표 데이터 병합
# print("\n🔍 경제 지표 데이터 로드 중...")
# fed_rate = pd.read_csv(f"{data_dir}/DFF.csv", index_col="observation_date", parse_dates=True).rename(columns={"DFF": "FED_Rate"})
# unemployment_rate = pd.read_csv(f"{data_dir}/UNRATE.csv", index_col="observation_date", parse_dates=True).rename(columns={"UNRATE": "Unemployment_Rate"})
# wti_oil = pd.read_csv(f"{data_dir}/DCOILWTICO.csv", index_col="observation_date", parse_dates=True).rename(columns={"DCOILWTICO": "WTI_Oil_Price"})
# core_cpi = pd.read_csv(f"{data_dir}/CORESTICKM159SFRBATL.csv", index_col="observation_date", parse_dates=True).rename(columns={"CORESTICKM159SFRBATL": "CPI"})

# # ✅ 데이터 병합
# sp500 = sp500.join([fed_rate, unemployment_rate, wti_oil, core_cpi], how="left")

# # ✅ 결측값 처리
# sp500.ffill(inplace=True)
# sp500.bfill(inplace=True)
# sp500.dropna(inplace=True)

# # 📌 [4] 이벤트 변수 추가
# print("\n🔍 이벤트 변수 추가 중...")
# sp500["Vaccine_Announced"] = (sp500.index >= "2020-11-09").astype(int)
# sp500["Covid_Announced"] = (sp500.index >= "2020-02-27").astype(int)
# sp500["Fed_Rate_Cut"] = (sp500.index == "2020-03-15").astype(int)
# sp500["Ukraine_War"] = (sp500.index >= "2022-02-24").astype(int)

# sp500["Fed_Hike"] = (sp500.index >= "2022-03-16").astype(int)
# sp500["Oil_Surge"] = (sp500.index == "2022-03-08").astype(int)
# sp500["Oil_Crash"] = (sp500.index == "2020-04-20").astype(int)

# # ✅ 데이터 저장
# sp500.to_csv(f"{data_dir}/sp500_processed.csv")
# print(f"✅ 데이터 저장 완료: {data_dir}/sp500_processed.csv")

# # ✅ FastAPI 설정
# app = FastAPI()

# # ✅ [5] Causal AI 기반 금융 분석
# def analyze_financial_question(question):
#     print(f"📌 입력된 질문: {question}")  # 디버깅 로그 추가

#     if "금리" in question or "FED" in question:
#         X = sp500[["Unemployment_Rate", "CPI"]]
#         y = sp500["Volatility"]
#         treatment = sp500["FED_Rate"]
#         factor = "FED_Rate"
#     else:
#         print("🚨 관련 변수를 찾을 수 없습니다.")  # 디버깅 로그 추가
#         return {"error": "관련 변수를 찾을 수 없습니다."}

#     try:
#         # 🚀 DoWhy 모델 설정
#         model = CausalModel(
#             data=sp500, 
#             treatment=factor, 
#             outcome=y.name, 
#             common_causes=X.columns.tolist()
#         )

#         identified_estimand = model.identify_effect()
#         print(f"📌 Identified estimand 정보: {identified_estimand}")

#         # 🚀 인과 효과 추정
#         estimate = model.estimate_effect(
#             identified_estimand, 
#             method_name="backdoor.linear_regression"
#         )
        
#         # ✅ 결과 값 추출
#         dowhy_effect = estimate.value if hasattr(estimate, "value") else estimate
#         print(f"📌 DoWhy 결과값: {dowhy_effect}")

#         # ✅ Counterfactual 분석 적용
#         try:
#             intervention_value = float(sp500[factor].median()) if factor in sp500.columns else 0.0
#             print(f"📌 Median {factor}: {intervention_value}")

#             # ✅ 올바른 데이터 형식 전달
#             intervention_dict = {factor: intervention_value}
#             print(f"📌 DoWhy Intervention Dict: {intervention_dict}")

#             # 🔥 `do()` 실행 시 identified_estimand 제거
#             counterfactual_result = model.do(x=intervention_dict)

#             # ✅ Counterfactual 결과 처리
#             if isinstance(counterfactual_result, pd.DataFrame):
#                 counterfactual_effect = counterfactual_result[y.name].mean()
#             elif isinstance(counterfactual_result, (float, int, np.float64)):
#                 counterfactual_effect = float(counterfactual_result)
#             else:
#                 counterfactual_effect = np.nan

#         except Exception as e:
#             print(f"🚨 `model.do()` 실행 중 오류 발생: {e}")
#             counterfactual_effect = np.nan  # 오류 방지

#         return {
#             "question": question,
#             "DoWhy_result": dowhy_effect if math.isfinite(dowhy_effect) else None,
#             "Counterfactual_result": counterfactual_effect if math.isfinite(counterfactual_effect) else None,
#         }

#     except Exception as e:
#         print(f"🚨 분석 중 오류 발생: {e}")
#         return {"error": str(e)}

# # ✅ [6] FastAPI 엔드포인트 정의
# @app.get("/")
# def root():
#     return {"message": "Causal ML Server is Running 🚀"}

# @app.get("/analyze")
# def analyze_question(question: str = Query(..., title="Economic Question")):
#     try:
#         response = analyze_financial_question(question)
#         return response
#     except Exception as e:
#         return {"error": str(e)}

# # ✅ 실행
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)


import pandas as pd
import yfinance as yf
import numpy as np
import os
import statsmodels.api as sm
import matplotlib.pyplot as plt
from dowhy import CausalModel
from fastapi import FastAPI, Query
import uvicorn
import math

# 📌 데이터 저장 경로 설정
data_dir = "/app/data"
os.makedirs(data_dir, exist_ok=True)

# ✅ [1] S&P 500 데이터 로드 및 변동성 계산
print("\n🔍 S&P 500 데이터 로드 중...")
sp500 = pd.read_csv(f"{data_dir}/SP500.csv", index_col="observation_date", parse_dates=True)

# ✅ Time 컬럼 추가 (시간 흐름 고려한 회귀 분석용)
sp500["Time"] = range(len(sp500))

# ✅ 변동성 계산 (20일 이동 표준편차 적용)
sp500["Daily Return"] = sp500["SP500"].pct_change(fill_method=None).fillna(0)
sp500["Volatility"] = sp500["Daily Return"].rolling(window=20).std().fillna(0)

# ✅ 로그 수익률 추가
sp500["Log_Return"] = np.log(sp500["SP500"] / sp500["SP500"].shift(1)).fillna(0)

# ✅ 거래량 변화율 추가
sp500["Volume_Change"] = sp500["SP500"].pct_change(fill_method=None).fillna(0)

# 📌 [2] 산업별 변동성 계산
print("\n🔍 산업별 변동성 데이터 추가 중...")
tickers = ["QQQ", "XLF", "XLE"]
data = yf.download(tickers, start="2015-01-01", end="2025-01-01", auto_adjust=False)

# ✅ 데이터 변환
data = data.swaplevel(axis=1).sort_index(axis=1)

# ✅ 산업별 변동성 추가
sp500["Tech_Volatility"] = data["QQQ"]["Close"].pct_change(fill_method=None).rolling(20).std()
sp500["Finance_Volatility"] = data["XLF"]["Close"].pct_change(fill_method=None).rolling(20).std()
sp500["Energy_Volatility"] = data["XLE"]["Close"].pct_change(fill_method=None).rolling(20).std()

# 📌 [3] 주요 경제 지표 데이터 병합
print("\n🔍 경제 지표 데이터 로드 중...")
fed_rate = pd.read_csv(f"{data_dir}/DFF.csv", index_col="observation_date", parse_dates=True).rename(columns={"DFF": "FED_Rate"})
unemployment_rate = pd.read_csv(f"{data_dir}/UNRATE.csv", index_col="observation_date", parse_dates=True).rename(columns={"UNRATE": "Unemployment_Rate"})
wti_oil = pd.read_csv(f"{data_dir}/DCOILWTICO.csv", index_col="observation_date", parse_dates=True).rename(columns={"DCOILWTICO": "WTI_Oil_Price"})
core_cpi = pd.read_csv(f"{data_dir}/CORESTICKM159SFRBATL.csv", index_col="observation_date", parse_dates=True).rename(columns={"CORESTICKM159SFRBATL": "CPI"})

# ✅ 데이터 병합
sp500 = sp500.join([fed_rate, unemployment_rate, wti_oil, core_cpi], how="left")

# ✅ 결측값 처리
sp500.ffill(inplace=True)
sp500.bfill(inplace=True)
sp500.dropna(inplace=True)

# 📌 [4] 이벤트 변수 추가
print("\n🔍 이벤트 변수 추가 중...")
sp500["Vaccine_Announced"] = (sp500.index >= "2020-11-09").astype(int)
sp500["Covid_Announced"] = (sp500.index >= "2020-02-27").astype(int)
sp500["Fed_Rate_Cut"] = (sp500.index == "2020-03-15").astype(int)
sp500["Ukraine_War"] = (sp500.index >= "2022-02-24").astype(int)

sp500["Fed_Hike"] = (sp500.index >= "2022-03-16").astype(int)
sp500["Oil_Surge"] = (sp500.index == "2022-03-08").astype(int)
sp500["Oil_Crash"] = (sp500.index == "2020-04-20").astype(int)

# ✅ 데이터 저장
sp500.to_csv(f"{data_dir}/sp500_processed.csv")
print(f"✅ 데이터 저장 완료: {data_dir}/sp500_processed.csv")

# ✅ FastAPI 설정
app = FastAPI()

# ✅ [5] Causal AI 기반 금융 분석
def analyze_financial_question(question):
    print(f"📌 입력된 질문: {question}")

    if "금리" in question or "FED" in question:
        X = sp500[["Unemployment_Rate", "CPI"]]
        y = sp500["Volatility"]
        treatment = sp500["FED_Rate"]
        factor = "FED_Rate"
    else:
        print("🚨 관련 변수를 찾을 수 없습니다.")
        return {"error": "관련 변수를 찾을 수 없습니다."}

    try:
        # 🚀 DoWhy 모델 설정
        model = CausalModel(
            data=sp500, 
            treatment=factor, 
            outcome=y.name, 
            common_causes=X.columns.tolist()
        )

        identified_estimand = model.identify_effect()
        print(f"📌 Identified estimand 정보: {identified_estimand}")

        # 🚀 인과 효과 추정
        estimate = model.estimate_effect(
            identified_estimand, 
            method_name="backdoor.linear_regression"
        )

        # ✅ 결과 값 추출
        dowhy_effect = estimate.value if hasattr(estimate, "value") else estimate
        print(f"📌 DoWhy 결과값: {dowhy_effect}")

        # ✅ Counterfactual 분석 적용 (do() 없이 추정)
        try:
            intervention_value = float(sp500[factor].median()) if factor in sp500.columns else 0.0
            print(f"📌 Median {factor}: {intervention_value}")

            # ✅ DoWhy 모델 정의
            model = CausalModel(
                data=sp500,
                treatment=factor,
                outcome=y.name,
                common_causes=X.columns.tolist()
            )

            # ✅ 인과 효과 식별
            identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
            print(f"📌 Identified estimand 정보: {identified_estimand}")

            # ✅ 인과 효과 추정 (do() 없이 수행)
            causal_estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression"  # ✅ 선형 회귀로 변경
            )

            print(f"📌 Causal Estimate: {causal_estimate.value}")

            counterfactual_effect = causal_estimate.value if hasattr(causal_estimate, "value") else float(causal_estimate)

        except Exception as e:
            print(f"🚨 분석 중 오류 발생: {e}")
            counterfactual_effect = np.nan  # 오류 방지

        return {
            "question": question,
            "DoWhy_result": dowhy_effect if math.isfinite(dowhy_effect) else None,
            "Counterfactual_result": counterfactual_effect if math.isfinite(counterfactual_effect) else None,
        }

    except Exception as e:
        print(f"🚨 분석 중 오류 발생: {e}")
        return {"error": str(e)}

# ✅ FastAPI 엔드포인트 정의
@app.get("/analyze")
def analyze_question(question: str = Query(..., title="Economic Question")):
    return analyze_financial_question(question)

# ✅ 실행
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
