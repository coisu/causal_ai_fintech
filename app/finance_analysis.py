import pandas as pd
import yfinance as yf
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import os


# 📌 [1] 데이터 로드 및 전처리
print("\n🔍 데이터 로드 중...")
sp500 = pd.read_csv("/app/SP500.csv", index_col="observation_date", parse_dates=True)

# ✅ observation_date가 인덱스로 설정되었는지 확인
sp500.index.name = "observation_date"

# ✅ Time 컬럼이 없는 경우 추가
if "Time" not in sp500.columns:
    sp500["Time"] = range(len(sp500))

# ✅ Time을 정수형(int)으로 변환 (불필요한 문자열 변환 방지)
sp500["Time"] = sp500["Time"].astype(int)

# 📌 Daily Return & Volatility 계산
sp500["Daily Return"] = sp500["SP500"].pct_change(fill_method=None).fillna(0)
sp500["Volatility"] = sp500["Daily Return"].rolling(window=20).std().fillna(0)

# 📌 로그 수익률 추가
sp500["Log_Return"] = np.log(sp500["SP500"] / sp500["SP500"].shift(1)).fillna(0)

# 📌 거래량 변화율 추가
sp500["Volume_Change"] = sp500["SP500"].pct_change(fill_method=None).fillna(0)


# 📌 [1-2] 산업별 변동성 계산
print("\n🔍 산업별 변동성 데이터 추가 중...")

tickers = ["QQQ", "XLF", "XLE"]
data = yf.download(tickers, start="2015-01-01", end="2025-01-01", auto_adjust=False)

print("\n\n📌 다운로드된 데이터 컬럼 확인 (data):", data.columns)

# ✅ MultiIndex 해제
# data = data.stack(level=0).rename_axis(["Date", "Ticker"]).unstack(level=1)
data = data.swaplevel(axis=1).sort_index(axis=1)
print("\n\n📌 변환된 데이터 컬럼 확인 (data):", data.columns)


# ✅ 변동성 계산 (20일 이동 표준편차 적용)
sp500["Tech_Volatility"] = data["QQQ"]["Close"].pct_change().rolling(20).std()
sp500["Finance_Volatility"] = data["XLF"]["Close"].pct_change().rolling(20).std()
sp500["Energy_Volatility"] = data["XLE"]["Close"].pct_change().rolling(20).std()

# tech = yf.download("QQQ", start="2015-01-01", end="2025-01-01", auto_adjust=False)
# finance = yf.download("XLF", start="2015-01-01", end="2025-01-01", auto_adjust=False)
# energy = yf.download("XLE", start="2015-01-01", end="2025-01-01", auto_adjust=False)

# print("\n\n📌 다운로드된 데이터 컬럼 확인 (Tech):", tech.columns)
# print("📌 다운로드된 데이터 컬럼 확인 (Finance):", finance.columns)
# print("📌 다운로드된 데이터 컬럼 확인 (Energy):", energy.columns)
# print("\n\n")

# tech.columns = tech.columns.droplevel(1)
# finance.columns = finance.columns.droplevel(0)
# energy.columns = energy.columns.droplevel(0)

# print("\n\n📌 수정 종가 컬럼 확인 (Tech):", tech.columns)
# print("📌 수정 종가 컬럼 확인 (Finance):", finance.columns)
# print("📌 수정 종가 컬럼 확인 (Energy):", energy.columns)

# sp500["Tech_Volatility"] = tech["Close"]["QQQ"].pct_change().rolling(20).std()
# sp500["Finance_Volatility"] = finance["Close"]["XLF"].pct_change().rolling(20).std()
# sp500["Energy_Volatility"] = energy["Close"]["XLE"].pct_change().rolling(20).std()



# 📌 [2] 주요 경제 지표 데이터 결합
print("\n🔍 경제 지표 데이터 로드 중...")

fed_rate = pd.read_csv("/app/DFF.csv", index_col="observation_date", parse_dates=True)
fed_rate.rename(columns={"DFF": "FED_Rate"}, inplace=True)

unemployment_rate = pd.read_csv("/app/UNRATE.csv", index_col="observation_date", parse_dates=True)
unemployment_rate.rename(columns={"UNRATE": "Unemployment_Rate"}, inplace=True)

wti_oil = pd.read_csv("/app/DCOILWTICO.csv", index_col="observation_date", parse_dates=True)
wti_oil.rename(columns={"DCOILWTICO": "WTI_Oil_Price"}, inplace=True)

core_cpi = pd.read_csv("/app/CORESTICKM159SFRBATL.csv", index_col="observation_date", parse_dates=True)
core_cpi.rename(columns={"CORESTICKM159SFRBATL": "CPI"}, inplace=True)

# 📌 데이터 병합
sp500 = sp500.join([fed_rate, unemployment_rate, wti_oil, core_cpi], how="left")

# 📌 결측값 처리
sp500.ffill(inplace=True)
sp500.bfill(inplace=True)
sp500.dropna(inplace=True)

# ✅ 데이터 저장
sp500.to_csv("/app/sp500_processed.csv", index=True)
print("✅ 데이터 저장 완료: /app/sp500_processed.csv")


# 📌 [3] 이벤트 변수 추가
print("\n🔍 이벤트 변수 추가 중...")

sp500["Vaccine_Announced"] = (sp500.index >= "2020-11-09").astype(int)
sp500["Covid_Announced"] = (sp500.index >= "2020-02-27").astype(int)  
sp500["Fed_Rate_Cut"] = (sp500.index == "2020-03-15").astype(int)
sp500["Ukraine_War"] = (sp500.index >= "2022-02-24").astype(int)

sp500["Fed_Hike"] = (sp500.index >= "2022-03-16").astype(int)  
sp500["Oil_Surge"] = (sp500.index == "2022-03-08").astype(int)  
sp500["Oil_Crash"] = (sp500.index == "2020-04-20").astype(int)  

print("✅ 이벤트 변수 추가 완료")


# 📌 [4] OLS 회귀 분석 (금리 변동 vs 산업 변동성)
print("\n🔍 OLS 회귀 분석 수행 중...")

for industry in ["Tech_Volatility", "Finance_Volatility", "Energy_Volatility"]:
    X = sp500[["Time", "Fed_Hike", "FED_Rate", "Unemployment_Rate", "WTI_Oil_Price", "CPI"]]
    X = sm.add_constant(X)  # 상수 추가
    y = sp500[industry]

    model = sm.OLS(y, X).fit()
    print(f"\n📈 {industry} OLS 금리 변동 분석 결과:")
    print(model.summary())


# 📌 [5] 유가 급등/급락 영향 분석 (DML 적용)
print("\n🔍 유가 급등/급락 인과적 효과 분석 중...")

from econml.dml import DML
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(sp500[["FED_Rate", "Unemployment_Rate", "CPI"]])

ridge_cv = RidgeCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1])

for industry in ["Tech_Volatility", "Finance_Volatility", "Energy_Volatility"]:
    y = sp500[industry]

    dml = DML(model_y=RandomForestRegressor(n_estimators=100, random_state=42),
              model_t=RandomForestRegressor(n_estimators=100, random_state=42),
              model_final=ridge_cv)

    dml.fit(y, sp500["Oil_Surge"], X=X_scaled)
    treatment_effects = dml.effect(X_scaled)

    print(f"\n🎯 유가 급등이 {industry}에 미치는 인과적 효과:", np.mean(treatment_effects))


# 📌 [6] 최종 결과 출력 및 시각화
print("\n✅ 모든 분석 완료. 결과를 검토하세요.")

plt.figure(figsize=(12, 6))
sns.lineplot(x=sp500.index, y=sp500["Volatility"], label="S&P 500 Volatility", color="blue")
plt.title("S&P 500 Volatility Over Time")
plt.xlabel("Date")
plt.ylabel("Volatility (20-day Rolling)")
plt.legend()
plt.savefig("/app/sp500_volatility.png")

print("✅ Volatility Plot 저장 완료: /app/sp500_volatility.png")
