import streamlit as st
import requests
import os
from PIL import Image

# Streamlit 설정
st.title("📊 금융 시장 인과 분석 대시보드")
st.write("경제 질문을 입력하면 금융 시장 변동성에 미치는 영향을 분석합니다.")

# 사용자 질문 입력
question = st.text_input("💡 분석할 경제 질문을 입력하세요:", "금리가 오르면 금융 시장 변동성은 어떻게 변할까?")

if st.button("🔍 분석 실행"):
    # FastAPI 엔드포인트 호출
    api_url = "http://causal_ai_app:8000/analyze"
    response = requests.get(api_url, params={"question": question})

    if response.status_code == 200:
        result = response.json()

        # 결과가 존재하는지 확인
        if "DoWhy_result" in result and "Counterfactual_result" in result:
            dowhy_result = result["DoWhy_result"]
            counterfactual_result = result["Counterfactual_result"]

            # 백분율 변환 (너무 작은 값일 경우)
            dowhy_percentage = dowhy_result * 100
            counterfactual_percentage = counterfactual_result * 100

            st.subheader("📌 분석 결과")
            if abs(dowhy_result) < 0.001 and abs(counterfactual_result) < 0.001:
                st.write("🔹 **금리 변화가 금융 시장 변동성에 미치는 영향이 매우 미미합니다.**")
            else:
                st.write(f"**DoWhy 검증 결과:** {dowhy_result:.4f} ({dowhy_percentage:.2f}%)")
                st.write(f"**반사실 분석 (Counterfactual Analysis) 결과:** {counterfactual_result:.4f} ({counterfactual_percentage:.2f}%)")

            # 결과 이미지 표시
            image_path = result.get("Visualization", "")
            if image_path and os.path.exists(image_path):
                image = Image.open(image_path)
                st.image(image, caption="인과 효과 분포 그래프", use_column_width=True)
            else:
                st.warning("⚠️ 그래프 이미지를 불러올 수 없습니다.")
        else:
            st.error(f"🚨 분석 결과가 충분하지 않습니다: {result}")
    else:
        st.error(f"API 호출 실패! 상태 코드: {response.status_code}")
