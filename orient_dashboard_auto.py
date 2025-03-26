
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup

st.set_page_config(layout="wide")
st.title("오리엔트정공 감성 분석 기반 주가 예측 대시보드")

# 감성 키워드 기반 점수화 함수
positive_keywords = ['무죄', '급등', '반등', '수혜', '가시화', '강세']
negative_keywords = ['탄핵', '주의보', '불안정']

def classify_sentiment(text):
    pos = sum(1 for word in positive_keywords if word in text)
    neg = sum(1 for word in negative_keywords if word in text)
    if pos > neg:
        return 1
    elif neg > pos:
        return -1
    else:
        return 0

# 네이버 뉴스 자동 수집 함수
def fetch_naver_news(query="오리엔트정공", max_count=5):
    news_list = []
    url = f"https://search.naver.com/search.naver?where=news&query={query}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, 'html.parser')
    articles = soup.select("a.news_tit")
    for i, a in enumerate(articles):
        if i >= max_count:
            break
        news_list.append(a['title'])
    return news_list

# 입력 뉴스
st.subheader("1. 뉴스 수집 방법 선택")
mode = st.radio("뉴스 입력 방식 선택", ["직접 입력", "네이버 자동 수집"])

if mode == "직접 입력":
    news_input = st.text_area("오리엔트정공 관련 뉴스 제목 또는 내용 입력", "이재명 무죄 확정, 테마주 반등 기대")
    score = classify_sentiment(news_input)
    st.write(f"→ 감성 점수: {score}")
    sentiment_scores = [score]
else:
    if st.button("📰 네이버 뉴스 불러오기"):
        news_list = fetch_naver_news("오리엔트정공")
        st.write("불러온 뉴스 제목:")
        for i, title in enumerate(news_list, 1):
            st.markdown(f"{i}. {title}")
        sentiment_scores = [classify_sentiment(title) for title in news_list]
        avg_score = np.mean(sentiment_scores)
        st.write(f"→ 평균 감성 점수: {avg_score:.2f}")
    else:
        sentiment_scores = []

# 학습용 데이터셋 (샘플)
X_train = pd.DataFrame({'sentiment_score': [-2, -1, 0, 1, 2, 3]})
y_train = pd.Series([7249, 7363, 7477, 7592, 7706, 7821])
model = LinearRegression().fit(X_train, y_train)

# 예측
if sentiment_scores:
    input_score = np.mean(sentiment_scores)
    predicted_price = model.predict([[input_score]])[0]
    st.subheader("2. 예측 결과")
    st.metric(label="예측 종가", value=f"{predicted_price:,.0f} 원")

    # 시각화
    st.subheader("3. 감성 점수별 주가 예측 시각화")
    fig, ax = plt.subplots()
    ax.plot(X_train['sentiment_score'], y_train, marker='o', label='예측 선')
    ax.scatter(input_score, predicted_price, color='red', label='현재 입력')
    ax.set_xlabel('감성 점수')
    ax.set_ylabel('예측 주가')
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
