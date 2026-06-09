import pandas as pd
import re


# 1. 원본 데이터 불러오기
df = pd.read_csv(
    './app/review/dataset/kr3_raw.tsv',
    sep="\t",
    encoding="utf-8"
)

print("원본 데이터 크기:", df.shape)
print("원본 컬럼:", df.columns.tolist())


# 2. 필요한 컬럼만 사용
# Region은 우리 서비스에서 사용하지 않음
df = df[['Rating', 'Category', 'Review']].copy()


# 3. 카페 카테고리만 필터링
# Category는 필터링에만 사용하고 최종 저장 파일에서는 제거함
cafe_df = df[df['Category'].astype(str).str.contains('카페', na=False)].copy()

print("카페 필터링 후:", cafe_df.shape)


# 4. Review 비어있는 데이터 제거
cafe_df = cafe_df.dropna(subset=['Review']).copy()

print("Review null 제거 후:", cafe_df.shape)


# 5. 리뷰 텍스트 정제 함수
def clean_review(text):
    text = str(text)

    # 줄바꿈, 탭 제거
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('\t', ' ')

    # URL 제거
    text = re.sub(r'http\S+|www\.\S+', ' ', text)

    # HTML 태그 제거
    text = re.sub(r'<[^>]+>', ' ', text)

    # 여러 공백을 하나로 정리
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# 6. 리뷰 텍스트 정제 적용
cafe_df['review_text'] = cafe_df['Review'].apply(clean_review)


# 7. 빈 문자열 제거
cafe_df = cafe_df[cafe_df['review_text'] != ''].copy()

print("빈 리뷰 제거 후:", cafe_df.shape)


# 8. 한글 없는 리뷰 제거
def has_korean(text):
    return bool(re.search(r'[가-힣]', str(text)))


cafe_df = cafe_df[cafe_df['review_text'].apply(has_korean)].copy()

print("한글 필터 후:", cafe_df.shape)


# 9. 리뷰 길이 필터링
# review_length는 필터링용으로만 사용하고 최종 저장하지 않음
cafe_df['review_length'] = cafe_df['review_text'].str.len()

cafe_df = cafe_df[
    (cafe_df['review_length'] >= 6) &
    (cafe_df['review_length'] <= 1000)
].copy()

print("길이 필터 후:", cafe_df.shape)


# 10. Rating 정리
# 숫자로 변환 불가능한 값은 NaN으로 처리
cafe_df['Rating'] = pd.to_numeric(cafe_df['Rating'], errors='coerce')

# 0, 1, 2만 사용
cafe_df = cafe_df[cafe_df['Rating'].isin([0, 1, 2])].copy()
cafe_df['Rating'] = cafe_df['Rating'].astype(int)

print("Rating 정리 후:", cafe_df.shape)


# 11. Rating을 원본 기준 전체 감정 라벨로 변환
# 이 값은 최종 감정이 아니라 원본 데이터셋에서 온 참고 감정임
def map_sentiment(rating):
    if rating == 0:
        return 'NEGATIVE'
    elif rating == 1:
        return 'POSITIVE'
    elif rating == 2:
        return 'NEUTRAL'


cafe_df['source_overall_sentiment'] = cafe_df['Rating'].apply(map_sentiment)


# 12. 중복 리뷰 제거
before = len(cafe_df)
cafe_df = cafe_df.drop_duplicates(subset=['review_text']).copy()
after = len(cafe_df)

print("중복 제거:", before, "->", after)


# 13. review_id 새로 생성
cafe_df = cafe_df.reset_index(drop=True)
cafe_df['review_id'] = cafe_df.index + 1


# 14. 최종 전처리 데이터 생성
# 최종 저장 컬럼은 3개만 사용
final_df = cafe_df[[
    'review_id',
    'review_text',
    'source_overall_sentiment'
]].copy()


# 15. 전체 정제 데이터 저장
final_df.to_csv(
    './app/review/dataset/cafe_reviews_clean.csv',
    index=False,
    encoding='utf-8-sig'
)


# 16. 전체 감성 모델 학습용 저장
# 현재는 cafe_reviews_clean.csv와 같은 구조지만,
# 나중에 전체 감성 모델 학습용 파일로 따로 관리하기 위해 분리 저장함
final_df.to_csv(
    './app/review/dataset/cafe_overall_sentiment.csv',
    index=False,
    encoding='utf-8-sig'
)


# 17. Qwen 라벨링용 균형 샘플 만들기
# 긍정/부정/중립을 최대 800개씩 뽑음
positive_df = final_df[final_df['source_overall_sentiment'] == 'POSITIVE']
negative_df = final_df[final_df['source_overall_sentiment'] == 'NEGATIVE']
neutral_df = final_df[final_df['source_overall_sentiment'] == 'NEUTRAL']

positive_sample = positive_df.sample(
    n=min(800, len(positive_df)),
    random_state=42
)

negative_sample = negative_df.sample(
    n=min(800, len(negative_df)),
    random_state=42
)

neutral_sample = neutral_df.sample(
    n=min(800, len(neutral_df)),
    random_state=42
)

qwen_sample_df = pd.concat([
    positive_sample,
    negative_sample,
    neutral_sample
])

# 섞기
qwen_sample_df = qwen_sample_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# 18. Qwen 라벨링용 파일 저장
qwen_sample_df.to_csv(
    './app/review/dataset/cafe_reviews_for_qwen_labeling.csv',
    index=False,
    encoding='utf-8-sig'
)


# 19. 결과 확인
print("\n저장 완료")
print("1. cafe_reviews_clean.csv")
print("2. cafe_overall_sentiment.csv")
print("3. cafe_reviews_for_qwen_labeling.csv")

print("\n전체 감성 분포")
print(final_df['source_overall_sentiment'].value_counts())

print("\nQwen 라벨링 샘플 분포")
print(qwen_sample_df['source_overall_sentiment'].value_counts())

print("\n최종 데이터 샘플")
print(final_df.head(10))