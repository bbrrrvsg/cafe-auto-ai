import pandas as pd
import re


# 1. 전처리 완료 파일 불러오기
df = pd.read_csv(
    './app/review/dataset/cafe_reviews_clean.csv',
    encoding='utf-8-sig'
)


# 2. 긍정/부정 키워드 패턴
# 이건 최종 라벨링용이 아니라, Qwen에게 보낼 샘플을 고르기 위한 임시 기준입니다.
positive_patterns = [
    r"맛있",
    r"좋[아았음다네고은을]",
    r"친절",
    r"깔끔",
    r"깨끗",
    r"넓",
    r"편하|편했|편안",
    r"만족",
    r"추천",
    r"최고",
    r"예쁘|이쁘",
    r"훌륭",
    r"괜찮",
    r"무난",
    r"재방문",
    r"가성비",
    r"시원",
    r"따뜻",
    r"안락",
    r"조용",
    r"분위기.*좋",
    r"고소",
    r"부드럽",
    r"신선",
    r"빠르",
    r"감사",
    r"강추",
    r"성공적",
    r"매력적",
    r"알차"
]

negative_patterns = [
    r"별로",
    r"불친절",
    r"늦",
    r"오래.*기다|기다",
    r"비싸",
    r"아쉽|아쉬",
    r"더럽",
    r"끈적",
    r"시끄",
    r"복잡",
    r"불편",
    r"짜증",
    r"실망",
    r"최악",
    r"맛없|맛이 없",
    r"밍밍",
    r"불만",
    r"부족",
    r"차갑",
    r"딱딱",
    r"좁",
    r"느리",
    r"안 좋|안좋",
    r"나쁘",
    r"그닥",
    r"그저그래|그저 그런",
    r"기대.*못|기대.*덜|기대.*비해",
    r"다시.*안|재방문.*안|굳이.*안",
    r"떨어지",
    r"느끼"
]

positive_re = re.compile("|".join(positive_patterns))
negative_re = re.compile("|".join(negative_patterns))


def has_positive_signal(text):
    return bool(positive_re.search(str(text)))


def has_negative_signal(text):
    return bool(negative_re.search(str(text)))


# 3. NEUTRAL 안에서 의심 유형 나누기
df['has_positive_signal'] = df['review_text'].apply(has_positive_signal)
df['has_negative_signal'] = df['review_text'].apply(has_negative_signal)


def make_check_type(row):
    source = row['source_overall_sentiment']

    if source != 'NEUTRAL':
        return f"SOURCE_{source}"

    has_pos = row['has_positive_signal']
    has_neg = row['has_negative_signal']

    if has_pos and has_neg:
        return "NEUTRAL_COMPLEX_LIKE"

    if has_pos:
        return "NEUTRAL_POSITIVE_LIKE"

    if has_neg:
        return "NEUTRAL_NEGATIVE_LIKE"

    return "NEUTRAL_NO_SIGNAL"


df['label_check_type'] = df.apply(make_check_type, axis=1)


# 4. 점검 플래그 포함 전체 파일 저장
# 이 파일은 사람이 데이터 상태를 확인할 때 씁니다.
df_for_check = df[[
    'review_id',
    'review_text',
    'source_overall_sentiment',
    'label_check_type'
]].copy()

df_for_check.to_csv(
    './app/review/dataset/cafe_reviews_clean_with_check_type.csv',
    index=False,
    encoding='utf-8-sig'
)


# 5. Qwen 라벨링용 샘플 생성
def safe_sample(data, n):
    return data.sample(
        n=min(n, len(data)),
        random_state=42
    )


source_positive = df[df['label_check_type'] == 'SOURCE_POSITIVE']
source_negative = df[df['label_check_type'] == 'SOURCE_NEGATIVE']
neutral_positive_like = df[df['label_check_type'] == 'NEUTRAL_POSITIVE_LIKE']
neutral_negative_like = df[df['label_check_type'] == 'NEUTRAL_NEGATIVE_LIKE']
neutral_complex_like = df[df['label_check_type'] == 'NEUTRAL_COMPLEX_LIKE']
neutral_no_signal = df[df['label_check_type'] == 'NEUTRAL_NO_SIGNAL']


sample_df = pd.concat([
    safe_sample(source_positive, 600),
    safe_sample(source_negative, 600),
    safe_sample(neutral_positive_like, 300),
    safe_sample(neutral_negative_like, 200),
    safe_sample(neutral_complex_like, 500),
    safe_sample(neutral_no_signal, 300)
])

sample_df = sample_df.sample(frac=1, random_state=42).reset_index(drop=True)


# Qwen에 필요한 컬럼만 저장
qwen_sample_df = sample_df[[
    'review_id',
    'review_text',
    'source_overall_sentiment',
    'label_check_type'
]].copy()

qwen_sample_df.to_csv(
    './app/review/dataset/cafe_reviews_for_qwen_labeling.csv',
    index=False,
    encoding='utf-8-sig'
)


print("\nQwen 라벨링 샘플 저장 완료")
print(qwen_sample_df.shape)

print("\nQwen 샘플 source_overall_sentiment 분포")
print(qwen_sample_df['source_overall_sentiment'].value_counts())

print("\nQwen 샘플 label_check_type 분포")
print(qwen_sample_df['label_check_type'].value_counts())

print("\n샘플 미리보기")
print(qwen_sample_df.head(10))