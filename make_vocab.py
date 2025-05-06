import pickle
import re
from collections import Counter
from nltk.tokenize import word_tokenize
import pandas as pd

def preprocess(text):
    text = re.sub(r"[^\w\s]", "", str(text))
    return word_tokenize(text.lower())

# 1. 데이터 불러오기
df = pd.read_csv("naver_shopping.txt", sep="\t", names=["label", "review"])
df = df[df["label"].isin([1, 2, 4, 5])]
df["label"] = df["label"].replace({1: 1, 2: 1, 4: 0, 5: 0})

# 2. 토큰화
df["tokens"] = df["review"].apply(preprocess)

# 3. 단어 인덱스 생성 (단어 수를 8701로 맞추기 위해 빈도 수 기준 조정)
counter = Counter()
for tokens in df["tokens"]:
    counter.update(tokens)

# 상위 8699개만 남기고 (pad, unk 포함 = 총 8701)
most_common = counter.most_common(8699)
vocab = ["<pad>", "<unk>"] + [word for word, freq in most_common]
word_to_index = {word: idx for idx, word in enumerate(vocab)}

# 4. 저장
with open("word_to_index.pkl", "wb") as f:
    pickle.dump(word_to_index, f)

