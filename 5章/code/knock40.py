import os
from dotenv import load_dotenv
import google.generativeai as genai
import os

os.makedirs("/home/koyama/nlp-100knocks/5章/out", exist_ok=True)
output_file = os.path.join("/home/koyama/nlp-100knocks/5章/out", "out_40.txt")


load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")

question ="""
9世紀に活躍した人物に関係するできごとについて述べた次のア～ウを年代の古い順に正しく並べよ。

ア　藤原時平は，策謀を用いて菅原道真を政界から追放した。
イ　嵯峨天皇は，藤原冬嗣らを蔵人頭に任命した。
ウ　藤原良房は，承和の変後，藤原氏の中での北家の優位を確立した。
"""

prompt = (
    f"""
    あなたは歴史の専門家です．以下の問題を解答してください．

    ###制約事項
    ー各選択肢の出来事が西暦何年の出来事か推論してください
    ー推論結果から，年代の古い順に並べた結果を出力してください
    ー出力は，記号だけにしてください

    ###問題
    {question}

    """
)
response = model.generate_content(prompt)
print(response.text)
with open(output_file, "w", encoding="utf-8") as output_f:
    output_f.write(f"{response.text}")