import os
from dotenv import load_dotenv
import google.generativeai as genai
import time

os.makedirs("/home/koyama/nlp-100knocks/5章/out", exist_ok=True)
output_file = os.path.join("/home/koyama/nlp-100knocks/5章/out", "out_45.txt")

# API設定
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel("gemini-2.0-flash")

question_text1="つばめちゃんは渋谷駅から東急東横線に乗り、自由が丘駅で乗り換えました。東急大井町線の大井町方面の電車に乗り換えたとき、各駅停車に乗車すべきところ、間違えて急行に乗車してしまったことに気付きました。自由が丘の次の急行停車駅で降車し、反対方向の電車で一駅戻った駅がつばめちゃんの目的地でした。目的地の駅の名前を答えてください。"
question_text2="さらに、つばめちゃんが自由が丘駅で乗り換えたとき、先ほどとは反対方向の急行電車に間違って乗車してしまった場合を考えます。目的地の駅に向かうため、自由が丘の次の急行停車駅で降車した後、反対方向の各駅停車に乗車した場合、何駅先の駅で降りれば良いでしょうか？"

prompt = (
    f"""
    あなたは問題解答のアシスタントです．以下の問題を解答してください．
    以下の情報を参考にして，路線の停車駅を順に追いながら論理的に推論して答えてください．
    東急東横線: 渋谷 → 代官山 → 中目黒 → 祐天寺 → 学芸大学 → 都立大学 → 自由が丘 → 田園調布 → 多摩川 → 新丸子 → 武蔵小杉...

    東急大井町線: 大井町 → 下神明 → 戸越公園 → 中延 → 荏原町 → 旗の台 → 北千束 → 大岡山 → 緑が丘 → 自由が丘 → 九品仏 → 尾山台 → 等々力 → 上野毛 → 二子玉川 → 二子新地 → 高津 → 溝の口 → 梶が谷 → 宮崎台 → 宮前平 → 鷺沼 → たまプラーザ → あざみ野 → 江田 → 市が尾 → 藤が丘 → 青葉台 → 田奈 → 長津田 → つきみ野 → 中央林間

    東急大井町線の急行停車駅: 大井町、大岡山、自由が丘、二子玉川、溝の口、長津田、中央林間

    ###問題
    {question_text1}
    """
)

chat = model.start_chat(history=[])

response1 = chat.send_message(prompt)
response2 = chat.send_message(question_text2)

with open(output_file, "w", encoding="utf-8") as output_f:
    output_f.write(f"[問1]\n{response1.text}\n\n[問2]\n{response2.text}")