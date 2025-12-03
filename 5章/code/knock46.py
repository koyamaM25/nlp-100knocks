import os
from dotenv import load_dotenv
import google.generativeai as genai
import time

os.makedirs("/home/koyama/nlp-100knocks/5章/out", exist_ok=True)
output_file = os.path.join("/home/koyama/nlp-100knocks/5章/out", "out_46.txt")

# API設定
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel("gemini-2.0-flash")

prompt = (
    f"""
    あなたは川柳制作の専門家です．
    以下の情報をもとに，「言語」というお題で川柳の案を10個作成してください．

    ###川柳とは
    ー上五・中七・下五の十七音から成り立っています。ただし、上句は七音程度は許される場合もあります。
    ー 普段使っている言葉で見たり、聞いたり、感じたこと、訴えたいこと、願っていることなどを言葉に託して自分の気持ちを詠みます。
    ー表記は口語体、現代仮名遣いを使います。俳句のような切れ字や季語は必要としません。
    
    ### 出力形式
    後で集計するため、以下のように番号付きリストで出力してください。余計な前置きは不要です。

    1. 川柳の本文
    2. 川柳の本文
    """
)

response = model.generate_content(prompt)
with open(output_file, "w", encoding="utf-8") as output_f:
    output_f.write(f"{response.text}")