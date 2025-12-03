import os
from dotenv import load_dotenv
import google.generativeai as genai
import time

os.makedirs("/home/koyama/nlp-100knocks/5章/out", exist_ok=True)
input_file = "/home/koyama/nlp-100knocks/5章/out/out_46.txt"
output_file = os.path.join("/home/koyama/nlp-100knocks/5章/out", "out_47.txt")

with open(input_file, "r", encoding = "utf-8") as f:
    senryuu = f.readlines()
    senryuu_text = "".join(senryuu)


# API設定
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel("gemini-2.0-flash")

prompt = (
    f"""
    あなたは川柳制作の専門家です．
    以下の情報をもとに，「言語」というお題で作成された川柳の面白さを10段階（1~10）評価してください．

    ###川柳とは
    ー上五・中七・下五の十七音から成り立っています。ただし、上句は七音程度は許される場合もあります。
    ー 普段使っている言葉で見たり、聞いたり、感じたこと、訴えたいこと、願っていることなどを言葉に託して自分の気持ちを詠みます。
    ー表記は口語体、現代仮名遣いを使います。俳句のような切れ字や季語は必要としません。
    
    ###出力形式
    ー面白さは10段階（1~10）で評価してください．
    ー出力は以下のようにしてください
    [川柳1]
    1.面白さ：
    2.理由：

    [川柳2]
    1.面白さ：
    2.理由：

    ###評価先
    {senryuu_text}
    """
)

response = model.generate_content(prompt)
with open(output_file, "w", encoding="utf-8") as output_f:
    output_f.write(f"{response.text}")