import os
from dotenv import load_dotenv
import google.generativeai as genai
import time

os.makedirs("/home/koyama/nlp-100knocks/5章/out", exist_ok=True)
output_file = os.path.join("/home/koyama/nlp-100knocks/5章/out", "out_49.txt")

# API設定
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel("gemini-2.0-flash")

question_text="""吾輩は猫である。名前はまだ無い。
どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。吾輩はここで始めて人間というものを見た。しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。この書生というのは時々我々を捕えて煮て食うという話である。しかしその当時は何という考もなかったから別段恐しいとも思わなかった。ただ彼の掌に載せられてスーと持ち上げられた時何だかフワフワした感じがあったばかりである。掌の上で少し落ちついて書生の顔を見たのがいわゆる人間というものの見始であろう。この時妙なものだと思った感じが今でも残っている。第一毛をもって装飾されべきはずの顔がつるつるしてまるで薬缶だ。その後猫にもだいぶ逢ったがこんな片輪には一度も出会わした事がない。のみならず顔の真中があまりに突起している。そうしてその穴の中から時々ぷうぷうと煙を吹く。どうも咽せぽくて実に弱った。これが人間の飲む煙草というものである事はようやくこの頃知った。
"""

token_info = model.count_tokens(question_text)
token_cnt = token_info.total_tokens

with open(output_file, "w", encoding="utf-8") as output_f:
    output_f.write(f"{question_text}\n\n")
    output_f.write(f"文字数:{len(question_text)}\n")
    output_f.write(f"トークン数:{token_cnt}")