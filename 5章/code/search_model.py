import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

print("利用可能なモデル一覧:")
for m in genai.list_models():
    # "generateContent"（文章生成）に対応しているモデルだけを表示
    if 'generateContent' in m.supported_generation_methods:
        print(f"- {m.name}")