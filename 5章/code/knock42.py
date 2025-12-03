from datasets import load_dataset
import os
from dotenv import load_dotenv
import google.generativeai as genai
import time

question_dataset = load_dataset("nlp-waseda/JMMLU", "japanese_history", split="test",trust_remote_code=True)

print(f"問題数: {len(question_dataset)}")
print(question_dataset[0])

os.makedirs("/home/koyama/nlp-100knocks/5章/out", exist_ok=True)
output_file = os.path.join("/home/koyama/nlp-100knocks/5章/out", "out_42.txt")


load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")

correct_counter=0
answer_list = []
for row in question_dataset:
    question_text = row['question']
    answer_option = f"A. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}"
    answer_text = row['answer']
    prompt = (
        f"""
        あなたは歴史の専門家です．以下の問題を解答してください．

        ###制約事項
        ー問題文章から各選択肢のうちから正解となる選択肢を一つ選んでください
        ー出力は，記号(A,B,C,D)だけにしてください
        - 解説や文章は書かないでください

        ###問題
        {question_text}

        ###解答の選択肢
        {answer_option}
        """
    )
    response = model.generate_content(prompt)
    print(response.text)

    pred = response.text.strip()
    answer_list.append(pred)

    if pred == answer_text:
        correct_counter += 1

    time.sleep(7)

correct_per = correct_counter/len(question_dataset) * 100
print(f"正解数: {correct_counter}/{len(question_dataset)}  正解率: {correct_per:.2f}%")

with open(output_file, "w", encoding="utf-8") as output_f:
    output_f.write(f"予測\t正解\n")
    for ans,row in zip(answer_list,question_dataset):
        output_f.write(f"{ans}\t{row['answer']}\n")
    output_f.write(f"正解数: {correct_counter}/{len(question_dataset)}  正解率：{correct_per:.2f}%")