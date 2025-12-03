from datasets import load_dataset
import os
from dotenv import load_dotenv
import google.generativeai as genai
import time

# データセット読み込み
question_dataset = load_dataset("nlp-waseda/JMMLU", "japanese_history", split="test", trust_remote_code=True)

print(f"問題数: {len(question_dataset)}")
print("実験開始: 全ての正解を[D]に入れ替えて推論します")

# 出力先設定
os.makedirs("/home/koyama/nlp-100knocks/5章/out", exist_ok=True)
output_file = os.path.join("/home/koyama/nlp-100knocks/5章/out", "out_43.txt")

# API設定
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel("gemini-2.0-flash")

correct_counter = 0
answer_list = []

for i, row in enumerate(question_dataset):
    question_text = row['question']
    
    # --- 選択肢の入れ替えロジック ---
    
    # 1. 元の正解の記号と、その文章を取得
    original_ans_char = row['answer']         # 例: 'B'
    correct_content = row[original_ans_char]  # 例: '宗尊親王' (正解の文章)
    
    # 2. 不正解の文章をリストに集める
    wrong_contents = []
    for char in ['A', 'B', 'C', 'D']:
        if char != original_ans_char:
            wrong_contents.append(row[char])
            
    # 3. 新しい配置を作成（Dを正解に固定！）
    # A, B, C には不正解の文章を順番に入れる
    new_A = wrong_contents[0]
    new_B = wrong_contents[1]
    new_C = wrong_contents[2]
    new_D = correct_content  # <--- 正解をDに配置
    
    # 4. プロンプト用の選択肢テキストを作成
    answer_option = f"A. {new_A}\nB. {new_B}\nC. {new_C}\nD. {new_D}"
    
    # ----------------------------------------------

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
    
    try:
        response = model.generate_content(prompt)
        
        # 予測の取得と整形
        pred_text = response.text.strip()
        if len(pred_text) > 0:
            pred = pred_text[0]
        else:
            pred = ""
            
        print(f"問{i+1}: 予測[{pred}] (正解はD固定)")
        answer_list.append(pred)

        # 今回は正解をDに固定したので、Dなら正解
        if pred == 'D':
            correct_counter += 1
            
    except Exception as e:
        print(f"エラー: {e}")
        answer_list.append("Error")

    # レート制限対策
    time.sleep(7)

# 正解率計算
total_questions = len(question_dataset)
correct_per = (correct_counter / total_questions) * 100

print(f"最終結果: 正解数 {correct_counter}/{total_questions}  正解率: {correct_per:.2f}%")

# ファイル保存（フォーマット修正版）
with open(output_file, "w", encoding="utf-8") as output_f:
    # ヘッダー書き込み
    output_f.write(f"予測\t正解\n")
    
    for ans in answer_list:
        # この実験では正解は常に 'D' なので、正解列には 'D' を書き込みます
        output_f.write(f"{ans}\tD\n")
        
    # 最終行に集計結果
    output_f.write(f"正解数: {correct_counter}/{total_questions}  正解率：{correct_per:.2f}%")