from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import os

# =========================
# Path / Settings
# =========================
OUTPUT_PATH = "/home/koyama/nlp-100knocks/10章/out"
os.makedirs(OUTPUT_PATH, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_PATH, "out_92.txt")

text = "The movie was full of"

temperatures = [0.5, 0.7, 1.0, 1.5, 2.0]
NUM_RETURN_SEQS = 1
MAX_NEW_TOKENS = 50

# =========================
# Load model
# =========================
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()

# Tokenize prompt
inputs = tokenizer(text, return_tensors="pt")
prompt_len = inputs["input_ids"].shape[1]  # N

# =========================
# Helper: compute per-token probs for generated tokens
# =========================
def compute_token_probs(gen_out, prompt_len: int):
    sequences = gen_out.sequences  # (K, N+M)
    scores = gen_out.scores        # list length M, each (K, V)

    # M = number of generated tokens actually produced
    M = len(scores)
    K = sequences.shape[0]

    # Generated token ids (K, M)
    token_ids_gen = sequences[:, prompt_len:prompt_len + M]

    # Collect logprobs for chosen tokens
    logprobs_list = []
    for t in range(M):
        logits_t = scores[t]  # (K, V)
        logp_t = F.log_softmax(logits_t, dim=-1)  # (K, V)

        # chosen token ids at step t: token_ids_gen[:, t]  (K,)
        chosen_ids = token_ids_gen[:, t].unsqueeze(-1)     # (K, 1)
        chosen_logp = logp_t.gather(dim=-1, index=chosen_ids).squeeze(-1)  # (K,)
        logprobs_list.append(chosen_logp)

    token_logprobs = torch.stack(logprobs_list, dim=1)  # (K, M)
    token_probs = torch.exp(token_logprobs)             # (K, M)
    return sequences, token_ids_gen, token_logprobs, token_probs

# =========================
# Run + Write
# =========================
with open(OUTPUT_FILE, "w", encoding="UTF-8") as f:
    for temp in temperatures:
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                do_sample=True,
                temperature=temp,
                max_new_tokens=MAX_NEW_TOKENS,
                num_return_sequences=NUM_RETURN_SEQS,
                return_dict_in_generate=True,
                output_scores=True,
            )

        sequences, token_ids_gen, token_logprobs, token_probs = compute_token_probs(gen, prompt_len)
        K, M = token_ids_gen.shape

        f.write(f"Temperature: {temp}\n")
        f.write(f"Prompt: {text}\n")
        f.write(f"Generated tokens: {M}\n\n")

        for k in range(K):
            full_text = tokenizer.decode(sequences[k], skip_special_tokens=True)

            sum_logp = token_logprobs[k].sum().item()
            avg_logp = (token_logprobs[k].mean().item() if M > 0 else float("nan"))

            f.write(f"[{k+1}] {full_text}\n")
            f.write(f"sum_logprob: {sum_logp:.6f}\n")
            f.write(f"avg_logprob: {avg_logp:.6f}\n")
            f.write("t\ttoken\tprob\tlogprob\n")

            for t in range(M):
                tid = token_ids_gen[k, t].item()
                tok = tokenizer.decode([tid])  # keep leading spaces visible in raw form
                p = token_probs[k, t].item()
                lp = token_logprobs[k, t].item()
                f.write(f"{t+1}\t{repr(tok)}\t{p:.8f}\t{lp:.8f}\n")

            f.write("\n")
        f.write("=" * 60 + "\n\n")