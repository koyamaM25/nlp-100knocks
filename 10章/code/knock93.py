from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import os

# =========================
# Paths
# =========================
OUTPUT_PATH = "/home/koyama/nlp-100knocks/10章/out"
os.makedirs(OUTPUT_PATH, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_PATH, "out_93.txt")

# =========================
# Sentences to evaluate (edit freely)
# =========================
sentences = [
    "The movie was full of big laughs.",
    "The movie was full of big laugh.",
    "The movie was full of action, comedy, and a lot of fun.",
    "The movie was full of action comedy and lot fun.",
    "I have a pen.",
    "I has a pen."
]

# =========================
# Load model
# =========================
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()

def perplexity_for_text(text: str) -> dict:
    """
    Compute perplexity for a single text using GPT-2 as a causal LM.
    Returns a dict with ppl, avg_nll, n_tokens.
    """
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Forward: logits for next-token prediction at each position
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits  # (1, L, V)

    # Shift so that logits[t] predicts token[t+1]
    shift_logits = logits[:, :-1, :]          # (1, L-1, V)
    shift_labels = input_ids[:, 1:]           # (1, L-1)

    # Token-level negative log-likelihood
    # CrossEntropyLoss expects (N, C) and (N,)
    nll_per_token = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
    )  # ((L-1),)

    avg_nll = nll_per_token.mean().item()
    ppl = float(torch.exp(nll_per_token.mean()).item())
    n_tokens = shift_labels.numel()

    return {"ppl": ppl, "avg_nll": avg_nll, "n_tokens": n_tokens}

with open(OUTPUT_FILE, "w", encoding="UTF-8") as f:
    f.write("text\tppl\tavg_nll\tn_tokens\n")
    for s in sentences:
        r = perplexity_for_text(s)
        f.write(f"{s}\t{r['ppl']:.6f}\t{r['avg_nll']:.6f}\t{r['n_tokens']}\n")

print(f"Wrote: {OUTPUT_FILE}")