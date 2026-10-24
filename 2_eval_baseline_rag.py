import os
import json
import re
import string
from tqdm import tqdm
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# --- 1. SAFE CACHE & AUTH ---
cache_path = os.environ.get("HF_HOME", "hf_cache")
os.environ["HF_HOME"] = cache_path
os.environ["TRANSFORMERS_CACHE"] = cache_path
os.makedirs(cache_path, exist_ok=True)
hf_token = os.environ.get("HF_TOKEN")  # None: use the cached login in HF_HOME

# --- 2. SCORING LOGIC ---
def normalize_answer(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def f1_score(prediction, ground_truth):
    p_tokens = normalize_answer(prediction).split()
    g_tokens = normalize_answer(ground_truth).split()
    common = Counter(p_tokens) & Counter(g_tokens)
    num_same = sum(common.values())
    if num_same == 0: return 0
    precision = 1.0 * num_same / len(p_tokens)
    recall = 1.0 * num_same / len(g_tokens)
    return (2 * precision * recall) / (precision + recall)

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def main():
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    batch_size = 8 

    print("Loading dev data & BM25 Pre-Retrieved Cache...")
    with open('data/dev.jsonl', 'r') as f:
        dev_data = [json.loads(line) for line in f if line.strip()]
        
    with open('data/bm25_top20_cache.json', 'r') as f:
        bm25_cache = json.load(f)

    print(f"Loading {model_id} in pure 16-bit precision...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_path, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" 

    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        cache_dir=cache_path,
        token=hf_token
    )

    # Master dictionary for all EM/F1 scores
    aggregate_results = {}

    for n in range(1, 11):
        print(f"\n--- Evaluating Baseline RAG (n={n}) ---")
        predictions = []
        run_logs = {} 
        
        for i in tqdm(range(0, len(dev_data), batch_size), desc=f"Processing n={n}"):
            batch = dev_data[i : i + batch_size]
            prompts = []
            contexts_used = [] 
            
            for item in batch:
                question = item['question']
                context_texts = bm25_cache.get(question, [])[:n]
                combined_context = "\n\n".join(context_texts)
                contexts_used.append(combined_context)
                
                # NATIVE CHAT TEMPLATE IMPLEMENTATION
                messages = [
                    {"role": "system", "content": "You are a strict, factual question-answering system. Provide ONLY the exact entity, date, or phrase required to answer the question. DO NOT write full sentences. DO NOT explain your reasoning."},
                    {"role": "user", "content": f"Context:\n{combined_context}\n\nQuestion:\n{question}"}
                ]
                
                prompt_str = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                prompts.append(prompt_str)

            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4000).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=20, 
                    pad_token_id=tokenizer.eos_token_id, 
                    do_sample=False
                )
            
            for j, output in enumerate(outputs):
                gen_text = tokenizer.decode(output[inputs.input_ids.shape[1]:], skip_special_tokens=True)
                pred_answer = gen_text.strip().split('\n')[0]
                predictions.append(pred_answer)
                
                question_text = batch[j]['question']
                ground_truth = batch[j]['answer']
                q_type = batch[j].get('type', 'unknown')
                is_correct = exact_match_score(pred_answer, ground_truth)
                
                run_logs[question_text] = {
                    "type": q_type,
                    "ground_truth": ground_truth,
                    "prediction": pred_answer,
                    "is_correct": is_correct,
                    "context_used": contexts_used[j]
                }

        em_score = sum(exact_match_score(p, g['answer']) for p, g in zip(predictions, dev_data)) / len(dev_data)
        f1_val = sum(f1_score(p, g['answer']) for p, g in zip(predictions, dev_data)) / len(dev_data)
        
        # Log the individual run
        with open(f'full_log_baseline_n{n}.json', 'w') as f:
            json.dump(run_logs, f, indent=4)
            
        # Update the master summary block
        aggregate_results[f"n_{n}"] = {"EM": em_score, "F1": f1_val}
        
        # Save the master block continuously so we don't lose data if a crash happens
        with open('results_baseline_summary.json', 'w') as f:
            json.dump(aggregate_results, f, indent=4)
            
        print(f"Results for n={n} -> EM: {em_score:.4f}, F1: {f1_val:.4f}")

    print("\nAll 10 iterations complete. Summary saved to results_baseline_summary.json")

if __name__ == "__main__":
    main()