import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel
from groq import Groq
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# --- Config ---
BASE_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
ADAPTER_PATH = "./llama-sft-model"
GROQ_EVAL_MODEL = "llama-3.1-8b-instant"
JUDGE_MODEL = "llama-3.3-70b-versatile" # The 70B model acts as our impartial judge

SYSTEM_PROMPT = "You are a senior Indian Tax Law expert. You explain complex legal text clearly, accurately, and with professional reasoning. Always use simple language for the assistant role."

def load_eval_dataset(filepath="eval_sft.jsonl"):
    dataset = []
    if not os.path.exists(filepath):
        print(f"File {filepath} not found.")
        return dataset
        
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                messages = data['messages']
                user_content = next(m['content'] for m in messages if m['role'] == 'user')
                assistant_content = next(m['content'] for m in messages if m['role'] == 'assistant')
                dataset.append({
                    "question": user_content,
                    "ground_truth": assistant_content
                })
            except Exception as e:
                continue
    return dataset

def evaluate_with_llm_judge(client, question, ground_truth, model_answer):
    """Uses a 70B model to score an answer out of 10 for factual strictness and style compared to ground truth."""
    prompt = f"""You are an objective evaluator comparing an AI's answer against a Ground Truth answer.
The Question typically contains a piece of Indian Tax Law text. 
Score the Model Answer from 0 to 10 based on:
1. FACTUAL ACCURACY (Does it correctly interpret the law exactly as the Ground Truth does, without hallucination?)
2. CLARITY/STYLE (Is it easy to understand and does it follow the instructions?)

Question: {question}

---
Ground Truth (Perfect Answer): 
{ground_truth}

---
Model Answer to Evaluate: 
{model_answer}

Provide your evaluation in the following strict format:
SCORE: [0-10]
REASONING: [1 sentence explanation]
"""
    try:
        completion = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150
        )
        result = completion.choices[0].message.content
        
        # Parse score
        score_line = [line for line in result.split('\n') if 'SCORE:' in line][0]
        score = float(score_line.split(':')[1].strip())
        return score, result
    except Exception as e:
        print(f"Error evaluating: {e}")
        return 0.0, "Error"

def main():
    print("="*60)
    print("🚀 AUTOMATED NUMERIC EVALUATION (28-SAMPLE DATASET)")
    print("="*60)
    
    # Load Dataset
    eval_dataset = load_eval_dataset("eval_sft.jsonl")
    print(f"Loaded {len(eval_dataset)} evaluation samples.")
    if not eval_dataset:
        return

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    # Load Local Model
    print("\nLoading Local Fine-Tuned Model (GPU)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    # Supress deprecation warnings
    import transformers
    transformers.logging.set_verbosity_error()
    
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    
    text_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

    base_total = 0
    ft_total = 0

    print("\nStarting evaluation...")
    for i, item in enumerate(eval_dataset):
        q = item["question"]
        gt = item["ground_truth"]
        
        # Print short version of question
        short_q = q.split('\n')[0][:50] + "..." if len(q) > 50 else q.split('\n')[0]
        print(f"\n[{i+1}/{len(eval_dataset)}] {short_q}")
        
        # --- Generate Base ---
        try:
            base_completion = client.chat.completions.create(
                model=GROQ_EVAL_MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}],
                temperature=0.1, max_tokens=256
            )
            base_ans = base_completion.choices[0].message.content
        except Exception as e:
            print(f"Error with Base API: {e}")
            base_ans = "Error"
            
        # --- Generate FT ---
        try:
            conversation = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}]
            prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            ft_ans = text_pipeline(prompt, max_new_tokens=256, temperature=0.1, do_sample=False, pad_token_id=tokenizer.eos_token_id, return_full_text=False)[0]['generated_text']
        except Exception as e:
            print(f"Error with Local Model: {e}")
            ft_ans = "Error"

        # --- Judge ---
        base_score, _ = evaluate_with_llm_judge(client, q, gt, base_ans)
        ft_score, _ = evaluate_with_llm_judge(client, q, gt, ft_ans)
        
        base_total += base_score
        ft_total += ft_score
        
        print(f"  Base Score: {base_score}/10")
        print(f"  FT Score  : {ft_score}/10")
        
    print("\n" + "="*60)
    print("📊 FINAL AUTOMATED RESULTS")
    print("="*60)
    
    avg_base = (base_total / (len(eval_dataset) * 10)) * 100
    avg_ft = (ft_total / (len(eval_dataset) * 10)) * 100
    
    print(f"Total Samples Evaluated: {len(eval_dataset)}")
    print(f"Base Model Accuracy: {avg_base:.1f}%")
    print(f"Fine-Tuned Accuracy: {avg_ft:.1f}%")
    
    if avg_base > 0:
        diff = ((avg_ft - avg_base) / avg_base) * 100
        if diff > 0:
            print(f"\n🔥 RESULT: Fine-Tuned Model is {diff:.1f}% BETTER than Base Model.")
        else:
            print(f"\n🚨 RESULT: Fine-Tuned Model is {abs(diff):.1f}% WORSE than Base Model.")

if __name__ == "__main__":
    main()
