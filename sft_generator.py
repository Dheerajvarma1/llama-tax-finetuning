import json
import os
import random
import time
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Switch back to Groq (Free/High-rate alternative)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Expert Task Distribution & Styles
TASKS = {
    "explain": {
        "weight": 0.40,
        "templates": [
            "Explain this legal section in plain English for a non-expert:",
            "What does the following law mean in simple terms?",
            "Provide a clear explanation of this tax provision:"
        ],
        "style": "Provide a clear, conversational paragraph explanation."
    },
    "summarize": {
        "weight": 0.20,
        "templates": [
            "Summarize the key tax implications of the following clause:",
            "What is the main point of this legal section?"
        ],
        "style": "Provide a concise summary using bullet points."
    },
    "key_points": {
        "weight": 0.15,
        "templates": [
            "Extract the most important points from this tax law section:",
            "List the critical requirements mentioned below:"
        ],
        "style": "Provide a numbered list of the most important takeaways."
    },
    "qa": {
        "weight": 0.15,
        "templates": [
            "What is the core requirement described here?",
            "Based on the text below, answer: What are the conditions for this provision?"
        ],
        "style": "Provide a direct, accurate answer based strictly on the text."
    },
    "interpretation": {
        "weight": 0.10,
        "templates": [
            "Interpret the following legal text for a taxpayer:",
            "Explain the reasoning behind this specific provision:"
        ],
        "style": "Provide a detailed interpretation focusing on the 'why' and the practical impact."
    }
}

SYSTEM_PROMPT = "You are a senior Indian Tax Law expert. You explain complex legal text clearly, accurately, and with professional reasoning. Always use simple language for the assistant role."

def add_noise(text):
    """Simulates messy user queries (typos, ambiguous phrasing) for 10% of samples."""
    if random.random() < 0.10:
        text = text.lower()
        text = text.replace("income tax", "incm tax").replace("section", "sec")
        if random.random() < 0.5:
            text += " (pls explain simply)"
    return text

def get_processed_chunks(files):
    processed = set()
    for f_path in files:
        if os.path.exists(f_path):
            with open(f_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        user_content = data['messages'][1]['content']
                        match = re.search(r'TEXT:\n(.*)', user_content, re.DOTALL)
                        if match:
                            processed.add(match.group(1).strip())
                    except: continue
    return processed

def generate_sft_data(input_file, train_out, eval_out, max_samples=None):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    processed_chunks = get_processed_chunks([train_out, eval_out])
    print(f"🔄 Resuming: Found {len(processed_chunks)} already processed chunks.")

    raw_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            if entry['content'].strip() not in processed_chunks:
                raw_data.append(entry)

    if max_samples:
        raw_data = raw_data[:max_samples]
    
    print(f"🚀 To process: {len(raw_data)} new chunks.")
    random.shuffle(raw_data)
    
    for i, entry in enumerate(raw_data):
        content = entry['content']
        task_name = random.choices(list(TASKS.keys()), weights=[t['weight'] for t in TASKS.values()])[0]
        task = TASKS[task_name]
        user_instruction = random.choice(task['templates'])
        user_query = add_noise(f"{user_instruction}\n\nTEXT:\n{content}")
        
        # Determine split (90% Train, 10% Eval) on-the-fly
        is_eval = random.random() < 0.10
        output_file = eval_out if is_eval else train_out
        label = "EVAL" if is_eval else "TRAIN"

        try:
            print(f"[{label}] [{i+1}/{len(raw_data)}] Task: {task_name}...")
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": f"{SYSTEM_PROMPT} Your response style should be: {task['style']}"},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.6,
                max_tokens=1024
            )
            explanation = completion.choices[0].message.content.strip()
            if len(explanation) < 50: continue

            sft_entry = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": explanation}
                ]
            }
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(sft_entry) + '\n')
            
            time.sleep(1.5) # Gentle rate handling
        except Exception as e:
            print(f"❌ Error: {e}. Sleeping 10s...")
            time.sleep(10)
            continue

if __name__ == "__main__":
    generate_sft_data("raw_chunks.jsonl", "train_sft.jsonl", "eval_sft.jsonl", max_samples=2100)
