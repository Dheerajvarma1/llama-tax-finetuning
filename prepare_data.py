import os
import json
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

import re

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def convert_to_instruction_format(text):
    # Improved splitting: Split by Section headers (e.g., "\n1. ", "\n2. ")
    # This regex looks for a digit at the start of a line followed by a dot.
    sections = re.split(r'\n(?=\d+\.\s)', text)
    
    # Further split sections that are too large (e.g., over 2000 chars)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    
    final_chunks = []
    for section in sections:
        if len(section) > 2500:
            final_chunks.extend(text_splitter.split_text(section))
        else:
            final_chunks.append(section.strip())
    
    dataset = []
    for chunk in final_chunks:
        if len(chunk) < 50: continue # Skip very small artifacts
        
        # Initial format (will be expanded by sft_generator.py)
        entry = {
            "content": chunk
        }
        dataset.append(entry)
    return dataset

def save_dataset(dataset, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    pdf_path = "income-tax.pdf"
    output_file = "raw_chunks.jsonl" # Renamed to show it's the raw stage
    
    if os.path.exists(pdf_path):
        print(f"Extracting text from {pdf_path}...")
        text = extract_text_from_pdf(pdf_path)
        print("Formatting dataset (splitting by section)...")
        dataset = convert_to_instruction_format(text)
        print(f"Saving {len(dataset)} raw chunks to {output_file}...")
        save_dataset(dataset, output_file)
        print("Done! Next step: Run sft_generator.py to augment this data.")
    else:
        print(f"File {pdf_path} not found.")
