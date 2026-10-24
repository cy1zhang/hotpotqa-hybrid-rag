import json
import os
import subprocess
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher

def main():
    print("1. Extracting unique passages from wiki_snippets_dev.json...")
    with open('data/wiki_snippets_dev.json', 'r') as f:
        wiki_snippets = json.load(f)
    
    unique_passages = set()
    for question, passages in wiki_snippets.items():
        for p in passages:
            unique_passages.add(p)
            
    # Convert to list for indexing
    passage_list = list(unique_passages)
    
    print(f"Found {len(passage_list)} unique passages. Creating Pyserini collection...")
    os.makedirs("collection", exist_ok=True)
    
    with open('collection/docs.jsonl', 'w') as f:
        for idx, text in enumerate(passage_list):
            doc = {"id": str(idx), "contents": text}
            f.write(json.dumps(doc) + '\n')
            
    print("2. Building BM25 Index via Pyserini...")
    index_dir = "indexes/wiki_bm25"
    subprocess.run([
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", "collection",
        "--index", index_dir,
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "4",
        "--storePositions", "--storeDocvectors", "--storeRaw"
    ], check=True)
    
    print("3. Querying the Index for the Dev Set...")
    searcher = LuceneSearcher(index_dir)
    
    with open('data/dev.jsonl', 'r') as f:
        dev_data = [json.loads(line) for line in f if line.strip()]
        
    bm25_cache = {}
    
    for item in tqdm(dev_data, desc="Pre-retrieving Top 20"):
        question = item['question']
        hits = searcher.search(question, k=20)
        
        retrieved_texts = []
        for hit in hits:
            # Parse the raw JSON back out of the Lucene document
            raw_doc = json.loads(searcher.doc(hit.docid).raw())
            retrieved_texts.append(raw_doc['contents'])
            
        bm25_cache[question] = retrieved_texts
        
    print("4. Saving cache to data/bm25_top20_cache.json...")
    with open('data/bm25_top20_cache.json', 'w') as f:
        json.dump(bm25_cache, f, indent=4)
        
    print("Done! BM25 Pre-Retrieval complete.")

if __name__ == "__main__":
    main()