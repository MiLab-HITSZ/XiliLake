# Copyright (c) 2026 MiLab. All rights reserved.
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from cdh_caption_claims import build_caption_claim_schema

class CDHBenchLoader:
    def __init__(self, jsonl_path: str):
        self.jsonl_path = jsonl_path
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.jsonl_path):
            raise FileNotFoundError(f"JSONL file not found at {self.jsonl_path}")
        
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
            
    def get_categories(self) -> List[str]:
        return sorted(list(set(item['category'] for item in self.data)))
        
    def get_subcategories(self, category: Optional[str] = None) -> List[str]:
        if category:
            return sorted(list(set(item['subcategory'] for item in self.data if item['category'] == category)))
        return sorted(list(set(item['subcategory'] for item in self.data)))
        
    def filter_by_category(self, category_keyword: str) -> List[Dict[str, Any]]:
        """Filter by category using a keyword like 'Counting', 'Relational', or 'Attribute'."""
        return [item for item in self.data if category_keyword in item['category']]
        
    def filter_by_subcategory(self, subcategory_keyword: str) -> List[Dict[str, Any]]:
        """Filter by subcategory using a keyword."""
        return [item for item in self.data if subcategory_keyword in item['subcategory']]

    def get_qa_format(self, items: List[Dict[str, Any]], type_pref: str = "counterfactual") -> List[Dict[str, Any]]:
        """
        Formats items for Direct QA.
        type_pref: 'counterfactual' or 'commonsense'
        """
        results = []
        for item in items:
            qa_data = item.get('direct_qa', {})
            if qa_data and 'question' in qa_data:
                results.append({
                    "pair_id": item['pair_id'],
                    "category": item['category'],
                    "subcategory": item['subcategory'],
                    "question": qa_data['question'],
                    "answer": qa_data.get(f'{type_pref}_gt'),
                    "prompt": item.get(f'{type_pref}_prompt')
                })
        return results

    def get_caption_format(self, items: List[Dict[str, Any]], type_pref: str = "counterfactual") -> List[Dict[str, Any]]:
        """Formats items for Captioning."""
        results = []
        for item in items:
            caption_data = item.get('captioning', {})
            if caption_data and 'question' in caption_data:
                results.append({
                    "pair_id": item['pair_id'],
                    "category": item['category'],
                    "subcategory": item['subcategory'],
                    "question": caption_data['question'],
                    "answer": caption_data.get(f'{type_pref}_gt'),
                    "claims": build_caption_claim_schema(item),
                    "prompt": item.get(f'{type_pref}_prompt')
                })
        return results

    def get_choice_format(self, items: List[Dict[str, Any]], type_pref: str = "counterfactual") -> List[Dict[str, Any]]:
        """Formats items for Multiple Choice."""
        results = []
        for item in items:
            choice_data = item.get('multiple_choice', {})
            if choice_data and 'question' in choice_data:
                results.append({
                    "pair_id": item['pair_id'],
                    "category": item['category'],
                    "subcategory": item['subcategory'],
                    "question": choice_data['question'],
                    "options": choice_data.get('options', []),
                    "answer": choice_data.get(f'{type_pref}_gt'),
                    "prompt": item.get(f'{type_pref}_prompt')
                })
        return results

if __name__ == "__main__":
    loader = CDHBenchLoader(str(Path(__file__).resolve().parent / 'CDH-Bench.revised.strict.jsonl'))
    
    # Example: Get all counting-related questions in QA format
    counting_items = loader.filter_by_category("Counting")
    qa_counting = loader.get_qa_format(counting_items, type_pref="counterfactual")
    
    print(f"Total Counting items: {len(counting_items)}")
    if qa_counting:
        print(f"Example QA item: {json.dumps(qa_counting[0], indent=2, ensure_ascii=False)}")
    
    # Example: Get all relational questions in caption format
    rel_items = loader.filter_by_category("Relational")
    caption_rel = loader.get_caption_format(rel_items, type_pref="commonsense")
    print(f"Total Relational items: {len(rel_items)}")
    
    # Example: Get all attribute questions in choice format
    attr_items = loader.filter_by_category("Attribute")
    choice_attr = loader.get_choice_format(attr_items)
    print(f"Total Attribute items: {len(attr_items)}")
