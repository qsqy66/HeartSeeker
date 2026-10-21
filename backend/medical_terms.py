"""Small, curated bilingual medical terminology expansion.

This is not a medical ontology.  It is a deterministic local aid for the most
common abbreviations and Chinese/English equivalents seen in clinical papers,
so keyword retrieval does not miss a paper that says "myocardial infarction"
when the student types "心梗" (or vice versa).
"""
from __future__ import annotations

import re

# Keys are matched case-sensitively when they are short ASCII abbreviations,
# and as substrings when they contain CJK characters.
SYNONYMS: dict[str, list[str]] = {
    "MI": ["myocardial infarction", "心梗", "心肌梗死"],
    "ACS": ["acute coronary syndrome", "急性冠脉综合征"],
    "HF": ["heart failure", "心力衰竭", "心衰"],
    "HFrEF": ["heart failure with reduced ejection fraction", "射血分数降低的心力衰竭"],
    "HFpEF": ["heart failure with preserved ejection fraction", "射血分数保留的心力衰竭"],
    "AF": ["atrial fibrillation", "心房颤动", "房颤"],
    "COPD": ["chronic obstructive pulmonary disease", "慢性阻塞性肺疾病", "慢阻肺"],
    "CKD": ["chronic kidney disease", "慢性肾脏病"],
    "CVD": ["cardiovascular disease", "心血管疾病"],
    "RCT": ["randomized controlled trial", "随机对照试验"],
    "NNT": ["number needed to treat", "需治疗人数"],
    "ARR": ["absolute risk reduction", "绝对风险降低"],
    "HR": ["hazard ratio", "风险比"],
    "OR": ["odds ratio", "比值比"],
    "CI": ["confidence interval", "置信区间"],
    "心衰": ["heart failure", "心力衰竭"],
    "心力衰竭": ["heart failure", "心衰"],
    "心梗": ["myocardial infarction", "急性心肌梗死"],
    "心肌梗死": ["myocardial infarction", "心梗"],
    "高血压": ["hypertension", "血压升高"],
    "糖尿病": ["diabetes mellitus", "diabetes"],
    "房颤": ["atrial fibrillation", "心房颤动"],
    "慢阻肺": ["chronic obstructive pulmonary disease", "COPD"],
    "卒中": ["stroke", "脑卒中", "中风"],
    "中风": ["stroke", "卒中", "脑卒中"],
    "肺炎": ["pneumonia"],
    "哮喘": ["asthma"],
    "肾病": ["kidney disease", "renal disease"],
    "随机对照试验": ["randomized controlled trial", "RCT"],
    "系统综述": ["systematic review"],
    "荟萃分析": ["meta-analysis", "meta analysis"],
}

_CJK_RE = re.compile(r"[㐀-鿿]")
_ASCII_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")


def expand_query(query: str) -> list[str]:
    """Return additional search terms for a query, without duplicates."""
    query = (query or "").strip()
    if not query:
        return []
    original = query.lower()
    extras: list[str] = []
    upper_tokens = set(_ASCII_WORD_RE.findall(query.upper()))
    for key, values in SYNONYMS.items():
        matched = False
        if _CJK_RE.search(key):
            matched = key in query
        elif key.isupper():
            matched = key in upper_tokens
        else:
            matched = bool(re.search(rf"\b{re.escape(key)}\b", query, re.IGNORECASE))
        if not matched:
            continue
        for value in values:
            if value.lower() not in original and value not in extras:
                extras.append(value)
    return extras
