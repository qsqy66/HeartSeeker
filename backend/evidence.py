"""Deterministic, GRADE-inspired evidence appraisal.

This is not a systematic review and it does not replace GRADE reviewers.  It
reads study text and flags the signals that commonly drive a certainty rating:

* study design → starting certainty
* risk of bias (randomisation, allocation concealment, blinding, ITT, attrition)
* inconsistency (heterogeneity in meta-analyses)
* indirectness (animal/in-vitro data, surrogate endpoints)
* imprecision (small sample, wide confidence interval)
* publication bias (funnel plot / Egger only when reported)

Every domain records the matched excerpts so the appraisal is auditable and can
be shown next to the source text.  The output is deliberately conservative:
unknown domains are reported as "not assessable" rather than assumed adequate.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, Optional

CERTAINTY_LABELS = {
    "high": "高",
    "moderate": "中",
    "low": "低",
    "very_low": "极低",
}
CERTAINTY_STARS = {"high": 4, "moderate": 3, "low": 2, "very_low": 1}
STARTING = {"high": 4, "moderate": 3, "low": 2, "very_low": 1}

DESIGN_PATTERNS: list[tuple[str, str, str, list[str]]] = [
    ("systematic_review", "系统综述 / Meta 分析", "high", [
        r"systematic\s+review", r"meta[- ]?analys(?:is|es)", r"系统综述", r"荟萃分析",
        r"pooled\s+analysis",
    ]),
    ("rct", "随机对照试验", "high", [
        r"randomi[sz]ed\s+controlled\s+trial", r"randomi[sz]ed\s+trial",
        r"randomi[sz]ed\b[^.\n]{0,80}?trial", r"trial\b[^.\n]{0,80}?randomi[sz]ed",
        r"\bRCT\b", r"随机对照", r"随机分组",
    ]),
    ("cohort", "队列研究", "low", [
        r"prospective\s+cohort", r"retrospective\s+cohort", r"\bcohort\s+study\b",
        r"\bcohort\b", r"队列研究",
    ]),
    ("case_control", "病例对照研究", "low", [
        r"case[- ]control", r"病例对照",
    ]),
    ("cross_sectional", "横断面研究", "low", [
        r"cross[- ]sectional", r"横断面",
    ]),
    ("case_series", "病例系列 / 病例报告", "very_low", [
        r"case\s+series", r"case\s+report", r"病例系列", r"病例报告",
    ]),
    ("preclinical", "动物 / 体外研究", "very_low", [
        r"\bin\s+vivo\b", r"\bin\s+vitro\b", r"\bmice\b", r"\bmouse\b", r"\bmurine\b",
        r"\brats?\b", r"cell\s+line", r"动物实验", r"细胞实验",
    ]),
]

SAMPLE_RE = re.compile(
    r"(?:(?:n\s*=\s*|sample\s+size\s+(?:of\s+)?|共(?:纳入|入组|收集)?\s*))?"
    r"(\d{1,3}(?:,\d{3})+|\d+)\s*"
    r"(?:patients?|participants?|subjects?|adults?|children|infants?|"
    r"individuals?|cases?|例|名|人)",
    re.IGNORECASE,
)
CI_RE = re.compile(
    r"95%\s*(?:CI|confidence\s+interval)[^0-9\-]{0,12}"
    r"(-?\d+(?:\.\d+)?)\s*(?:to|–|-|,)\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
EFFECT_RE = re.compile(
    r"\b(?:HR|hazard\s+ratio|OR|odds\s+ratio|RR|relative\s+risk)\b"
    r"[^0-9\-]{0,12}(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
LOSS_RE = re.compile(
    r"(?:lost\s+to\s+follow[- ]?up|withdrew|withdrawal|失访|退出)[^.%]{0,40}?"
    r"(\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)
HETEROGENEITY_RE = re.compile(r"I\s*(?:²|2|\^2)\s*[=:]?\s*(\d+(?:\.\d+)?)\s*%",
                              re.IGNORECASE)


def _excerpt(text: str, start: int, end: int, width: int = 70) -> str:
    left = max(0, start - width)
    right = min(len(text), end + width)
    snippet = re.sub(r"\s+", " ", text[left:right]).strip()
    return ("…" if left > 0 else "") + snippet + ("…" if right < len(text) else "")


def _matches(text: str, patterns: list[str], limit: int = 3) -> list[str]:
    found: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            found.append(_excerpt(text, match.start(), match.end()))
            if len(found) >= limit:
                return found
    return found


def _detect_design(text: str) -> tuple[str, str, str, list[str]]:
    for design, label, starting, patterns in DESIGN_PATTERNS:
        signals = _matches(text, patterns)
        if signals:
            return design, label, starting, signals
    return "unknown", "无法从文本判定研究设计", "low", []


def _has(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text, re.IGNORECASE))


def _domain(domain: str, label: str, concern: str, reasons: list[str],
            downgrade: int = 0, signals: Optional[list[str]] = None) -> dict[str, Any]:
    return {
        "domain": domain, "label": label, "concern": concern,
        "downgrade": downgrade, "reasons": reasons, "signals": signals or [],
    }


def _risk_of_bias(text: str, design: str) -> dict[str, Any]:
    reasons: list[str] = []
    signals: list[str] = []
    downgrade = 0
    if design == "systematic_review":
        risk_tool = _matches(text, [r"risk\s+of\s+bias", r"\bRoB\s*2?\b",
                                     r"Newcastle[- ]Ottawa", r"偏倚风险"])
        if risk_tool:
            signals.extend(risk_tool)
            reasons.append("报告了对纳入研究的偏倚风险评估。")
        else:
            reasons.append("未报告纳入研究的偏倚风险评估。")
            downgrade += 1
        search = _matches(text, [r"PubMed", r"Embase", r"Cochrane", r"search\s+strategy",
                                  r"systematic\s+search", r"检索策略"], limit=2)
        if search:
            signals.extend(search)
            reasons.append("报告了系统检索来源或检索策略。")
        else:
            reasons.append("未报告系统检索策略，可能遗漏研究。")
            downgrade += 1
        registered = _matches(text, [r"PROSPERO", r"registered\s+protocol", r"研究注册"],
                              limit=1)
        if registered:
            signals.extend(registered)
            reasons.append("报告了研究方案注册。")
    elif design == "rct":
        random_signals = _matches(text, [r"randomi[sz]ed", r"随机"])
        if random_signals:
            signals.extend(random_signals)
            reasons.append("提及随机分组。")
        else:
            reasons.append("未在文本中找到随机分组描述。")
            downgrade += 1

        concealment = _matches(text, [r"allocation\s+conceal", r"sealed\s+envelope",
                                      r"central\s+randomi[sz]ation", r"分配隐藏"])
        if concealment:
            signals.extend(concealment)
            reasons.append("提及分配隐藏或中心随机。")
        else:
            reasons.append("未报告分配隐藏，存在选择偏倚风险。")
            downgrade += 1

        blinding = _matches(text, [r"double[- ]blind", r"single[- ]blind", r"placebo",
                                    r"blinded", r"盲法", r"双盲", r"单盲"])
        if blinding:
            signals.extend(blinding)
            reasons.append("报告了盲法或安慰剂对照。")
        else:
            reasons.append("未报告盲法；对主观终点需谨慎。")
            downgrade += 1

        itt = _matches(text, [r"intention[- ]to[- ]treat", r"\bITT\b", r"意向性治疗"])
        if itt:
            signals.extend(itt)
            reasons.append("采用意向性治疗分析。")
        else:
            reasons.append("未报告意向性治疗分析。")
    elif design in {"cohort", "case_control", "cross_sectional"}:
        adjustment = _matches(text, [r"adjusted\s+for", r"multivariable", r"multivariate",
                                      r"propensity[- ]score", r"多因素", r"校正", r"调整"])
        if adjustment:
            signals.extend(adjustment)
            reasons.append("报告了混杂因素校正。")
        else:
            reasons.append("未报告混杂因素校正，残余混杂可能较大。")
            downgrade += 1
        consecutive = _matches(text, [r"consecutive", r"population[- ]based", r"nationwide",
                                       r"registry", r"连续", r"全国", r"登记"])
        if consecutive:
            signals.extend(consecutive)
            reasons.append("样本来源具有一定的连续性或人群代表性。")
        else:
            reasons.append("未说明是否为连续/人群为基础抽样，选择偏倚难以排除。")
    elif design == "preclinical":
        reasons.append("动物或体外研究不能直接外推到患者。")
        downgrade += 2
    else:
        reasons.append("研究设计不明确，无法充分评估偏倚风险。")

    loss = LOSS_RE.search(text)
    if loss:
        percentage = float(loss.group(1))
        signals.append(_excerpt(text, loss.start(), loss.end()))
        if percentage > 20:
            reasons.append(f"失访/退出比例较高（{percentage:g}%）。")
            downgrade += 1
        elif percentage > 10:
            reasons.append(f"失访/退出比例偏高（{percentage:g}%）。")
        else:
            reasons.append(f"失访/退出比例可接受（{percentage:g}%）。")

    if downgrade >= 2:
        concern = "serious"
    elif downgrade == 1:
        concern = "some"
    else:
        concern = "low"
    return _domain("risk_of_bias", "偏倚风险", concern, reasons, min(downgrade, 2), signals)


def _inconsistency(text: str, design: str) -> dict[str, Any]:
    if design != "systematic_review":
        return _domain("inconsistency", "不一致性", "not_assessable",
                       ["单一研究无法评价研究间不一致性。"],
                       signals=[])
    reasons: list[str] = []
    signals: list[str] = []
    downgrade = 0
    heterogeneity = HETEROGENEITY_RE.search(text)
    if heterogeneity:
        value = float(heterogeneity.group(1))
        signals.append(_excerpt(text, heterogeneity.start(), heterogeneity.end()))
        if value >= 75:
            reasons.append(f"I² 很高（{value:g}%），研究间异质性明显。")
            downgrade = 2
        elif value >= 50:
            reasons.append(f"I² 较高（{value:g}%）。")
            downgrade = 1
        else:
            reasons.append(f"I² 为 {value:g}%，异质性可接受。")
    elif _has(text, r"heterogeneity|异质性"):
        reasons.append("提及异质性但未给出可解析的 I² 数值。")
        downgrade = 1
    else:
        reasons.append("未报告异质性指标。")
    text_signal = _matches(text, [r"subgroup\s+analysis", r"random[- ]effects",
                                   r"fixed[- ]effects", r"亚组分析"], limit=2)
    signals.extend(text_signal)
    concern = "serious" if downgrade >= 2 else "some" if downgrade else "low"
    return _domain("inconsistency", "不一致性", concern, reasons, downgrade, signals)


def _indirectness(text: str, design: str) -> dict[str, Any]:
    reasons: list[str] = []
    signals: list[str] = []
    downgrade = 0
    if design == "preclinical":
        reasons.append("证据来自动物/体外实验，与临床问题存在间接性。")
        downgrade = 2
    surrogate = _matches(text, [r"surrogate\s+(?:endpoint|marker|outcome)", r"替代终点",
                                 r"biomarker\s+endpoint"])
    if surrogate:
        signals.extend(surrogate)
        reasons.append("使用了替代终点，需确认其与患者重要结局的关系。")
        downgrade = max(downgrade, 1)
    single_center = _matches(text, [r"single[- ]cent(?:er|re)", r"单中心"])
    if single_center:
        signals.extend(single_center)
        reasons.append("单中心研究可能限制外推性。")
        downgrade = max(downgrade, 1)
    if not reasons:
        reasons.append("未发现明显的间接性信号。")
    concern = "serious" if downgrade >= 2 else "some" if downgrade else "low"
    return _domain("indirectness", "间接性", concern, reasons, downgrade, signals)


def _sample_size(text: str) -> Optional[int]:
    values = [int(match.replace(",", "")) for match in SAMPLE_RE.findall(text)]
    # Ignore years and tiny numbers that are unlikely to be a trial sample size.
    values = [value for value in values if 10 <= value <= 10_000_000]
    return max(values) if values else None


def _imprecision(text: str) -> tuple[dict[str, Any], Optional[int]]:
    reasons: list[str] = []
    signals: list[str] = []
    downgrade = 0
    sample = _sample_size(text)
    if sample is None:
        reasons.append("未找到可解析的样本量。")
    elif sample < 100:
        reasons.append(f"样本量较小（n={sample}）。")
        downgrade = 1
    else:
        reasons.append(f"样本量 n={sample}。")

    ci = CI_RE.search(text)
    if ci:
        low, high = float(ci.group(1)), float(ci.group(2))
        signals.append(_excerpt(text, ci.start(), ci.end()))
        if low < 0 < high:
            reasons.append("置信区间跨过无效值 0。")
        if low != 0 and high != 0:
            ratio = max(abs(low), abs(high)) / max(1e-9, min(abs(low), abs(high)))
            if ratio > 2:
                reasons.append("置信区间较宽，精确性有限。")
                downgrade = max(downgrade, 1)
    else:
        reasons.append("未报告可解析的 95% 置信区间。")

    concern = "serious" if downgrade >= 1 and sample is not None and sample < 100 else (
        "some" if downgrade else "low")
    return _domain("imprecision", "精确性", concern, reasons, downgrade, signals), sample


def _publication_bias(text: str, design: str) -> dict[str, Any]:
    if design != "systematic_review":
        return _domain("publication_bias", "发表偏倚", "not_assessable",
                       ["发表偏倚通常只能在系统综述层面评价。"])
    signals = _matches(text, [r"funnel\s+plot", r"Egger", r"Begg", r"publication\s+bias",
                              r"发表偏倚", r"漏斗图"], limit=3)
    if signals:
        return _domain("publication_bias", "发表偏倚", "low",
                       ["报告了发表偏倚检查。"], signals=signals)
    return _domain("publication_bias", "发表偏倚", "not_assessable",
                   ["未报告漏斗图或 Egger 检验，无法评价发表偏倚。"])


def _upgrades(text: str) -> list[dict[str, str]]:
    upgrades: list[dict[str, str]] = []
    effect = EFFECT_RE.search(text)
    if effect:
        value = abs(float(effect.group(1)))
        if value >= 5 or 0 < value <= 0.2:
            upgrades.append({"label": "效应量大",
                             "reason": f"报告的点估计为 {value:g}，提示很大的效应。",
                             "signal": _excerpt(text, effect.start(), effect.end())})
    dose = _matches(text, [r"dose[- ]response", r"剂量反应"], limit=1)
    if dose:
        upgrades.append({"label": "剂量-反应关系", "reason": "报告了剂量-反应关系。",
                         "signal": dose[0]})
    return upgrades


def appraise(text: str, title: str = "") -> dict[str, Any]:
    """Return a deterministic GRADE-style appraisal of one study or review."""
    text = (text or "").strip()
    if len(text) < 40:
        raise ValueError("文本过短，无法评价研究证据")
    design, design_label, starting, design_signals = _detect_design(text)
    domains = [
        _risk_of_bias(text, design),
        _inconsistency(text, design),
        _indirectness(text, design),
    ]
    imprecision, sample = _imprecision(text)
    domains.append(imprecision)
    domains.append(_publication_bias(text, design))
    upgrades = _upgrades(text)

    downgrades = sum(domain["downgrade"] for domain in domains)
    score = STARTING[starting] - downgrades + len(upgrades)
    # Large-effect and dose-response upgrades are capped by GRADE practice; a
    # heuristic never raises above "high" or above one upgrade for RCTs.
    score = max(1, min(4, score))
    certainty = {4: "high", 3: "moderate", 2: "low", 1: "very_low"}[score]

    serious = [domain["label"] for domain in domains
               if domain["concern"] in {"some", "serious"}]
    if serious:
        summary = (f"按 GRADE 启发式评价，该证据的总体确定性为"
                   f"「{CERTAINTY_LABELS[certainty]}」。主要降级因素："
                   + "、".join(serious) + "。")
    else:
        summary = (f"按 GRADE 启发式评价，该证据的总体确定性为"
                   f"「{CERTAINTY_LABELS[certainty]}」，未发现明显的降级因素。")
    if upgrades:
        summary += " 可能的升级因素：" + "、".join(item["label"] for item in upgrades) + "。"

    return {
        "title": title,
        "design": {"type": design, "label": design_label,
                   "starting_certainty": starting,
                   "starting_label": CERTAINTY_LABELS[starting],
                   "signals": design_signals},
        "certainty": certainty,
        "certainty_label": CERTAINTY_LABELS[certainty],
        "stars": CERTAINTY_STARS[certainty],
        "downgrades": downgrades,
        "upgrades": upgrades,
        "domains": domains,
        "sample_size": sample,
        "summary": summary,
        "note": "结果为规则化启发式评价，不能替代系统评价或 GRADE 专家组判断。",
    }


def appraise_paper(paper_id: int, db_path: Path | str) -> dict[str, Any]:
    """Appraise the text of an imported paper from the local database."""
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        paper = connection.execute(
            "SELECT id, title FROM papers WHERE id=?", (int(paper_id),)).fetchone()
        if paper is None:
            raise LookupError(f"论文 {paper_id} 不存在")
        rows = connection.execute(
            "SELECT c.text, c.heading, c.section FROM chunks c "
            "JOIN pages p ON p.id=c.page_id WHERE c.paper_id=? "
            "ORDER BY p.page_number, c.id", (int(paper_id),)).fetchall()
    finally:
        connection.close()
    if not rows:
        raise ValueError("该论文没有可评价的文本")
    parts: list[str] = []
    total = 0
    for row in rows:
        text = (row["text"] or "").strip()
        if not text:
            continue
        parts.append(text)
        total += len(text)
        if total >= 24000:
            break
    result = appraise("\n".join(parts), title=paper["title"])
    result["paper_id"] = int(paper_id)
    return result
