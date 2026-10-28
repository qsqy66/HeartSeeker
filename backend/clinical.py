"""Deterministic bedside/clinical calculators.

These are local, auditable formulas used as tools by the agent and exposed over
the API.  They do **not** produce treatment recommendations: each result is a
score with its published risk stratification, plus a reminder that scores are
decision aids, not diagnoses.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

DISCLAIMER = "评分为临床决策辅助工具，不能替代完整病史、查体与医师判断。"


class CalculatorError(ValueError):
    """Raised when a calculator id or input is invalid."""


@dataclass(frozen=True)
class InputSpec:
    key: str
    label: str
    kind: str = "bool"  # bool | number | choice
    default: Any = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    unit: str = ""
    options: tuple[tuple[str, str], ...] = ()
    help: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "label": self.label, "kind": self.kind,
            "default": self.default, "min": self.minimum, "max": self.maximum,
            "unit": self.unit,
            "options": [{"value": value, "label": label} for value, label in self.options],
            "help": self.help,
        }


@dataclass
class CalculatorResult:
    calculator: str
    name: str
    score: float
    unit: str
    risk_level: str  # low | moderate | high | very_high
    interpretation: str
    criteria: list[dict[str, Any]] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    reference: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "calculator": self.calculator,
            "name": self.name,
            "score": round(self.score, 3),
            "unit": self.unit,
            "risk_level": self.risk_level,
            "interpretation": self.interpretation,
            "criteria": self.criteria,
            "details": self.details,
            "notes": [*self.notes, DISCLAIMER],
            "reference": self.reference,
        }


@dataclass(frozen=True)
class Calculator:
    key: str
    name: str
    description: str
    inputs: tuple[InputSpec, ...]
    compute: Callable[[dict[str, Any]], CalculatorResult]
    reference: str
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "name": self.name, "description": self.description,
            "inputs": [item.as_dict() for item in self.inputs],
            "reference": self.reference, "notes": list(self.notes),
        }


# --- input coercion ---------------------------------------------------------------

_TRUE = {"1", "true", "yes", "y", "是", "有", "positive"}
_FALSE = {"0", "false", "no", "n", "否", "无", "negative", ""}


def _as_bool(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise CalculatorError(f"{label} 需要布尔值（是/否）")


def _as_number(value: Any, spec: InputSpec) -> float:
    if value is None or value == "":
        if spec.default is not None:
            value = spec.default
        else:
            raise CalculatorError(f"缺少参数：{spec.label}")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise CalculatorError(f"{spec.label} 需要数字") from error
    if not math.isfinite(number):
        raise CalculatorError(f"{spec.label} 必须是有限数字")
    if spec.minimum is not None and number < spec.minimum:
        raise CalculatorError(f"{spec.label} 不能小于 {spec.minimum:g}")
    if spec.maximum is not None and number > spec.maximum:
        raise CalculatorError(f"{spec.label} 不能大于 {spec.maximum:g}")
    return number


def _as_choice(value: Any, spec: InputSpec) -> str:
    if value is None or value == "":
        if spec.default is not None:
            value = spec.default
        else:
            raise CalculatorError(f"缺少参数：{spec.label}")
    text = str(value).strip().lower()
    allowed = {item[0] for item in spec.options}
    if text not in allowed:
        raise CalculatorError(f"{spec.label} 必须是：{', '.join(sorted(allowed))}")
    return text


def _normalize(inputs: dict[str, Any], specs: tuple[InputSpec, ...]) -> dict[str, Any]:
    if not isinstance(inputs, dict):
        raise CalculatorError("inputs 必须是对象")
    unknown = set(inputs) - {spec.key for spec in specs}
    if unknown:
        raise CalculatorError(f"未知参数：{', '.join(sorted(unknown))}")
    values: dict[str, Any] = {}
    for spec in specs:
        raw = inputs.get(spec.key)
        if spec.kind == "bool":
            if raw is None:
                values[spec.key] = bool(spec.default)
            else:
                values[spec.key] = _as_bool(raw, spec.label)
        elif spec.kind == "number":
            values[spec.key] = _as_number(raw, spec)
        else:
            values[spec.key] = _as_choice(raw, spec)
    return values


def _criterion(key: str, label: str, points: float, met: bool,
               value: Any = None) -> dict[str, Any]:
    result = {"key": key, "label": label, "points": points, "met": bool(met)}
    if value is not None:
        result["value"] = value
    return result


# --- calculators --------------------------------------------------------------------

def _cha2ds2_vasc(inputs: dict[str, Any]) -> CalculatorResult:
    values = _normalize(inputs, _CHA2DS2_INPUTS)
    female = values["sex"] == "female"
    criteria = [
        _criterion("chf", "充血性心力衰竭 / 左室功能不全", 1, values["heart_failure"]),
        _criterion("hypertension", "高血压", 1, values["hypertension"]),
        _criterion("age75", "年龄 ≥75 岁", 2, values["age"] >= 75, values["age"]),
        _criterion("diabetes", "糖尿病", 1, values["diabetes"]),
        _criterion("stroke", "卒中 / TIA / 血栓栓塞史", 2, values["stroke_history"]),
        _criterion("vascular", "血管疾病（心肌梗死、外周动脉病、主动脉斑块）", 1,
                   values["vascular_disease"]),
        _criterion("age65_74", "年龄 65–74 岁", 1, 65 <= values["age"] < 75, values["age"]),
        _criterion("sex", "女性", 1, female),
    ]
    score = sum(item["points"] for item in criteria if item["met"])
    if female:
        low, moderate = 1, 2
    else:
        low, moderate = 0, 1
    if score <= low:
        level, interpretation = "low", "血栓栓塞风险低，一般可暂不抗凝（结合出血风险与患者意愿）。"
    elif score == moderate:
        level, interpretation = "moderate", "血栓栓塞风险中等，应考虑口服抗凝，需结合出血风险。"
    else:
        level, interpretation = "high", "血栓栓塞风险高，通常建议口服抗凝（需评估出血风险与禁忌）。"
    if female and score == 1:
        interpretation += " 女性性别本身不构成抗凝指征。"
    return CalculatorResult(
        calculator="cha2ds2_vasc", name="CHA₂DS₂-VASc 卒中风险",
        score=score, unit="分", risk_level=level, interpretation=interpretation,
        criteria=criteria, details={"sex": values["sex"], "age": values["age"]},
        notes=list(_CHA2DS2_NOTES), reference=_CHA2DS2_REFERENCE,
    )


_CHA2DS2_INPUTS = (
    InputSpec("heart_failure", "充血性心力衰竭 / 左室功能不全", default=False),
    InputSpec("hypertension", "高血压", default=False),
    InputSpec("age", "年龄", kind="number", minimum=18, maximum=120, unit="岁"),
    InputSpec("diabetes", "糖尿病", default=False),
    InputSpec("stroke_history", "卒中 / TIA / 血栓栓塞史", default=False),
    InputSpec("vascular_disease", "血管疾病", default=False),
    InputSpec("sex", "性别", kind="choice", default="male",
              options=(("male", "男"), ("female", "女"))),
)
_CHA2DS2_NOTES = (
    "房颤患者卒中风险评估；女性性别仅在合并其他危险因素时计分。",
    "评分不能单独决定是否抗凝，需同时评估 HAS-BLED 与患者意愿。",
)
_CHA2DS2_REFERENCE = "2019 AHA/ACC/HRS 房颤管理指南；ESC 2020 房颤指南"


def _has_bled(inputs: dict[str, Any]) -> CalculatorResult:
    values = _normalize(inputs, _HAS_BLED_INPUTS)
    criteria = [
        _criterion("hypertension", "高血压（收缩压 >160 mmHg）", 1, values["hypertension"]),
        _criterion("renal", "肾功能异常", 1, values["renal_disease"]),
        _criterion("liver", "肝功能异常", 1, values["liver_disease"]),
        _criterion("stroke", "卒中史", 1, values["stroke_history"]),
        _criterion("bleeding", "出血史或出血倾向", 1, values["bleeding_history"]),
        _criterion("labile_inr", "INR 不稳定（仅在用华法林时）", 1, values["labile_inr"]),
        _criterion("elderly", "年龄 >65 岁", 1, values["age"] > 65, values["age"]),
        _criterion("drugs", "合并增加出血风险的药物", 1, values["drugs"]),
        _criterion("alcohol", "酒精过量", 1, values["alcohol"]),
    ]
    score = sum(item["points"] for item in criteria if item["met"])
    if score <= 2:
        level, interpretation = "low", "出血风险较低，可考虑抗凝；仍需纠正可逆危险因素。"
    else:
        level, interpretation = "high", "出血风险高（≥3 分），应纠正可逆因素并加强监测，而非自动停用抗凝。"
    return CalculatorResult(
        calculator="has_bled", name="HAS-BLED 出血风险",
        score=score, unit="分", risk_level=level, interpretation=interpretation,
        criteria=criteria, details={"age": values["age"]},
        notes=list(_HAS_BLED_NOTES), reference=_HAS_BLED_REFERENCE,
    )


_HAS_BLED_INPUTS = (
    InputSpec("hypertension", "未控制的高血压（收缩压 >160 mmHg）", default=False),
    InputSpec("renal_disease", "肾功能异常", default=False),
    InputSpec("liver_disease", "肝功能异常", default=False),
    InputSpec("stroke_history", "卒中史", default=False),
    InputSpec("bleeding_history", "出血史或出血倾向", default=False),
    InputSpec("labile_inr", "INR 不稳定", default=False),
    InputSpec("age", "年龄", kind="number", minimum=18, maximum=120, unit="岁"),
    InputSpec("drugs", "合并抗血小板/NSAID/酒精等", default=False),
    InputSpec("alcohol", "酒精过量", default=False),
)
_HAS_BLED_NOTES = (
    "高分不是抗凝禁忌；应针对性纠正可逆因素并缩短随访。",
    "HAS-BLED 与 CHA₂DS₂-VASc 应联合解读。",
)
_HAS_BLED_REFERENCE = "Pisters R, et al. Chest 2010; ESC 2020 房颤指南"


def _curb_65(inputs: dict[str, Any]) -> CalculatorResult:
    values = _normalize(inputs, _CURB65_INPUTS)
    criteria = [
        _criterion("confusion", "意识障碍", 1, values["confusion"]),
        _criterion("urea", "血尿素氮 >7 mmol/L", 1, values["urea_mmol_l"] > 7,
                   values["urea_mmol_l"]),
        _criterion("respiratory_rate", "呼吸频率 ≥30 次/分", 1,
                   values["respiratory_rate"] >= 30, values["respiratory_rate"]),
        _criterion("blood_pressure", "收缩压 <90 或舒张压 ≤60 mmHg", 1,
                   values["systolic_bp"] < 90 or values["diastolic_bp"] <= 60,
                   {"systolic": values["systolic_bp"], "diastolic": values["diastolic_bp"]}),
        _criterion("age", "年龄 ≥65 岁", 1, values["age"] >= 65, values["age"]),
    ]
    score = sum(item["points"] for item in criteria if item["met"])
    if score <= 1:
        level, interpretation = "low", "低危，可考虑门诊治疗。"
    elif score == 2:
        level, interpretation = "moderate", "中危，建议住院或短期严密观察。"
    else:
        level, interpretation = "high", "高危，建议住院；≥4 分应考虑重症监护评估。"
    return CalculatorResult(
        calculator="curb_65", name="CURB-65 社区获得性肺炎严重度",
        score=score, unit="分", risk_level=level, interpretation=interpretation,
        criteria=criteria, details={}, notes=list(_CURB65_NOTES),
        reference=_CURB65_REFERENCE,
    )


_CURB65_INPUTS = (
    InputSpec("confusion", "新发意识障碍", default=False),
    InputSpec("urea_mmol_l", "血尿素", kind="number", minimum=0, maximum=100,
              unit="mmol/L"),
    InputSpec("respiratory_rate", "呼吸频率", kind="number", minimum=0, maximum=100,
              unit="次/分"),
    InputSpec("systolic_bp", "收缩压", kind="number", minimum=0, maximum=300, unit="mmHg"),
    InputSpec("diastolic_bp", "舒张压", kind="number", minimum=0, maximum=200, unit="mmHg"),
    InputSpec("age", "年龄", kind="number", minimum=18, maximum=120, unit="岁"),
)
_CURB65_NOTES = (
    "CURB-65 用于社区获得性肺炎严重度分层，不替代临床整体判断。",
    "低氧血症、合并症与社会支持同样影响是否住院。",
)
_CURB65_REFERENCE = "Lim WS, et al. Thorax 2003; BTS CAP guideline 2009"


def _wells_dvt(inputs: dict[str, Any]) -> CalculatorResult:
    values = _normalize(inputs, _WELLS_DVT_INPUTS)
    criteria = [
        _criterion("cancer", "活动性肿瘤（6 个月内治疗或姑息）", 1, values["cancer"]),
        _criterion("immobilization", "瘫痪/石膏固定，或卧床 >3 天/12 周内大手术", 1,
                   values["immobilization"]),
        _criterion("tenderness", "沿深静脉走行的局部压痛", 1, values["tenderness"]),
        _criterion("swelling", "整条腿肿胀", 1, values["swelling"]),
        _criterion("calf", "小腿肿胀，患侧较健侧 >3 cm", 1, values["calf_swelling"]),
        _criterion("edema", "患侧凹陷性水肿", 1, values["edema"]),
        _criterion("collateral", "患侧浅表侧支静脉", 1, values["collateral_veins"]),
        _criterion("previous_dvt", "既往确诊 DVT", 1, values["previous_dvt"]),
        _criterion("alternative", "其他诊断可能性 ≥ DVT", -2, values["alternative_diagnosis"]),
    ]
    score = sum(item["points"] for item in criteria if item["met"])
    if score <= 0:
        level, interpretation = "low", "DVT 可能性低（三级分层）；D-二聚体阴性可基本排除。"
    elif score <= 2:
        level, interpretation = "moderate", "DVT 可能性中等（三级分层）；建议 D-二聚体或超声。"
    else:
        level, interpretation = "high", "DVT 可能性高（三级分层）；建议尽快下肢静脉超声。"
    interpretation += " 两级分层：≥2 分为 DVT 可能，<2 分为 DVT 不太可能。"
    return CalculatorResult(
        calculator="wells_dvt", name="Wells 深静脉血栓评分",
        score=score, unit="分", risk_level=level, interpretation=interpretation,
        criteria=criteria, details={"two_tier": "likely" if score >= 2 else "unlikely"},
        notes=list(_WELLS_NOTES), reference=_WELLS_DVT_REFERENCE,
    )


_WELLS_DVT_INPUTS = (
    InputSpec("cancer", "活动性肿瘤", default=False),
    InputSpec("immobilization", "瘫痪/石膏固定/卧床/近期大手术", default=False),
    InputSpec("tenderness", "沿深静脉走行的局部压痛", default=False),
    InputSpec("swelling", "整条腿肿胀", default=False),
    InputSpec("calf_swelling", "小腿肿胀 >3 cm", default=False),
    InputSpec("edema", "患侧凹陷性水肿", default=False),
    InputSpec("collateral_veins", "患侧浅表侧支静脉", default=False),
    InputSpec("previous_dvt", "既往 DVT", default=False),
    InputSpec("alternative_diagnosis", "其他诊断可能性 ≥ DVT", default=False),
)


def _wells_pe(inputs: dict[str, Any]) -> CalculatorResult:
    values = _normalize(inputs, _WELLS_PE_INPUTS)
    criteria = [
        _criterion("dvt_signs", "临床 DVT 征象", 3, values["dvt_signs"]),
        _criterion("pe_likely", "PE 是最可能的诊断", 3, values["pe_most_likely"]),
        _criterion("heart_rate", "心率 >100 次/分", 1.5, values["heart_rate"] > 100,
                   values["heart_rate"]),
        _criterion("immobilization", "制动 ≥3 天或 4 周内手术", 1.5,
                   values["immobilization"]),
        _criterion("previous_vte", "既往 DVT/PE", 1.5, values["previous_vte"]),
        _criterion("hemoptysis", "咯血", 1, values["hemoptysis"]),
        _criterion("malignancy", "活动性肿瘤", 1, values["malignancy"]),
    ]
    score = sum(item["points"] for item in criteria if item["met"])
    if score < 2:
        level, interpretation = "low", "PE 可能性低（三级分层）。"
    elif score <= 6:
        level, interpretation = "moderate", "PE 可能性中等（三级分层）；建议 D-二聚体/CTPA 评估。"
    else:
        level, interpretation = "high", "PE 可能性高（三级分层）；建议尽快影像学评估。"
    interpretation += " 两级分层：>4 分为 PE 可能，≤4 分为 PE 不太可能。"
    return CalculatorResult(
        calculator="wells_pe", name="Wells 肺栓塞评分",
        score=score, unit="分", risk_level=level, interpretation=interpretation,
        criteria=criteria, details={"two_tier": "likely" if score > 4 else "unlikely"},
        notes=list(_WELLS_NOTES), reference=_WELLS_PE_REFERENCE,
    )


_WELLS_PE_INPUTS = (
    InputSpec("dvt_signs", "临床 DVT 征象", default=False),
    InputSpec("pe_most_likely", "PE 是最可能的诊断", default=False),
    InputSpec("heart_rate", "心率", kind="number", minimum=0, maximum=300, unit="次/分"),
    InputSpec("immobilization", "制动 ≥3 天或 4 周内手术", default=False),
    InputSpec("previous_vte", "既往 DVT/PE", default=False),
    InputSpec("hemoptysis", "咯血", default=False),
    InputSpec("malignancy", "活动性肿瘤", default=False),
)
_WELLS_NOTES = (
    "Wells 评分需结合 D-二聚体与影像学，不单独用于排除或确诊。",
    "妊娠、抗凝状态等特殊人群需使用相应流程。",
)
_WELLS_DVT_REFERENCE = "Wells PS, et al. Lancet 1997; NICE NG158"
_WELLS_PE_REFERENCE = "Wells PS, et al. Thromb Haemost 2000; ESC 2019 PE guideline"


def _egfr_ckd_epi_2021(inputs: dict[str, Any]) -> CalculatorResult:
    values = _normalize(inputs, _EGFR_INPUTS)
    female = values["sex"] == "female"
    creatinine = values["creatinine"]
    if values["creatinine_unit"] == "umol_l":
        creatinine = creatinine / 88.4
    if creatinine <= 0:
        raise CalculatorError("血肌酐必须大于 0")
    kappa = 0.7 if female else 0.9
    alpha = -0.241 if female else -0.302
    ratio = creatinine / kappa
    egfr = (142 * min(ratio, 1.0) ** alpha * max(ratio, 1.0) ** -1.200
            * 0.9938 ** values["age"] * (1.012 if female else 1.0))
    egfr = round(egfr, 1)
    if egfr >= 90:
        stage, level, interpretation = "G1", "low", "肾功能正常或升高（G1，≥90）。"
    elif egfr >= 60:
        stage, level, interpretation = "G2", "low", "轻度下降（G2，60–89）。"
    elif egfr >= 45:
        stage, level, interpretation = "G3a", "moderate", "轻中度下降（G3a，45–59）。"
    elif egfr >= 30:
        stage, level, interpretation = "G3b", "moderate", "中重度下降（G3b，30–44）。"
    elif egfr >= 15:
        stage, level, interpretation = "G4", "high", "重度下降（G4，15–29）。"
    else:
        stage, level, interpretation = "G5", "very_high", "肾衰竭（G5，<15）。"
    return CalculatorResult(
        calculator="egfr_ckd_epi_2021", name="eGFR（CKD-EPI 2021，无种族系数）",
        score=egfr, unit="mL/min/1.73m²", risk_level=level,
        interpretation=f"{interpretation} CKD 分期需结合白蛋白尿（A 分级）持续 ≥3 个月。",
        criteria=[], details={"stage": stage, "creatinine_mg_dl": round(creatinine, 3),
                              "age": values["age"], "sex": values["sex"]},
        notes=list(_EGFR_NOTES), reference=_EGFR_REFERENCE,
    )


_EGFR_INPUTS = (
    InputSpec("age", "年龄", kind="number", minimum=18, maximum=120, unit="岁"),
    InputSpec("sex", "性别", kind="choice", default="male",
              options=(("male", "男"), ("female", "女"))),
    InputSpec("creatinine", "血肌酐", kind="number", minimum=0.01, maximum=3000),
    InputSpec("creatinine_unit", "肌酐单位", kind="choice", default="mg_dl",
              options=(("mg_dl", "mg/dL"), ("umol_l", "µmol/L"))),
)
_EGFR_NOTES = (
    "CKD-EPI 2021 去除了种族系数；仅适用于成人稳定期血肌酐。",
    "急性肾损伤、极端体型、妊娠或透析患者不适用。",
)
_EGFR_REFERENCE = "Inker LA, et al. NEJM 2021; KDIGO 2024 CKD guideline"


CALCULATORS: dict[str, Calculator] = {
    "cha2ds2_vasc": Calculator(
        "cha2ds2_vasc", "CHA₂DS₂-VASc 卒中风险",
        "房颤患者卒中风险评估（0–9 分）。", _CHA2DS2_INPUTS, _cha2ds2_vasc,
        _CHA2DS2_REFERENCE, _CHA2DS2_NOTES,
    ),
    "has_bled": Calculator(
        "has_bled", "HAS-BLED 出血风险",
        "抗凝出血风险评估（0–9 分）。", _HAS_BLED_INPUTS, _has_bled,
        _HAS_BLED_REFERENCE, _HAS_BLED_NOTES,
    ),
    "curb_65": Calculator(
        "curb_65", "CURB-65 肺炎严重度",
        "社区获得性肺炎 30 天死亡风险分层（0–5 分）。", _CURB65_INPUTS, _curb_65,
        _CURB65_REFERENCE, _CURB65_NOTES,
    ),
    "wells_dvt": Calculator(
        "wells_dvt", "Wells DVT 评分",
        "疑似深静脉血栓的临床概率（-2–9 分）。", _WELLS_DVT_INPUTS, _wells_dvt,
        _WELLS_DVT_REFERENCE, _WELLS_NOTES,
    ),
    "wells_pe": Calculator(
        "wells_pe", "Wells 肺栓塞评分",
        "疑似肺栓塞的临床概率（0–12.5 分）。", _WELLS_PE_INPUTS, _wells_pe,
        _WELLS_PE_REFERENCE, _WELLS_NOTES,
    ),
    "egfr_ckd_epi_2021": Calculator(
        "egfr_ckd_epi_2021", "eGFR（CKD-EPI 2021）",
        "成人估算肾小球滤过率，采用 2021 无种族系数公式。", _EGFR_INPUTS,
        _egfr_ckd_epi_2021, _EGFR_REFERENCE, _EGFR_NOTES,
    ),
}


def list_calculators() -> list[dict[str, Any]]:
    return [calculator.as_dict() for calculator in CALCULATORS.values()]


def calculate(calculator: str, inputs: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    key = (calculator or "").strip().lower()
    if key not in CALCULATORS:
        raise CalculatorError(
            f"未知计算器：{calculator}。可用：{', '.join(sorted(CALCULATORS))}")
    return CALCULATORS[key].compute(inputs or {}).as_dict()
