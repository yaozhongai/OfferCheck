"""OfferCheck 四阶段评测集（含 ground truth），复用 nexa_agent eval_harness。

案例集：`cases.jsonl`（32 例，其中 6 条全英文 tag=english），按阶段用不同评分：

- **stage1 选岗决策（×5，裁定级）**：`expected_verdict`（推荐→reliable /
  谨慎→suspicious / 不推荐→likely_scam，引擎的 [Verdict] label 已被分类器映射到同一枚举）。
- **stage3 沟通证伪（×4）+ stage4 offer 证伪（×12，裁定级）**：`expected_verdict`
  ∈ reliable | suspicious | likely_scam（靠谱/存疑/大概率有坑）。
- **stage2 简历定向（×5，关键词召回）**：非裁定型自由文本清单，用 `expected_keywords`
  预置「JD 要求而简历缺失、正确定向分析必须指出的差距关键词」，按命中召回率评分。

运行::

    python -m nexa_agent.eval_harness run --suite offercheck
    python -m nexa_agent.eval_harness analyze --input <results.jsonl>
    python -m nexa_agent.eval_harness compare --baseline A.jsonl --current B.jsonl  # 回归门禁

评分逻辑在 `nexa_agent/eval_harness.py`：
- 裁定级：`classify_prediction_verdict`（只信任 [Verdict] label 本身，避开 reason 里
  否定语境的 scam 关键词误伤）+ `compute_verdict_metrics`（准确率 / 误报=把靠谱判成有坑 /
  漏报=把诈骗判成靠谱 / 拒答=无裁定 + 混淆矩阵）。
- 关键词召回：`score_keyword_recall` + `compute_keyword_metrics`（平均召回率 + 达标率，
  recall ≥ KEYWORD_RECALL_THRESHOLD=0.6 判 correct）。

每条 case 恰好带 `expected_verdict` 或 `expected_keywords` 之一，harness 据此选评分模式。
"""
