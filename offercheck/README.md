# OfferCheck 场景层

建在 `nexa_agent` 核心引擎之上的求职反诈应用。四阶段共用同一自主调查引擎
（发现驱动的证伪循环），通过注入不同 stage 任务定义与领域工具区分。

```
offercheck/
├── stages/      # 四阶段 prompt + stage 配置 + 裁定 schema
├── tools/       # 领域工具：whois / company_registry ...
└── eval_suite/  # 求职诈骗/正常案例集（expected_verdict）
```

## 现状
- ✅ 四阶段 prompt：`nexa_agent/prompts/offercheck_stage{1,2,3,4}.txt`
- ✅ domain_whois_lookup：`nexa_agent/tools.py`
- ✅ 裁定解析 parse_offer_verdict：`nexa_agent/verifier.py`
- ✅ eval_suite：`eval_suite/cases.jsonl`（32 例覆盖全四阶段，含 6 条全英文）——裁定级 21 例（stage1/3/4，expected_verdict）
  + 关键词召回 5 例（stage2，expected_keywords）接入 Eval Harness
  （`python -m nexa_agent.eval_harness run --suite offercheck`）
- ⬜ company_registry_search（需外部工商 API，当前用 web_search + whois 兜底）
