# OfferCheck 场景层

建在 `nexa_agent` 核心引擎之上的求职反诈应用。四阶段共用同一自主调查引擎
（发现驱动的证伪循环），通过注入不同 stage 任务定义与领域工具区分。

```
offercheck/
├── stages/      # 四阶段 prompt + stage 配置 + 裁定 schema
├── tools/       # 领域工具：whois / company_registry ...
└── eval_suite/  # 求职诈骗/正常案例集（expected_verdict）
```

## 现状（骨架）
- ✅ stage1(选岗) / stage4(offer 证伪) prompt：`nexa_agent/prompts/offercheck_stage{1,4}.txt`
- ✅ domain_whois_lookup：`nexa_agent/tools.py`
- ✅ 裁定解析 parse_offer_verdict：`nexa_agent/verifier.py`
- ⬜ stage2/stage3、company_registry_search、eval_suite（按 7/3-7/4 计划）
