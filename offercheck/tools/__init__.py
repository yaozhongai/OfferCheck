"""OfferCheck 领域工具。

`domain_whois_lookup` 与 `analyze_image_cloud`（读图 OCR）现注册于
`nexa_agent/tools.py`（通用工具表），四阶段共用。

未建：`company_registry_search`（企业信用/诉讼/黑名单）——依赖外部工商 API，
按 SPEC §2/§5 延后；当前 stage3/4 用 `web_search` + `whois` 多源兜底核实公司实体。
后续领域工具在此扩展或迁移。
"""
