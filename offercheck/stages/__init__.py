"""四阶段任务定义层（stage prompt + stage 配置 + 裁定 schema）。

四阶段 prompt 现集中在 `nexa_agent/prompts/offercheck_stage{1,2,3,4}.txt`
（选岗调研 / 简历定向 / 沟通证伪 / offer 证伪），由引擎按 `--stage` 追加到
通用 `react_system.txt` 之后。三态裁定（靠谱/存疑/大概率有坑）由引擎在
`submit_verdict` 结构化提交，前端 `parseStructuredAnswer` 解析渲染裁定卡。

本包保留为场景层占位：若后续需要把 stage 配置/schema 从核心迁出集中管理，
在此扩展；当前无需引入独立模块。
"""
