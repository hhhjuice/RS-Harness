# RS-Harness Repository Guide

## 项目目标

RS-Harness 是一个面向遥感数据智能分析的 Agent 基础框架。它使用 ReAct（Reason + Act + Observe）式决策循环，让模型根据用户 query、遥感影像以及工具观察，逐步选择分析动作并生成可审计结果。

本项目的 Harness 框架设计实现参考 [github.com/shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code) 。本项目的工具可以参考复用 `/home/heguozhen/Workspace/Space-Agent-dev/Space-Agent` 项目已具备的推理算法工具和通用的遥感数据处理工具，但不直接继承 Space-Agent 的固定流水线、算法卡或注册表。

## 核心原则

1. **模型负责观察决策，Harness负责稳定执行。** 模型可以观察理解用户指令、输入图像和工具执行结果进行思考决策；Harness实现模型以外的所有基础设施，如指令、工具、环境、状态、反馈等必要的部分（参考[walkinglabs.github.io/learn-harness-engineering/zh](https://walkinglabs.github.io/learn-harness-engineering/zh/)）。
2. **遥感数据的处理通过 MCP 实现。** RS-Harness 遥感数据处理能力由 MCP 扩展实现。
3. **轨迹可追溯。** 用户的输入、模型的决策以及工具的执行必须有迹可循，RS-Harness 处理链路清晰记录。
4. **证据优先。** 最终结果只能基于已确认的工具观察和产物。

### 运行模式

- **指令模式**：用户指令只要存在 自然语言指令 query 就使用此模式（无论是否有 image）。该模式下 RS-Harness 的最终回答 Answer 只需解答 query。
- **探索模式**：用户指令仅有 image 且没有 query 时使用此模式。该模式下 RS-Harness 会自主通过 MCP 扩展的所有遥感数据处理能力挖掘遥感影像 image 具备的所有信息，并生成分析报告保存。

## 运行流程

```text
RS-Harness 初始化（加载 系统指令、基础工具、MCP、Skills和运行环境 等）
        │
        ▼
接收用户指令（query / image）
        │
        ├─ 有 image：通过 MCP 建立元数据、质量和预览事实
        └─ 无 image：跳过影像感知，只使用可用的非影像上下文
        │
        ▼
Agent Loop 决策循环：选择工具 → 调用 → 记录结构化观察 → 再决策
        │
        ▼
按模式输出回答/报告，同时写入轨迹日志
```

### 工具边界

必须通过 MCP 扩展实现的工具包括：遥感元数据和质量读取、预览、波段/空间处理、格式转换、统计、矢量化、结果检查、可视化、算法推理及其后处理。

可以由框架运行时直接提供的基础工具包括：接收输入引用、创建工作目录、读写普通 JSON/文本、配置、日志、哈希、进程管理、资源监控和产物打包。运行时不得借此处理遥感语义。
