# RS-Harness 初版项目设计（v0.1.0）

状态：草案（按最新项目约束修订）
日期：2026-09-04
适用范围：最小 Harness、可持续 session 基础问答

## 1. 设计定位

RS-Harness 是面向遥感数据智能分析的 Agent Harness。长期目标是让模型根据用户指令、遥感影像事实和工具观察自主决策；Harness 提供稳定的消息循环、模型客户端、运行环境、状态和轨迹记录。

本设计参考 [shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code) 的 s01 Agent Loop 和“模型 + Harness”分工。核心不是预先编排固定的感知/规划/执行流水线，而是：

```text
messages[] → OpenAI SDK → assistant message
                         ├─ 有 tool call：记录并返回 unsupported_action
                         └─ 无 tool call：提取文本，结束循环
```

v0.1.0 暂不注册工具，因此正常路径只有一次模型调用；消息列表和内容块结构仍按上述模式设计，为后续 MCP 工具接入保留稳定边界。

本版本的关键约束：

1. **必须连接真实大模型。** RS-Harness 不提供离线模式、确定性的 Stub 客户端或无模型时的伪回答。配置缺失、客户端不可用或模型调用失败，都必须显式返回错误。
2. **直接使用 OpenAI SDK。** Agent Loop 直接调用 OpenAI 官方 SDK，不维护客户端目录、统一客户端协议或额外的模型抽象。
3. **Config + `.env` 分工。** 模型、超时、预算、提示词和会话目录等基础设置从 Config 文件加载；API Key 等敏感环境变量可由项目 `.env` 提供，`.env` 不提交到版本库。
4. **遥感数据通过 MCP 处理。** Harness 运行时不得直接读取影像像元、执行栅格/矢量算法或绕过 MCP 调用遥感能力。
5. **Session 是持久化的模型上下文。** 创建一次 session 后，可通过多次 `AgentLoop.run(session_id, request)` 追加交互；一次模型请求不等于一个 session。Session 由调用方显式关闭，不使用 `runs/` 目录或全局 run registry。
6. **轨迹与证据可追溯。** 用户输入、每次模型响应、工具调用响应（接入后）都要求详细记录保存。

## 2. v0.1.0 范围

### 2.1 必须实现

- Python 包和 CLI 入口。
- `query` / `image` 输入检查：至少提供一个非空输入；两者都为空时不启动 session、不调用模型、无错误结果。
- 指令模式（存在 `query`）的基础问答。
- 持久 Agent Loop：按 session 读取并维护完整 `messages.jsonl`，调用 OpenAI SDK，追加 assistant/tool message 并提取本轮文本。
- 直接使用 OpenAI 官方 SDK 的真实模型接入。
- Config 文件加载、`.env` 加载和 Pydantic 校验。
- 超时、最大轮次、最大输出 token 和模型异常的显式处理。
- `SessionState` / `SessionResult` 以及 session 目录中的 JSON 状态快照和 JSONL 轨迹。
- 使用真实模型凭据的集成测试；单元测试只测试数据模型、状态和错误边界。

### 2.2 暂不实现

以下功能只保留清晰的扩展边界，不在 v0.1.0 实现：

- MCP 服务发现、工具注册/执行、遥感元数据/预览/波段处理、算法推理和后处理。
- Hooks、插件系统、权限审批和人工介入。
- 上下文压缩、摘要、外部记忆或向量数据库；ledger 永久保留完整消息。
- 多 Agent、任务分解、并行执行、后台任务、流式输出和 Web 服务。
- 直接读取本地影像，或把影像字节上传给模型。
- 仅有影像时自动执行全部遥感能力的探索模式。

参考仓库 s01 用一个 `bash` 工具演示工具回填；RS-Harness v0.1.0 将 `tools=[]`，不实现任何工具 handler。后续 MCP 工具接入时，沿用同一个消息回填循环，不把工具流程硬编码成新的 Agent 类型。

## 3. 输入与输出契约

### 3.1 输入

```json
{
  "query": "请解释 NDVI 的含义",
  "image": null
}
```

- `query: string | null`：自然语言指令；空白字符串视为未提供。
- `image: string | null`：影像路径、URI 或对象 ID，仅作为不透明引用进入 session 状态。
- `query` 和 `image` 至少有一个非空。两者都为空时，CLI 直接结束，不创建 session、不调用模型，也不返回错误对象。

v0.1.0 的模式边界：

| 输入                      | 模式         | v0.1.0 行为                                                                                       |
| ------------------------- | ------------ | ------------------------------------------------------------------------------------------------- |
| `query`，可带 `image` | 指令模式     | 必须调用真实模型回答 query。若带 image，提示中明确当前版本没有影像观察事实，不得猜测影像内容。    |
| 仅`image`               | 探索模式占位 | 创建 session 并返回`not_supported`，说明 MCP/遥感能力目录尚未接入；不读取影像、不伪造分析报告。 |
| 两者都为空                | 无           | 不启动 session、不调用模型、不输出错误结果。                                                      |

### 3.2 输出

一次 `AgentLoop.run(session_id, request)` 返回本轮 `InteractionResult`；显式
`close-session` 将 session 标记为终态并返回最终 `SessionResult`（不额外生成结果文件）。Session 在每次交互后保持 `running`，
消息上下文由 `messages.jsonl` 恢复。

任何需要回答的 query 都必须经过真实模型调用；没有模型时不得降级为默认文本。

## 4. Config 文件设计

### 4.1 文件与加载规则

基础设置通过 Config 文件提供，默认路径为 `config/default.toml`，CLI 可用 `--config <path>` 指定其他文件。`src/rs_harness/config.py` 中的 `Config` Pydantic 模型负责：

1. 读取 TOML；
2. 解析相对路径（相对于 Config 文件所在目录）；
3. 校验 API key 环境变量名、正数预算和必需字段；
4. 生成不可变的运行配置快照；
5. 配置只在运行时使用，session 文件和消息 ledger 永不写入完整 API key。

Config 文件负责非敏感基础设置；API Key 等敏感值通过项目 `.env` 注入。启动时使用 `python-dotenv` 读取 Config 指定的 `.env` 文件，`api_key_env` 再从进程环境中解析实际密钥。除这些显式声明的敏感值外，其他环境变量不得覆盖 Config 文件。`.env` 必须加入 `.gitignore`，生产环境也可使用部署平台注入的同名环境变量。

### 4.2 配置示例

```toml
# config/default.toml

[model]
model = "gpt-4o-mini"
base_url = ""                        # 空值使用客户端默认地址
api_key_env = "OPENAI_API_KEY"       # API Key 的环境变量名
timeout_s = 60
max_tokens = 4096

[loop]
max_turns = 1

[runtime]
sessions_dir = "sessions"
system_prompt_file = "prompts/system.md"
```

对应的 `.env`（不提交）示例：

```dotenv
OPENAI_API_KEY=your-openai-key
```

`api_key_env` 必须指向已解析的环境变量；若 `.env` 不存在、变量为空、模型 endpoint 或模型名不可用，运行直接进入 `failed/config_error` 或 `failed/model_unavailable`，不创建 Stub 客户端。

## 5. 最小 Agent Loop

### 5.1 循环形状

```text
打开已有 Session，读取 messages.jsonl 中的完整上下文
        │
        ▼
调用 OpenAI Chat Completions 接口（tools=[]）
        │
        ▼
按 provider 原始结构追加 assistant/tool message 到 messages.jsonl
        │
        ├─ 无 tool call → 提取 text → succeeded
        ├─ 有 tool call  → v0.1.0 返回 unsupported_action
        ├─ 空文本       → empty_model_response
        └─ 异常/超预算   → failed
```

### 5.2 伪代码

伪代码沿用参考实现的 `while` 和 assistant 回填；`tools=[]` 是 v0.1.0 的硬约束：

```python
def run(session_id: str, request: InputRequest, config: Config) -> InteractionResult:
    validate_config(config)
    state = open_session(session_id, config)
    append_message(state, build_user_message(request))

    openai_client = OpenAI(
        api_key=load_api_key(config), base_url=config.model.base_url or None
    )

    messages = read_messages(state)
    for _ in range(config.loop.max_turns):
        state.turn += 1
        response = openai_client.chat.completions.create(
            model=config.model.model,
            messages=messages,
            max_tokens=config.model.max_tokens,
            tools=[],
        )
        assistant_message = response.choices[0].message
        assistant_content = assistant_message.content or ""
        tool_calls = list(assistant_message.tool_calls or [])
        answer = assistant_content

        append_message(state, assistant_message)
        if tool_calls:
            return interaction_failed(state, code="unsupported_action")

        if answer.strip():
            return interaction_success(state, answer)
        return interaction_failed(state, code="empty_model_response")

    return interaction_failed(state, code="budget_exceeded")
```

每次 `AgentLoop.run(session_id, request)` 都先把 user message 追加到该 session 的 `messages.jsonl`，再从 ledger 重建上下文；调用完成后追加模型实际产生的 message，并返回本轮 `InteractionResult`，session 保持 `running`。不做摘要、压缩、记忆检索或自动重试。未来工具可用时，在 `tool_calls` 分支执行 MCP handler，将结果按 tool message 追加，再进入下一轮；Loop 的决策权仍属于模型。

### 5.3 提示词约束

system prompt 至少要求模型：

- 直接回答 query，使用用户问题的语言；
- 只依据消息中可见的事实，不编造影像、测量或算法结果；
- 识别到当前 `tools=[]` 时不要要求伪造工具结果；
- 不输出隐藏的逐步思考，只返回面向用户的内容。

带 `image` 的 query 必须在 user 消息中附加限制说明：影像引用已收到，但 v0.1.0 尚未通过 MCP 建立影像事实，因此不能据此判断场景、数值或目标。

## 6. OpenAI SDK 与核心模块

### 6.1 OpenAI SDK 接入

v0.1.0 直接在 `AgentLoop` 中创建 `openai.OpenAI` 并调用 `chat.completions.create(...)`。项目不维护 `clients/` 目录、统一客户端协议或客户端工厂；Loop 直接使用 OpenAI 的 `messages` 和 `choices[0].message` 类型。

未来接入工具时，在同一个 Loop 中按 OpenAI 原生 tool call 格式处理，再转换成 MCP 所需的 `tool_result` 消息。

SDK 初始化或首次调用失败必须返回明确错误；不能切换到 Stub、默认文本或另一家客户端掩盖问题。

### 6.2 核心模块

- `AgentLoop`：状态迁移、轮次/预算检查、OpenAI SDK 调用和终态保证；不导入遥感库。
- `Config`：读取 Config 文件并加载 `.env`，校验模型和预算，生成只读配置快照。
- `MessageLedger`：将完整 provider message 追加写入 `messages.jsonl`，仅附带序号和时间戳元数据；不生成自定义模型调用事件。
- `Session`：通过 `create` / `open` 持久化模型上下文，保存 `session.json`、`messages.jsonl`；`close` 只更新终态并返回内存中的 `SessionResult`。
- `CLI`：`uv run rsh` 默认进入交互循环，支持 `/help`、`/status`、`/close`、`/quit`；同时提供 `create-session`、`ask` 和 `close-session` 脚本入口。

消息 ledger 不得包含 API key、Authorization header 或影像内容；`image` 只保留不透明引用或脱敏摘要。

## 7. 推荐目录结构

```text
RS-Harness/
├── pyproject.toml
├── README.md
├── AGENTS.md
├── config/
│   ├── default.toml                # 默认基础配置模板
│   └── example.toml                # 不含真实凭据的示例
├── .env.example                    # API Key 变量名示例，不含真实密钥
├── prompts/
│   └── system.md                   # 最小系统指令
├── docs/
│   └── 初版Harness设计.md
├── src/
│   └── rs_harness/
│       ├── __init__.py
│       ├── cli.py                  # CLI 入口
│       ├── config.py               # Config 模型、TOML 与 .env 加载
│       ├── agent_loop.py           # 最小 Agent Loop
│       └── runtime/
│           ├── __init__.py
│           ├── session.py          # Session 创建与生命周期
│           └── trace.py            # JSONL 轨迹写入
├── tests/
│   ├── unit/
│   │   ├── test_config.py
│   │   ├── test_agent_loop.py
│   │   ├── test_session.py
│   │   └── test_trace.py
│   └── integration/
│       └── test_openai_client.py
└── sessions/                       # session 运行时生成，不提交结果
```

Session 由显式创建并可接受多次 ask；产物只归属于该 session：

```text
sessions/<session_id>/
├── session.json     # 当前 SessionState 快照（不含完整消息）
└── messages.jsonl   # 完整、追加式 provider message ledger
```

不创建 `stub.py`、统一 `messages_client.py` 或客户端工厂等运行时降级/抽象模块；测试可在 SDK transport 边界使用 mock，但生产 CLI 始终调用真实 SDK。

## 8. 技术栈

| 层次             | 选择                            | 用途与原因                                               |
| ---------------- | ------------------------------- | -------------------------------------------------------- |
| 语言             | Python 3.12+                    | 便于实现模块边界、异步扩展和后续 MCP 接入。              |
| 包管理/构建      | `uv` + `pyproject.toml`     | 锁定依赖并提供可复现运行。                               |
| 数据契约         | Pydantic 2                      | 校验 Config、输入、session 状态和结果。                  |
| Config 文件      | TOML + Python 标准库`tomllib` | 集中保存基础设置，便于审查和复现。                       |
| 敏感配置         | `python-dotenv` + `.env`    | 仅加载 API Key 等敏感环境变量，不覆盖 Config 基础设置。  |
| OpenAI 客户端    | `openai` Python SDK           | 直接调用 OpenAI Chat Completions 接口。                  |
| CLI              | Typer                           | 提供默认交互会话、`ask`、session 管理和 `--config`。 |
| 持久化           | `pathlib` + JSON/JSONL        | 按 session 保存状态和审计轨迹，不使用`runs/`。         |
| 测试             | pytest                          | 状态/分支单元测试，真实凭据集成测试单独标记。            |
| 编排             | 自研小型同步 Loop               | 直接体现参考项目的核心模式，不引入 LangGraph/LangChain。 |

模型调用是运行时必需依赖：不提供 Stub/offline 开关或无密钥默认回答；`.env` 只用于提供密钥等敏感环境变量。

## 9. 配置与运行示例

```bash
# Config 文件中配置 model 和 api_key_env，API Key 放在 .env
uv run rsh ask --config config/default.toml "什么是 NDVI？"
uv run rsh ask --config config/default.toml \
  --image s3://bucket/scene.tif \
  "请说明当前版本能否分析这张影像" --json
```

启动顺序：

1. 解析 CLI 输入；交互模式创建或打开 session，脚本模式按需创建 session；
2. 读取并校验 `--config` 指定的 Config 文件，同时加载其指定的 `.env`；
3. 创建 OpenAI SDK 客户端，并检查必需凭据；
4. 将 system/user/assistant message 按顺序追加到 `messages.jsonl`；
5. 更新 `session.json` 状态；关闭 session 时不创建额外结果文件。

Config 或真实模型不可用时，CLI 使用非零退出码并输出可读错误；不得切换离线模式。

## 10. 错误、预算与安全

- **空输入**：CLI 在创建 session 和模型客户端前直接结束，不调用模型、不写错误结果。
- **Config 错误**：`failed/config_error`，缺字段、路径不可读或凭据来源无效。
- **客户端错误**：`failed/model_unavailable`、`failed/model_timeout` 或 `failed/model_response_error`。
- **模型主动返回 tool call**：`failed/unsupported_action`，完整 assistant message 仍写入 ledger，不执行未知动作。
- **空文本**：`failed/empty_model_response`。
- **预算**：默认最多 1 次模型调用、单次 60 秒、最多 4096 输出 token；超出后返回 `failed/budget_exceeded`。v0.1.0 不自动重试，避免重复计费和隐藏延迟。
- **日志**：所有实际发送或接收的 message 追加到 `messages.jsonl`；不写入自定义模型调用事件。
- **机密**：API key、Authorization header、完整 Config 密钥字段和影像内容不得进入日志或状态快照。

## 11. 测试与验收标准

1. Config 文件能配置 OpenAI 模型和凭据来源；无效配置被拒绝。
2. 对于有效输入，缺少 Config、模型名或凭据时运行失败，不生成 Stub、不返回默认文本。
3. 空请求直接结束，不创建 session、不调用模型、不输出错误结果。
4. query-only 请求调用一次 OpenAI SDK，追加 assistant 内容并返回非空答案。
5. query + image 请求不读取影像文件、不声称观察到影像事实，并把限制写入提示/轨迹。
6. image-only 请求返回 `not_supported`，不伪造探索报告。
7. 客户端超时、网络错误、空响应和非法内容块都会形成明确失败状态和轨迹事件。
8. `max_turns=1` 时最多一次模型调用；所有路径都有最终状态和停止原因。
9. OpenAI SDK 通过真实服务或受控 SDK transport contract test 验证原生响应提取和 session 记录。
10. 生产 CLI 不存在 Stub/offline 开关；非敏感基础设置来自 Config 文件，API Key 等敏感值来自 `.env` 或进程环境。

## 12. 后续演进顺序

在 v0.1.0 基础问答稳定后，按以下顺序扩展，并保持当前 `messages[]` 形状兼容：

1. MCP 客户端、能力目录和工具 handler：在 tool call 分支执行并回填 `tool_result`。
2. 影像感知：通过 MCP 获取元数据、质量和预览事实，再支持 query + image 问答。
3. Skills 按需加载和可控的系统知识注入。
4. 上下文窗口管理、摘要/压缩、可恢复运行和资源监控。
5. Hooks、权限审批、记忆、多 Agent 以及仅影像探索模式；探索模式必须让所有 MCP 遥感能力形成明确终态并生成报告。

每次扩展都要继续遵守：模型负责观察和决策，Harness 负责稳定执行；遥感处理只经过 MCP；工具失败不被静默吞掉；最终回答只引用已确认的观察和产物。
