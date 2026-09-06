# RS-Harness 初版项目设计（v0.1.0）

状态：草案（按最新项目约束修订）
日期：2026-09-04
适用范围：最小 Harness、单轮基础问答

## 1. 设计定位

RS-Harness 是面向遥感数据智能分析的 Agent Harness。长期目标是让模型根据用户指令、遥感影像事实和工具观察自主决策；Harness 提供稳定的消息循环、模型客户端、运行环境、状态和轨迹记录。

本设计参考 [shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code) 的 s01 Agent Loop 和“模型 + Harness”分工。核心不是预先编排固定的感知/规划/执行流水线，而是：

```text
messages[] → Anthropic/OpenAI SDK → assistant content blocks
                         ├─ 有 tool_use：执行并回填结果，继续循环
                         └─ 无 tool_use：提取文本，结束循环
```

v0.1.0 暂不注册工具，因此正常路径只有一次模型调用；消息列表和内容块结构仍按上述模式设计，为后续 MCP 工具接入保留稳定边界。

本版本的关键约束：

1. **必须连接真实大模型。** RS-Harness 不提供离线模式、确定性的 Stub 客户端或无模型时的伪回答。配置缺失、客户端不可用或模型调用失败，都必须显式返回错误。
2. **直接使用厂商 SDK。** 根据 Config 中的 provider 分支调用 Anthropic 和 OpenAI 官方 SDK，不额外定义统一的 Messages Client 协议，也不把 SDK 类型泄漏/转换成另一套客户端抽象。
3. **Config + `.env` 分工。** 模型、provider、超时、预算、提示词和会话目录等基础设置从 Config 文件加载；API Key 等敏感环境变量可由项目 `.env` 提供，`.env` 不提交到版本库。
4. **遥感数据通过 MCP 处理。** Harness 运行时不得直接读取影像像元、执行栅格/矢量算法或绕过 MCP 调用遥感能力。
5. **每次运行就是一个 session。** 不使用 `runs/` 目录或全局 run registry 维护运行；每次有效输入独立创建一个 session，并由该 session 保存自己的消息、状态和轨迹。
6. **轨迹与证据可追溯。** 用户输入、每次模型响应、工具调用响应（接入后）都要求详细记录保存。

## 2. v0.1.0 范围

### 2.1 必须实现

- Python 包和 CLI 入口。
- `query` / `image` 输入检查：至少提供一个非空输入；两者都为空时不启动 session、不调用模型、无错误结果。
- 指令模式（存在 `query`）的基础问答。
- 单轮 Agent Loop：维护 `messages[]`、调用所选厂商 SDK、追加 assistant 内容、提取最终文本。
- 直接使用 Anthropic 与 OpenAI 官方 SDK 的两个真实模型接入分支。
- Config 文件加载、`.env` 加载、Pydantic 校验和 provider 选择。
- 超时、最大轮次、最大输出 token 和模型异常的显式处理。
- `SessionState` / `SessionResult` 以及 session 目录中的 JSON 状态快照和 JSONL 轨迹。
- 使用真实模型凭据的集成测试；单元测试只测试数据模型、状态和错误边界。

### 2.2 暂不实现

以下功能只保留清晰的扩展边界，不在 v0.1.0 实现：

- MCP 服务发现、工具注册/执行、遥感元数据/预览/波段处理、算法推理和后处理。
- Hooks、插件系统、权限审批和人工介入。
- 跨轮上下文压缩、会话上下文管理、外部记忆或向量数据库。
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

```json
{
  "session_id": "20260904T120000Z-a1b2c3",
  "status": "succeeded",
  "answer": "NDVI 是用于表示植被活力的归一化指数……",
  "errors": [],
  "session_path": "sessions/20260904T120000Z-a1b2c3"
}
```

`status` 取值：

- `succeeded`：真实模型返回了非空文本回答。
- `not_supported`：输入合法，但当前版本没有对应能力（例如仅影像探索）。
- `failed`：配置、客户端、网络、超时、响应解析或运行时错误；`errors` 必须包含分类和可读原因。
- 空输入不产生 `SessionResult`：CLI 在启动 session 前直接结束，不调用模型，也不输出错误对象。

任何需要回答的 query 都必须经过真实模型调用；没有模型时不得降级为默认文本。

## 4. Config 文件设计

### 4.1 文件与加载规则

基础设置通过 Config 文件提供，默认路径为 `config/default.toml`，CLI 可用 `--config <path>` 指定其他文件。`src/rs_harness/config.py` 中的 `Config` Pydantic 模型负责：

1. 读取 TOML；
2. 解析相对路径（相对于 Config 文件所在目录）；
3. 校验客户端类型、API key 环境变量名、正数预算和必需字段；
4. 生成不可变的运行配置快照；
5. 将配置摘要写入 `session_started`，但永不写入完整 API key。

Config 文件负责非敏感基础设置；API Key 等敏感值通过项目 `.env` 注入。启动时使用 `python-dotenv` 读取 Config 指定的 `.env` 文件，`api_key_env` 再从进程环境中解析实际密钥。除这些显式声明的敏感值外，其他环境变量不得覆盖 Config 文件。`.env` 必须加入 `.gitignore`，生产环境也可使用部署平台注入的同名环境变量。

### 4.2 配置示例

```toml
# config/default.toml

[model]
provider = "anthropic"               # anthropic | openai
model = "claude-sonnet-4-20250514"
base_url = ""                        # 空值使用客户端默认地址
api_key_env = "ANTHROPIC_API_KEY"    # API Key 的环境变量名
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
ANTHROPIC_API_KEY=your-anthropic-key
OPENAI_API_KEY=your-openai-key
```

`provider` 只允许 `anthropic` 或 `openai`。`api_key_env` 必须指向已解析的环境变量；若 `.env` 不存在、变量为空、模型 endpoint 或模型名不可用，运行直接进入 `failed/config_error` 或 `failed/model_unavailable`，不创建 Stub 客户端。

## 5. 最小 Agent Loop

### 5.1 循环形状

```text
初始化 Config、真实厂商 SDK、SessionState 和 messages[]
        │
        ▼
调用已选厂商 SDK 的 Messages 接口（tools=[]）
        │
        ▼
追加完整 assistant content blocks 到 messages[]
        │
        ├─ 无 tool_use → 提取 text → succeeded
        ├─ 有 tool_use  → v0.1.0 返回 unsupported_action
        ├─ 空文本       → empty_model_response
        └─ 异常/超预算   → failed
```

### 5.2 伪代码

伪代码沿用参考实现的 `while`、assistant 回填和 content block 检查；`tools=[]` 是 v0.1.0 的硬约束：

```python
def run(request: InputRequest, config: Config) -> SessionResult | None:
    if not has_input(request):
        return None  # CLI 直接结束，不创建 session、不返回错误

    validate_config(config)
    state = start_session(request, config)

    if config.model.provider == "anthropic":
        anthropic_client = Anthropic(
            api_key=load_api_key(config), base_url=config.model.base_url or None
        )
    else:
        openai_client = OpenAI(
            api_key=load_api_key(config), base_url=config.model.base_url or None
        )

    if request.query is None:
        return finish_not_supported(state, reason="image_only_not_implemented")

    messages = [{"role": "user", "content": build_user_message(request)}]
    while state.turn < config.loop.max_turns:
        state.turn += 1
        if config.model.provider == "anthropic":
            response = anthropic_client.messages.create(
                model=config.model.model,
                system=build_system_prompt(config),
                messages=messages,
                max_tokens=config.model.max_tokens,
            )
            assistant_content = response.content
            tool_calls = [b for b in assistant_content if b.type == "tool_use"]
            answer = extract_anthropic_text(assistant_content)
        else:
            response = openai_client.chat.completions.create(
                model=config.model.model,
                messages=messages,
                max_tokens=config.model.max_tokens,
            )
            assistant_message = response.choices[0].message
            assistant_content = assistant_message.content or ""
            tool_calls = list(assistant_message.tool_calls or [])
            answer = assistant_content

        messages.append({"role": "assistant", "content": assistant_content})
        observe_model(state, response, provider=config.model.provider)
        if tool_calls:
            return finish_failed(state, code="unsupported_action")

        if answer.strip():
            return finish_success(state, answer)
        return finish_failed(state, code="empty_model_response")

    return finish_failed(state, code="budget_exceeded")
```

v0.1.0 的 `messages[]` 只属于当前 session，不跨 session 复用；不做摘要、压缩、记忆检索或自动重试。未来工具可用时，在 `tool_calls` 分支执行 MCP handler，将结果按 `tool_result` content block 追加为 user 消息，再进入下一轮；Loop 的决策权仍属于模型。

### 5.3 提示词约束

system prompt 至少要求模型：

- 直接回答 query，使用用户问题的语言；
- 只依据消息中可见的事实，不编造影像、测量或算法结果；
- 识别到当前 `tools=[]` 时不要要求伪造工具结果；
- 不输出隐藏的逐步思考，只返回面向用户的内容。

带 `image` 的 query 必须在 user 消息中附加限制说明：影像引用已收到，但 v0.1.0 尚未通过 MCP 建立影像事实，因此不能据此判断场景、数值或目标。

## 6. 厂商 SDK 与核心模块

### 6.1 两个厂商 SDK 接入

v0.1.0 不定义统一的 `MessagesClient` Protocol，也不把两个 SDK 强行转换成同一套客户端对象。`AgentLoop` 根据 `Config.model.provider` 选择明确的代码分支，直接调用厂商 SDK：

- Anthropic 分支调用 `anthropic.Anthropic(...).messages.create(...)`，使用 Anthropic 的 `messages` 与 content block 类型。
- OpenAI 分支调用 `openai.OpenAI(...).chat.completions.create(...)`，使用 OpenAI 的 `messages` 和 `choices[0].message` 类型。

两条分支只在 Loop 所需的最小位置共享辅助逻辑（追加 assistant 消息、提取文本、记录事件）；不共享客户端协议、响应模型或工具注册表。未来接入工具时，分别按照两个 SDK 的原生 tool call 格式处理，再转换成 MCP 所需的 `tool_result` 消息。

SDK 初始化或首次调用失败必须返回明确错误；不能切换到 Stub、默认文本或另一家客户端掩盖问题。

### 6.2 核心模块

- `AgentLoop`：状态迁移、轮次/预算检查、厂商 SDK 调用和终态保证；不导入遥感库。
- `Config`：读取 Config 文件并加载 `.env`，校验 provider、模型和预算，生成只读配置快照。
- `TraceWriter`：追加 JSONL 事件，不承担 Hooks 功能。
- `Session`：创建一个 session 的消息列表、状态和会话目录，保存 `session.json`、`result.json`。
- `CLI`：解析 `--config`、`query`、`--image` 和 `--json`，不包含模型决策逻辑。

至少记录：`session_started`、`model_call_started`、`model_call_finished`、`session_finished`、`session_error`。轨迹不得包含 API key、Authorization header 或影像内容；`image` 只保留不透明引用或脱敏摘要。

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
│       ├── clients/
│       │   ├── __init__.py
│       │   ├── anthropic.py        # Anthropic SDK 初始化与调用
│       │   └── openai.py           # OpenAI SDK 初始化与调用
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
│       ├── test_anthropic_client.py
│       └── test_openai_client.py
└── sessions/                       # session 运行时生成，不提交结果
```

每次运行就是一个独立 session，产物只归属于该 session：

```text
sessions/<session_id>/
├── session.json     # SessionState 最终快照（含本次 messages）
├── result.json      # 对外 SessionResult
└── events.jsonl     # 本次 session 的审计事件
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
| Anthropic 客户端 | `anthropic` Python SDK        | 直接调用 Anthropic Messages API。                        |
| OpenAI 客户端    | `openai` Python SDK           | 直接调用 OpenAI 的 messages 接口。                       |
| CLI              | Typer                           | 提供`ask` 和 `--config` 等最小命令行接口。           |
| 持久化           | `pathlib` + JSON/JSONL        | 按 session 保存状态和审计轨迹，不使用`runs/`。         |
| 测试             | pytest                          | 状态/分支单元测试，真实凭据集成测试单独标记。            |
| 编排             | 自研小型同步 Loop               | 直接体现参考项目的核心模式，不引入 LangGraph/LangChain。 |

模型调用是运行时必需依赖：不提供 Stub/offline 开关或无密钥默认回答；`.env` 只用于提供密钥等敏感环境变量。

## 9. 配置与运行示例

```bash
# Config 文件中配置 provider、model 和 api_key_env，API Key 放在 .env
uv run rs-harness ask --config config/default.toml "什么是 NDVI？"
uv run rs-harness ask --config config/default.toml \
  --image s3://bucket/scene.tif \
  "请说明当前版本能否分析这张影像" --json
```

启动顺序：

1. 解析 CLI 输入；若 query/image 都为空，直接结束，不加载客户端、不创建 session；
2. 读取并校验 `--config` 指定的 Config 文件，同时加载其指定的 `.env`；
3. 创建指定的 Anthropic 或 OpenAI SDK 客户端，并检查必需凭据；
4. 创建本次 session 目录、系统指令和轨迹写入器；
5. 执行 Agent Loop，在本次 session 目录保存最终状态和结果。

Config 或真实模型不可用时，CLI 使用非零退出码并输出可读错误；不得切换离线模式。

## 10. 错误、预算与安全

- **空输入**：CLI 在创建 session 和模型客户端前直接结束，不调用模型、不写错误结果。
- **Config 错误**：`failed/config_error`，缺字段、路径不可读或凭据来源无效。
- **客户端错误**：`failed/model_unavailable`、`failed/model_timeout` 或 `failed/model_response_error`。
- **模型主动返回 `tool_use`**：`failed/unsupported_action`，保留响应摘要，不执行未知动作。
- **空文本**：`failed/empty_model_response`。
- **预算**：默认最多 1 次模型调用、单次 60 秒、最多 4096 输出 token；超出后返回 `failed/budget_exceeded`。v0.1.0 不自动重试，避免重复计费和隐藏延迟。
- **日志**：所有终态尽力写入 `session_finished`；轨迹写入失败时状态标记为 failed，并在 stderr 保留原因。
- **机密**：API key、Authorization header、完整 Config 密钥字段和影像内容不得进入日志或 `result.json`。

## 11. 测试与验收标准

1. Config 文件能分别选择 Anthropic 和 OpenAI 客户端；非法 client 值被拒绝。
2. 对于有效输入，缺少 Config、模型名或凭据时运行失败，不生成 Stub、不返回默认文本。
3. 空请求直接结束，不创建 session、不调用模型、不输出错误结果。
4. query-only 请求调用一次已配置的 Anthropic 或 OpenAI SDK，追加 assistant 内容并返回非空答案。
5. query + image 请求不读取影像文件、不声称观察到影像事实，并把限制写入提示/轨迹。
6. image-only 请求返回 `not_supported`，不伪造探索报告。
7. 客户端超时、网络错误、空响应和非法内容块都会形成明确失败状态和轨迹事件。
8. `max_turns=1` 时最多一次模型调用；所有路径都有最终状态和停止原因。
9. Anthropic/OpenAI 分支分别通过真实服务或受控 SDK transport contract test 验证原生响应提取和 session 记录。
10. 生产 CLI 不存在 Stub/offline 开关；非敏感基础设置来自 Config 文件，API Key 等敏感值来自 `.env` 或进程环境。

## 12. 后续演进顺序

在 v0.1.0 基础问答稳定后，按以下顺序扩展，并保持当前 `messages[]` 形状兼容：

1. MCP 客户端、能力目录和工具 handler：在 `tool_use` 分支执行并回填 `tool_result`。
2. 影像感知：通过 MCP 获取元数据、质量和预览事实，再支持 query + image 问答。
3. Skills 按需加载和可控的系统知识注入。
4. 上下文窗口管理、摘要/压缩、可恢复运行和资源监控。
5. Hooks、权限审批、记忆、多 Agent 以及仅影像探索模式；探索模式必须让所有 MCP 遥感能力形成明确终态并生成报告。

每次扩展都要继续遵守：模型负责观察和决策，Harness 负责稳定执行；遥感处理只经过 MCP；工具失败不被静默吞掉；最终回答只引用已确认的观察和产物。
