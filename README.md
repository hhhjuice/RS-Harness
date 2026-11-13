# RS-Harness

基于 LangChain 和 LangGraph 的本地 Agent 运行框架，提供命令行交互与 Python 调用接口，支持模型工具调用、多轮会话持久化、任务规划和子代理协作。

项目目前处于开发阶段，详细设计与实现说明见 [docs/](docs/)。

## 功能特点

- **工具执行**：内置文件读取、写入、编辑、路径查找、Bash 命令和图片读取工具。
- **多模态输入**：支持文本、本地图片及 HTTP(S) 图片 URL，可在一次请求中传入多张图片。
- **持久会话**：通过 SQLite Checkpoint 保存上下文，支持跨进程续聊，并记录会话元数据和模型、工具调用轨迹。
- **任务协作**：支持 Todo 任务规划，以及具有独立提示词、工具范围和记忆模式的 SubAgent 委派。
- **能力扩展**：按需加载本地 Skill，通过 stdio 或 Streamable HTTP 接入 MCP 工具、资源和提示词。
- **运行控制**：支持工具权限与人工审批、模型调用预算、命令超时，以及上下文用量估算和推理强度配置。

## 快速开始

### 环境要求

- Python 3.12 或更高版本。
- uv，用于管理依赖和运行项目。
- 支持工具调用的 OpenAI 或 OpenAI 兼容模型服务；使用图片功能时，模型还需支持视觉输入。

### 安装依赖

在仓库根目录执行，首次使用时复制配置模板：

```bash
uv sync
cp .agents/example.toml .agents/config.toml
```

### 配置模型

编辑 `.agents/config.toml` 中的 `[model]`，填写模型名称、服务地址和 API Key：

```toml
[model]
model = "gpt-4o-mini"
base_url = ""
api_key = "your-api-key"
```

`base_url` 留空时使用默认 OpenAI 地址；接入兼容服务时填写其 API 地址。模型配置从 TOML 读取，不加载 `.env`。本地配置文件已加入 Git 忽略规则。

完整模板还包含工具选择、运行限制、子代理和 MCP 配置，见 [.agents/example.toml](.agents/example.toml)。

### 启动交互

```bash
uv run rsh
```

输入问题即可开始对话，输入 `/help` 查看交互命令。新会话默认以仓库根目录的 `workdir/` 为工作目录，不存在时自动创建；数据集评测任务的脚本与产物应在该目录中组织。

## 使用方法

### 单次请求与图片输入

```bash
# 文本请求
uv run rsh ask "查看工作目录并概述其中的文件"

# 图片请求：替换为实际图片路径，也可使用 HTTP(S) URL
uv run rsh ask --image /path/to/image.png "描述图片中的主要内容"

# 以 JSON 输出结果，便于脚本调用
uv run rsh ask --json "列出当前任务的执行步骤"
```

`--image` 可重复使用。未指定 `--session-id` 时，`ask` 会创建新会话。

### 会话管理

```bash
# 创建或列出会话
uv run rsh create-session
uv run rsh list-session

# 将 SESSION_ID 替换为实际会话 ID，恢复交互或发送单次请求
uv run rsh --session-id SESSION_ID
uv run rsh ask --session-id SESSION_ID "继续上一个任务"

# 关闭会话
uv run rsh close-session --session-id SESSION_ID
```

### 交互命令

| 命令 | 作用 |
| --- | --- |
| `/help` | 查看帮助 |
| `/status` | 查看当前会话状态 |
| `/context` | 查看上下文估算用量和分类占用 |
| `/skill` | 查看已安装的 Skill |
| `/mcp` | 查看配置中的 MCP 服务及工具 |
| `/reasoning` | 查看推理强度；如 `/reasoning high` 可修改配置 |
| `/quit`、`/exit` | 退出交互，保留会话以便续聊 |
| `/close` | 关闭会话并退出 |

工具需要审批时，交互终端会提示输入 `y/yes` 或 `n/no`。非交互调用遇到待审批操作时以退出码 `2` 暂停，可在交互终端通过 `--session-id` 恢复处理。

### 指定配置与工作目录

```bash
uv run rsh --config .agents/config.toml --workdir ./workdir
uv run rsh --help
uv run rsh ask --help
```

默认读取启动目录下的 `.agents/config.toml`。`--workdir` 用于指定新会话的工作目录，已有会话沿用创建时的目录。

## 配置与扩展

- **内置工具**：通过 `[agent].tools` 选择主 Agent 可用的工具。
- **Skill**：放置在 `.agents/skills/<name>/SKILL.md`，由 Agent 按需调用 `load_skill` 加载；使用 `uv run rsh list-skill` 查看目录。
- **SubAgent**：通过 `[[subagents]]` 配置角色、提示词、工具范围和记忆模式；模板提供 Researcher 和 Engineer 两个角色。
- **MCP**：通过 `[mcp.servers.<id>]` 配置服务，主 Agent 按需连接；使用 `uv run rsh list-mcp` 查看服务及工具。

配置中的相对资源路径以 TOML 文件所在目录为基准解析，`checkpoint_file` 相对于会话存储目录解析。

## 项目结构

```text
RS-Harness/
├── src/rs_harness/       # Agent、CLI、配置、工具与会话管理
├── .agents/
│   ├── example.toml     # 配置模板
│   ├── config.toml      # 本地配置（首次使用时创建）
│   ├── prompts/         # 主 Agent 与子代理提示词
│   ├── skills/          # 可选的本地技能目录
│   └── sessions/        # 运行时生成的会话、轨迹与 Checkpoint
├── docs/                # 设计、实现与验收文档
├── tests/               # 单元、集成与真实模型测试
├── workdir/             # 任务工作目录
├── pyproject.toml       # 项目元数据与依赖声明
└── uv.lock              # 依赖锁文件
```

## 开发与测试

```bash
# 离线测试，默认排除 live 测试
uv run pytest --tb=short

# 配置好模型服务后，运行需要网络和真实模型的测试
uv run pytest -m live --tb=short
```
