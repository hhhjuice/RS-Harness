# AGENTS.md — RS-Harness

## 项目定位

最小化、模型驱动的 Agent Harness，面向遥感分析 Agent。会话即持久模型上下文：全部 system/user/assistant/tool 消息追加到 JSONL 账本。通过 OpenAI 官方 SDK 真实调用模型（默认 DeepSeek，key 从 `.env` 的 `OPENAI_API_KEY` 读取），无离线 Stub。遥感栅格语义（GeoTIFF 等）留在 MCP 边界外——harness 不解码像素。

改设计前先读 `docs/初版Harness设计v0.1.0.md`。

## 常用命令

```bash
uv sync                                # 安装依赖（Python >= 3.12，uv 管理）
uv run pytest                          # 全部测试（pytest 配置已含 pythonpath=["src"]）
uv run pytest tests/unit/test_hooks.py # 单文件测试
uv run pytest tests/unit -k <名称>     # 按名称过滤
uv run rsh --config config/default.toml --workdir .   # 交互式会话
uv run rsh ask --config config/default.toml "问题"    # 单次请求
```

无独立 lint/typecheck 配置，遵循现有代码风格。

## 目录结构

- `src/rs_harness/agent_loop.py` — Agent Loop 核心：`run` / `create_session` / `close_session` / `AgentLoop` / `InputRequest`
- `src/rs_harness/cli.py` — Typer CLI（`ask` / `create-session` / `close-session` / `list-session`，无子命令时进交互模式）
- `src/rs_harness/config.py` — TOML Config 加载与校验
- `src/rs_harness/runtime/` — 会话与工具运行时：
  - `session.py` 会话状态与生命周期；`tools.py` 六个工具与 `safe_path` 围栏；`hooks.py` HookRegistry 与权限回调；`messages.py` 消息构造与工具参数解码；`images.py` 图像附件；`trace.py` messages/events 双 JSONL 账本
- `config/` — `default.toml`（实际运行）、`example.toml`（模板）
- `prompts/system.md` — 系统提示词，必须包含 `{workdir}` 占位符
- `tests/unit/` — 按模块拆分的单测，不真实调用模型
- `sessions/<session_id>/` — 运行时产物（`session.json` / `messages.jsonl` / `events.jsonl`），勿手动修改
- `workdir/` — 工具工作区示例目录

## 架构边界

- 分层依赖方向固定：`cli → agent_loop → runtime`；`runtime` 只依赖 `..config` 和自身子模块，禁止反向 import。
- 公共 API 经 `src/rs_harness/__init__.py` 显式导出。新增公共符号需同步 `__all__` 与 `runtime/__init__.py` 的再导出。
- 循环只暴露六个工具：`bash` / `read_file` / `write_file` / `edit_file` / `glob` / `read_image`。新工具要同时更新 `TOOLS` schema、`TOOL_HANDLERS` 与对应测试。
- 文件工具路径受 `safe_path` 围栏限制在会话 workdir 内；越界必须经 `permission_callback` 批准且仅单次生效。硬拒绝的 Bash 模式不可批准。
- 会话创建时绑定 workdir，恢复时不可更换。

## 关键约定

- Pydantic v2（`BaseModel` + `ConfigDict` + `field_validator`），模型类默认 `extra="forbid"`。
- 文件顶部 `from __future__ import annotations`，类型用现代写法 `X | None`。
- 工具执行失败不抛异常炸循环：错误作为 observation 返回给模型，由其自行恢复。
- 安全脱敏：工具输出、账本、日志在任何写出前按配置的 API key 脱敏；事件账本中参数与输出先截断再落盘。
- 图像：web 格式（JPEG/PNG/WebP/GIF，≤ 32 MB）经 `--image` 或 `glob` + `read_image` 进入；账本只存 `image_ref`（path/mime/size/sha256），字节每次请求时从磁盘重读。查询文本不扫描图片路径。
- CLI `--workdir` 可放在子命令前或后，子命令值优先。
- 提交信息用 Conventional Commits（见 git log：`feat` / `fix` / `refactor` 等）。

## 注意事项

- `sessions/`、`workdir/`、`artifacts/`、`.mimosa/` 为运行时/工具产物，勿提交、勿清理用户数据。
- `.env` 不入库；本地运行前从 `.env.example` 复制并填 key。
- 带未完成历史 tool call 的会话必须替换而非重放旧动作。
