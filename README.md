# RS-Harness v0.0.2

阶段 1 提供基于 LangChain 的无工具文本 Agent、SQLite 持久多轮上下文、本地 session 元数据、运行记录和 Typer CLI。阶段 2 增加官方 AgentMiddleware 组合入口、每轮模型调用预算和运行时 context。

```bash
uv sync
cp config/example.toml config/default.toml
```

在 `config/default.toml` 的 `[model]` 中填写 `model`、`base_url` 和 `api_key`；复制示例仅用于首次配置，已有本地配置无需覆盖。

`[runtime].max_model_calls` 默认为 `10`，限制一次交互中的模型循环调用次数。达到预算后，下一次模型调用返回 `ModelCallLimitExceededError`，交互以失败结果结束；新的交互重新获得预算。工作目录通过每次调用的 runtime context 注入 middleware 和后续工具，context 对象不作为消息历史保存；系统提示词保留现有工作目录说明。

```bash
uv run rsh --config config/default.toml
uv run rsh --config config/default.toml --session-id <SESSION_ID>
uv run rsh create-session --config config/default.toml
uv run rsh ask --config config/default.toml "你好"
```

交互模式可连续输入多轮问题，整个交互过程共用一个异步事件循环。`/quit` 或 `/exit` 保留会话，后续通过 `--session-id` 继续；`/close` 关闭会话。

终端输入支持中文编辑：左右键移动光标，Backspace/Delete 删除字符，Home/End 移至行首/行尾，上下键浏览本次运行的输入历史。历史仅保存在内存中；Enter 提交问题，输入提示符处 Ctrl+C 或空行 Ctrl+D 退出并保留会话。重定向或管道输入使用普通文本读取。

模型配置全部从 TOML 读取，不加载 `.env` 或从环境变量读取 API key。`config/default.toml` 已被 Git 忽略，仓库保留不含密钥的 `config/example.toml`。

`session_id` 是 RSH 唯一保存的会话标识；调用 LangGraph 时直接用它作为 `configurable.thread_id`，元数据、轨迹和 CLI 输出不再重复保存 `thread_id`。

`session.json` 只保存会话元数据，SQLite Checkpoint 是模型上下文来源，`trajectory.jsonl` 只用于查看交互与调用事件。轨迹中的输入、输出是观测副本，不参与上下文恢复；即使轨迹丢失、损坏或无法写入，模型仍从 Checkpoint 续聊。

当前开发版本不迁移旧 `session.json` 格式；元数据中的未知字段（包括 `thread_id`）会被拒绝。session 目录名和 `session.json.session_id` 必须一致，否则 session 不会打开。

默认测试使用 `uv run pytest --tb=short`，自动排除 `live` 测试；配置好 `model.api_key` 后，使用 `uv run pytest -m live --tb=short` 验证 CLI 连续三轮真实模型对话及前文记忆。live 测试将会话数据写入测试临时目录。

v0.0.2 开发完成，默认回归 62 项通过，覆盖 SQLite middleware 状态恢复、临时模型请求不进入历史、预算超限的 CLI JSON/退出码与继续交互。真实模型冒烟此前出现连接错误，本次联网复验被自动审批服务 503 阻塞，尚未通过。详见 [v0.0.2 实施记录](./docs/v0.0.2——AgentMiddleware%20扩展.md)。生产 Agent 当前仍不注册基础工具、Permission 或 Todo。
