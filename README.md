# RS-Harness v0.0.3

阶段 1 提供基于 LangChain 的 Agent、SQLite 持久多轮上下文、本地 session 元数据、运行记录和 Typer CLI。阶段 2 增加官方 AgentMiddleware 组合入口、每轮模型调用预算和运行时 context。阶段 3 接入五种基础工具、工具轨迹，以及本地和 URL 图片输入。

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
uv run rsh ask --config config/default.toml --image ./sample.png "描述图片"
uv run rsh ask --config config/default.toml --image ./sample.png --json
```

`--image` 可重复使用，图片和文本组成同一条标准消息；省略文本可提交仅图片输入。新会话的相对图片路径按 `--workdir` 或当前目录解析，续聊按 session 原工作目录解析。用户明确指定的绝对图片路径可以位于工作目录外。输入无效时不会创建新会话；`ask --json` 返回 JSON，错误退出码为 1。

本地图片按文件内容识别 JPEG、PNG、GIF、WebP，编码为带 MIME 的 base64 标准内容块。图片数据随 SQLite 保存，原文件删除后仍可续聊；HTTP(S) URL 直接交给 provider 获取，恢复后仍依赖 URL 可达。交互模式可用 `--session-id` 继续图片会话，图片提交入口目前为 `ask --image`。

已验证 `api.deepseek.com` 的 `deepseek-flash` 工具和视觉输入。[该服务的视觉限制](https://api-docs.deepseek.com/guides/vision)包括：单张内联图片 32 MiB、完整请求体 48 MiB、每请求最多 600 张图、无 file_id 时图片总计 64 MiB、URL 最长 8192 字符。本地图片统一受 harness 级 32 MiB 读取上限保护；DeepSeek 附加限制在读取/编码前及每次模型调用（含历史）时检查。完整请求 JSON 开销、远程下载、图片解码及尺寸限制由 provider 最终验证。该服务的单边尺寸上限为 8192 像素，15 张图起为 4096 像素；本地不缩放或转码。其他服务不套用 DeepSeek 附加限制，支持情况及更严格的限制由相应 provider 决定；失败会明确返回。

生产 Agent 提供 `read_file/write_file/edit_file/glob/bash`。文件工具限制在 session workdir 内并检查符号链接；read_file 按行读取 UTF-8 文本，write_file 创建或覆盖，edit_file 只替换唯一的精确匹配，glob 返回最多 200 条排序路径。父目录必须已存在。工具参数或预期执行错误通过官方 error ToolMessage 反馈给模型，未知异常终止本轮。

`[runtime].bash_timeout_s` 默认 60 秒，`max_tool_output_bytes` 默认 65536 字节；后者限制工具正文，Bash 的 stdout/stderr 共用预算。超时和取消会清理进程组，输出过长会截断但继续排空管道。Bash 的 cwd 不是操作系统沙箱，命令可访问当前用户拥有权限的路径。工具目前直接执行，人工审批在阶段 4 接入。同批工具默认可并发，有依赖的操作需等待前一步结果再发起。

交互模式可连续输入多轮问题，整个交互过程共用一个异步事件循环。`/quit` 或 `/exit` 保留会话，后续通过 `--session-id` 继续；`/close` 关闭会话。

终端输入支持中文编辑：左右键移动光标，Backspace/Delete 删除字符，Home/End 移至行首/行尾，上下键浏览本次运行的输入历史。历史仅保存在内存中；Enter 提交问题，输入提示符处 Ctrl+C 或空行 Ctrl+D 退出并保留会话。重定向或管道输入使用普通文本读取。

模型配置全部从 TOML 读取，不加载 `.env` 或从环境变量读取 API key。`config/default.toml` 已被 Git 忽略，仓库保留不含密钥的 `config/example.toml`。

`session_id` 是 RSH 唯一保存的会话标识；调用 LangGraph 时直接用它作为 `configurable.thread_id`，元数据、轨迹和 CLI 输出不再重复保存 `thread_id`。

`session.json` 只保存会话元数据，SQLite Checkpoint 是模型上下文来源，`trajectory.jsonl` 只用于查看交互与模型/工具调用事件。用户输入记录保存文本块和图片摘要（MIME、字节数、SHA-256）；本地图片 base64 不进入轨迹，完整内容仍由 Checkpoint 保存。工具事件按 run_id/parent_run_id 关联。轨迹中的输入、输出是观测副本，不参与上下文恢复；即使轨迹丢失、损坏或无法写入，模型仍从 Checkpoint 续聊，不因此重试工具。

当前开发版本不迁移旧 `session.json` 格式；元数据中的未知字段（包括 `thread_id`）会被拒绝。session 目录名和 `session.json.session_id` 必须一致，否则 session 不会打开。

默认测试使用 `uv run pytest --tb=short`，自动排除 `live` 测试；配置好支持工具及视觉的模型后，使用 `uv run pytest -m live --tb=short` 验证三轮文本、五工具、本地/URL 图片、仅图片及跨进程图片记忆。live 测试将会话数据写入临时目录，新冒烟使用独立进程，与实际 CLI 生命周期一致。

v0.0.3 离线回归 134 项、真实模型验收 6 项通过，详见 [阶段三实施文档](./docs/v0.0.3——五种基础工具接入与多模态输入.md)。本阶段已通过真实模型三轮文本复验，补足阶段二此前因网络无法完成的冒烟；Permission、Todo 和 SubAgent 按后续阶段实施。

Python 调用方应在同一事件循环内连续 `await ask(...)`。锁定版本的 langchain-openai 会缓存默认异步 HTTP 客户端，同一进程中反复用 `asyncio.run` 创建新循环可能触发 `Event loop is closed`；CLI 每次运行使用独立进程，交互模式保持同一循环。
