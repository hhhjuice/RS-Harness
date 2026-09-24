# RS-Harness v0.0.5

阶段 1 提供基于 LangChain 的 Agent、SQLite 持久多轮上下文、本地 session 元数据、运行记录和 Typer CLI。阶段 2 增加官方 AgentMiddleware 组合入口、模型调用预算和运行时 context。阶段 3 接入五种基础工具、工具轨迹，以及本地和 URL 图片输入。阶段 4 接入条件审批、硬拒绝，以及跨进程审批恢复。阶段 5 接入官方任务规划和持久任务进度。

```bash
uv sync
cp config/example.toml config/default.toml
```

在 `config/default.toml` 的 `[model]` 中填写 `model`、`base_url` 和 `api_key`；复制示例仅用于首次配置，已有本地配置无需覆盖。

`[runtime].max_model_calls` 默认为 `10`，限制一次 `ainvoke` 执行段中的模型调用次数。达到预算后，下一次模型调用返回 `ModelCallLimitExceededError`；新的普通交互或审批恢复重新获得预算，不跨恢复累计。工作目录通过每次调用的 runtime context 注入 middleware 和工具，context 对象不作为消息历史保存；系统提示词保留现有工作目录说明。

Agent 已接入 Stop Hook 协议，默认检查列表为空，不预设任务验收规则。在 `src/rs_harness/agents/middleware.py` 的 `build_middleware` 中，将末尾的 `StopHookMiddleware()` 改为 `StopHookMiddleware(hooks=(first_check, second_check))`，并定义或导入对应检查函数，即可集中注册；CLI、普通提问和审批恢复使用同一套注册。

`StopHook` 接收完整的 `list[BaseMessage]` 消息历史（含本次准备返回的回答），支持返回 `str | None` 的同步函数或异步函数。历史仅供读取，检查函数不应原地修改消息。仅在最后一条消息为没有工具调用的 `AIMessage` 时，按注册顺序运行检查：

| 返回值 | 行为 |
| --- | --- |
| `None` | 继续下一个检查；全部通过后正常结束 |
| 非空字符串 | 停止后续检查，追加对应的 `HumanMessage`，继续模型—工具循环 |
| 空字符串 `""` | 停止后续检查并正常结束，不追加消息 |
| 其他类型 | 抛出 `TypeError`，本轮失败 |

续跑使用官方 `aafter_agent` 跳转，仍处于同一次 `ainvoke`，不重置模型预算；持续要求续跑最终会触发现有预算错误。注入的提示随消息进入 Checkpoint，不创建新的用户交互或 `user_input` 记录。工具执行阶段和审批暂停时不检查；续跑可以再次调用工具或等待审批，审批恢复完成后再检查。检查异常按现有失败流程处理，调用任务的取消继续向上传播，最终只返回允许结束时的回答。

```bash
uv run rsh --config config/default.toml
uv run rsh --config config/default.toml --session-id <SESSION_ID>
uv run rsh create-session --config config/default.toml
uv run rsh ask --config config/default.toml "你好"
uv run rsh ask --config config/default.toml --image ./sample.png "描述图片"
uv run rsh ask --config config/default.toml --image ./sample.png --json
```

`--config`、`--session-id` 和 `--workdir` 可放在子命令前后。放在子命令前时作为缺省值，子命令上的显式参数优先。

`--image` 可重复使用，图片和文本组成同一条标准消息；省略文本可提交仅图片输入。新会话的相对图片路径按 `--workdir` 或当前目录解析，续聊按 session 原工作目录解析。用户明确指定的绝对图片路径可以位于工作目录外。输入无效时不会创建新会话；`ask --json` 返回 JSON，错误退出码为 1。

本地图片按文件内容识别 JPEG、PNG、GIF、WebP，编码为带 MIME 的 base64 标准内容块。图片数据随 SQLite 保存，原文件删除后仍可续聊；HTTP(S) URL 直接交给 provider 获取，恢复后仍依赖 URL 可达。交互模式可用 `--session-id` 继续图片会话；用户可通过 `ask --image` 提交图片，也可在对话中让 Agent 调用 `read_image` 查看本地图片。

已验证 `api.deepseek.com` 的 `deepseek-flash` 工具和视觉输入。[该服务的视觉限制](https://api-docs.deepseek.com/guides/vision)包括：单张内联图片 32 MiB、完整请求体 48 MiB、每请求最多 600 张图、无 file_id 时图片总计 64 MiB、URL 最长 8192 字符。本地图片统一受 harness 级 32 MiB 读取上限保护；DeepSeek 附加限制在读取/编码前及每次模型调用（含历史）时检查。完整请求 JSON 开销、远程下载、图片解码及尺寸限制由 provider 最终验证。该服务的单边尺寸上限为 8192 像素，15 张图起为 4096 像素；本地不缩放或转码。其他服务不套用 DeepSeek 附加限制，支持情况及更严格的限制由相应 provider 决定；失败会明确返回。

生产 Agent 提供 `read_file/read_image/write_file/edit_file/glob/bash`。文件路径按 session workdir 解析并跟随符号链接；区内读写编辑直接执行，区外访问需批准当前调用，批准后可实际访问，结果中使用绝对路径。read_file 按行读取 UTF-8 文本，write_file 创建或覆盖，edit_file 只替换唯一的精确匹配，glob 保持工作区内查询，返回最多 200 条排序路径。父目录必须已存在。工具参数或预期执行错误通过官方 error ToolMessage 反馈给模型，未知异常终止本轮。

`read_image(path)` 读取单张本地 JPEG、PNG、GIF 或 WebP，按文件头识别格式，单图上限 32 MiB；不下载 URL，不支持 TIFF/GeoTIFF，不缩放或转码。成功后模型会收到图片本身及路径说明；缺失、空文件、格式错误或超限会返回工具错误。图片随工具结果保存到 Checkpoint，删除原图后仍可恢复；发送模型请求时，将图片放在同批全部工具结果之后的用户图片消息中，不向会话重复保存附加消息。工具读图和用户上传图片共同接受完整历史的图片限制检查。

Agent 还通过 `TodoListMiddleware` 获得官方 `write_todos` 工具，可为复杂任务创建、更新和完成清单。每项包含 `content` 与 `pending/in_progress/completed` 状态，每次提交完整列表替换旧计划，空列表用于清空。计划随同一 session 的 SQLite Checkpoint 恢复，普通续聊不自动清空，不同 session 隔离；不符合官方 schema 的更新返回工具错误并保留原计划。使用官方默认提示，不额外限制 20 项、单个进行中任务，也不添加每三轮提醒。

规划工具只更新状态，不执行清单中的操作，也不提供提前授权。它与需审工具同批出现时，会一起等待审批；恢复后规划正常更新，实际操作按批准或拒绝执行。同一响应中的多个 `write_todos` 由官方中间件返回错误，其他工具继续遵循原有执行机制。规划产生的后续模型调用仍计入预算；默认 Stop Hook 不强制清空未完成任务。用户可在问答中要求汇报进度，开发者可读取 `agent.aget_state(...).values.get("todos", [])`；当前 `/status` 不展示计划，CLI JSON 结构保持不变。

`[runtime].bash_timeout_s` 默认 60 秒，`max_tool_output_bytes` 默认 65536 字节；后者限制文本工具正文，Bash 的 stdout/stderr 共用预算。`read_image` 使用独立的 32 MiB 图片上限，不截断图片数据。超时和取消会清理进程组，输出过长会截断但继续排空管道。Bash 的 cwd 不是操作系统沙箱，命令可访问当前用户拥有权限的路径。同批工具默认可并发，有依赖的操作需等待前一步结果再发起。

Permission 按 `deny → ask → allow` 判定，使用官方 `HumanInTheLoopMiddleware.when` 和工具 wrap hook：

| 操作 | 处理 |
| --- | --- |
| Bash 包含 `rm -rf /`、`sudo`、`shutdown`、`reboot`、`mkfs`、`dd if=`、`> /dev/sda` | 硬拒绝，不执行且不可批准 |
| Bash 未被硬拒绝，但包含 `rm `、`> /etc/`、`chmod 777` | 等待人工确认（交互输入 `y/yes` 或 `n/no`，协议值为 approve/reject） |
| 工作目录外的 read_file/read_image/write_file/edit_file | 等待人工确认（交互输入 `y/yes` 或 `n/no`，协议值为 approve/reject） |
| 其余调用，包括工作区内创建、覆盖、编辑和普通 Bash | 直接执行；glob 保持工作区限制 |

Bash 使用区分大小写的字面子串规则，例如 `rm -rf /tmp/demo` 也被硬拒绝。引号中的关键词可能误报，空白变化或脚本间接执行可能漏报；此策略不提供 Shell 语义分析或 OS 隔离。文件审批属于 Agent 调用链，直接调用 Python 工具函数不经过审批。审批不保存长期授权；文件类型、OS 权限、超时等仍可能导致已批准调用失败。

`ask` 在 stdin 和 stderr 均为 TTY 时直接执行工具审批：显示待审批数量、工具名称和实际操作，逐项接受 `y/yes` 或 `n/no`，收齐后立即恢复执行；若恢复后再次暂停，会继续使用最新操作重新询问。`ask` 的执行退出码为成功 `0`、失败 `1`、等待审批 `2`；`--json` 的 stdout 只包含最终一个 JSON 对象，待审批时 `answer/error` 均为 null。非 TTY、管道或重定向下不消费后续输入作为批准，而是以退出码 `2` 保留暂停状态；待审 session 不接受新的普通问题，可在交互终端用 `--session-id` 继续处理或关闭。同批只要存在 ask，整个工具节点都会等待；allow/deny 不占人工决定位置，恢复后仍由框架默认调度。

交互模式可连续输入多轮问题，整个交互过程共用一个异步事件循环。TTY 下普通 `rsh>` 输入提示显示为蓝色，审批输入提示显示为黄色。触发审批时显示待审批数量、工具名称和实际操作；审批阶段只接受 `y/yes` 或 `n/no`，分别对应批准或拒绝，其他输入不会改变审批状态。空输入不批准，收齐后才提交。审批期间 Ctrl+C 或 Ctrl+D 丢弃尚未提交的决定并保留暂停，之后可通过 `--session-id` 继续；普通交互阶段仍支持 `/status`、`/close`、`/quit` 和 `/exit`。审批展示和提示写入 stderr，最终回答写入 stdout。管道或重定向模式遇到待审即以 2 退出，不消费后续输入作为批准。

交互模式和普通 `ask` 会实时显示模型调用与普通工具完成/失败通知。工具不在开始时单独输出，完成时输出 `[Tool] Running <tool> (<elapsed>s)` 及操作摘要；摘要先脱敏再截断，最多 160 个字符，长操作的差异可能被截断。写入/编辑正文只显示长度，成功结果正文不展开。命令摘要隐藏敏感变量赋值和 Authorization 头，无法确定敏感值边界时隐藏整条命令摘要。运行提示默认使用英文。中间件默认静默，仅在以下情况通知：

| 中间件 | 显示内容 |
| --- | --- |
| `ModelCallLimitMiddleware` | 预算耗尽、阻止继续调用 |
| `ImageInputMiddleware` | 图片校验失败及简短原因 |
| `HumanInTheLoopMiddleware` | CLI 展示一次待审批数量和操作；恢复且决定实际处理后，显示批准/拒绝数量 |
| `ToolDenyMiddleware` | 实际拦截的工具名和原因 |
| `ToolErrorMiddleware` | 合并到工具失败通知，显示错误类型和短原因 |
| `TodoListMiddleware` | 计划更新、清空或更新被拒绝；展开全部 Todo 项，不额外打印普通 `write_todos` 工具行 |
| `StopHookMiddleware` | 要求继续执行，或检查异常 |

人工拒绝表示未执行，待审批、取消及框架中断不打印为工具失败。已经显示过原因的错误若终止本轮，最终仅补充 `This run failed.`。摘要写入 stderr，TTY 下带颜色，重定向时为纯文本；`--json` 关闭这些通知，保持 stdout 为单个 JSON 对象。展示依据实际回调和状态，不输出模型内部思维、完整消息历史或图片数据。

Python 调用默认不输出运行摘要；可向 `ask`/`resume` 传入 `callbacks=[...]` 接入 LangChain callback handler。CLI 展示与 `trajectory.jsonl` 记录各自独立，控制台写入失败不会重试工具或改变 Agent 的结果。

同名工具并发时，各条完成通知复用该次调用开始时生成的操作摘要，输出顺序跟随完成顺序。例如：

```text
[Tool] Running bash (0.02s) {"command":"ls -la /Users/hgz/WorkSpace/ 2>&1 | head -40"}
[Tool] Running bash (0.03s) {"command":"ls -la /Users/hgz/WorkSpace/RS-Harness/ 2>&1 | head -40"}
```

交互模式的模型调用计数随新输入重置，同一输入引发的审批恢复继续累计；独立启动的 `ask` 命令也保持同一输入的完整审批链路并从 1 计数。该数字表示当前展示范围内的调用次数，执行预算仍由中间件管理。

终端输入支持中文编辑：左右键移动光标，Backspace/Delete 删除字符，Home/End 移至行首/行尾，上下键浏览本次运行的输入历史。历史仅保存在内存中；Enter 提交问题，输入提示符处 Ctrl+C 或空行 Ctrl+D 退出并保留会话。重定向或管道输入使用普通文本读取。

模型配置全部从 TOML 读取，不加载 `.env` 或从环境变量读取 API key。`config/default.toml` 已被 Git 忽略，仓库保留不含密钥的 `config/example.toml`。

`session_id` 是 RSH 唯一保存的会话标识；调用 LangGraph 时直接用它作为 `configurable.thread_id`，元数据、轨迹和 CLI 输出不再重复保存 `thread_id`。

`session.json` 只保存会话元数据，SQLite Checkpoint 是模型上下文来源，`trajectory.jsonl` 只用于查看交互与模型/工具调用事件。用户图片输入与 `read_image` 工具结果均记录文本块和图片摘要（MIME、字节数、SHA-256）；本地图片 base64 不进入轨迹，完整内容仍由 Checkpoint 保存。工具事件按 run_id/parent_run_id 关联。轨迹中的输入、输出是观测副本，不参与上下文恢复；即使轨迹丢失、损坏或无法写入，模型仍从 Checkpoint 续聊，不因此重试工具。

暂停记录 `interaction_interrupted`，有效审批提交记录 `approval_submitted`；提交事件不代表副作用已完成。恢复使用同一 session 的 `Command(resume=...)`，从 Checkpoint metadata 保留原 interaction_id，不重发用户消息。人工拒绝和硬拒绝不伪造工具执行事件。

当前开发版本不迁移旧 `session.json` 格式；元数据中的未知字段（包括 `thread_id`）会被拒绝。session 目录名和 `session.json.session_id` 必须一致，否则 session 不会打开。

默认测试使用 `uv run pytest --tb=short`，自动排除 `live` 测试；配置好支持工具及视觉的模型后，使用 `uv run pytest -m live --tb=short` 验证三轮文本、基础工具、图片、read_image 看图及恢复、审批恢复与任务规划。live 仅对临时目录中核对过的预定动作批准或拒绝；会话数据写入临时目录，审批、图片恢复及计划续接使用独立进程。

v0.0.5 交付时离线测试 335 项、真实模型验收 14 项全部通过，包含规划创建、更新、跨进程续接，以及既有五工具、文本、图片与审批能力。实现及后续修订记录见 [阶段五实施文档](./docs/v0.0.5——任务规划能力.md)。SubAgent 按阶段六实施。

Python 调用方应在同一事件循环内连续 `await ask(...)`、`await get_pending_approvals(...)` 和 `await resume(...)`。锁定版本的 langchain-openai 会缓存默认异步 HTTP 客户端，同一进程中反复用 `asyncio.run` 创建新循环可能触发 `Event loop is closed`；CLI 每次运行的提问和审批恢复保持同一循环。
