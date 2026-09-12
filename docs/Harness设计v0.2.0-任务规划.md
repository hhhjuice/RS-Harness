# RS-Harness v0.2.0 设计记录：任务规划（todo_write）

状态：已实现
日期：2026-09-11
前置文档：`docs/初版Harness设计v0.1.0.md`
参考实现：[shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code) s05 TodoWrite（仅作功能设计参考）

## 1. 动机

长任务中模型容易漂移：工具观察不断填入上下文，最初的计划被挤出注意力，模型开始即兴发挥、漏掉步骤。v0.2.0 给循环增加一个**只负责规划状态**的工具 `todo_write`，让模型先列出步骤、再执行并持续更新状态。它不给模型任何额外执行能力——工作仍由既有六个工作区工具完成。

## 2. 功能概览

| 能力 | 行为 |
| --- | --- |
| `todo_write` 工具 | 整体替换会话任务列表，返回渲染文本作为 tool observation |
| 校验规则 | 最多 20 项；`content` 非空；`status` ∈ `pending` / `in_progress` / `completed`；最多一项 `in_progress`；先完整校验再替换（失败时旧列表保持不变） |
| 字符串输入 | 接受 JSON 数组字符串或 Python 列表字面量字符串（`ast.literal_eval`，不用 `eval`），解析失败返回错误观察 |
| 渲染格式 | `[ ]` 待办、`[>]` 进行中、`[x]` 已完成，末行 `(done/total completed)`；空列表渲染为 `No todos.` |
| 持久化 | 任务列表存于 `session.json` 的 `SessionState.todos`，每次成功更新后随会话保存；重开会话继续更新；旧格式 session（无 `todos` 字段）默认空列表可正常加载 |
| 提醒计数 | 一次 `run()` 内维护 `rounds_since_todo`：连续 3 个有工具调用但未成功更新 todo 的回合后，向下一次模型请求追加 `<reminder>Update your todos.</reminder>`；计数仅在 `todo_write` 校验通过、列表实际替换后归零（被拒更新不归零），每次新的 `run()` 重新开始 |
| reminder 落盘 | 仅存在于内存中的模型消息，不写入 `messages.jsonl`，不污染持久账本 |
| 终端显示 | 更新成功后经 console/hook 通道显示黄色 `[HOOK] ## Current Tasks`（`TODO_NOTICE_PREFIX` 常量，CLI 据此选色）+ 渲染列表；作为 hook 通知输出，不进入 `--json` payload |
| 权限 | 不涉及工作区路径，无需 `permission_callback` 审批 |

## 3. 模块变更

- `runtime/todos.py`（新增）：`TodoItem`（Pydantic，`extra="forbid"`）、`TodoStatus` 字面量、`TodoManager`（校验 + 原子替换 + 渲染）、`run_todo_write()`（`ValueError` 转错误观察）。
- `runtime/tools.py`：`_TOOL_SPECS` 注册 `todo_write`（schema：`todos` 数组 `maxItems=20`，项内 `content` 必填、`status` 为枚举），工具总数 7；`dispatch_tool()` 增加可选 `todo_manager` 上下文，不传时独立调用仍可工作。
- `runtime/session.py`：`SessionState.todos: list[TodoItem]`，默认空列表。
- `agent_loop.py`：`_drive_turns()` 每次 `run()` 从 session 状态初始化 `TodoManager` 并维护提醒计数；`_execute_tool_calls()` 返回本轮是否调用过 `todo_write`（同一批次多个工具调用中任一调用即视为已更新），执行后把 manager 状态同步回 `session.state`。
- `runtime/hooks.py`：新增 `todo_display_hook`（PostToolUse 观察者），注册进 `default_hooks`，经 console 通道输出 `[HOOK] ## Current Tasks`；观察者 hook 未激活时返回内部哨兵，不写事件账本（见 §5 降噪语义）。
- `cli.py`：`_cli_console` 对 `## Current Tasks` 通知使用黄色（33），其余 hook 通知保持灰色；非 TTY 仍为纯文本。
- 导出：`TodoItem` / `TodoManager` / `run_todo_write` / `todo_display_hook` 经 `runtime/__init__.py` 与包根 `__init__.py` 再导出；包版本升至 `0.2.0`。
- `prompts/system.md`：新增"复杂/多步任务先 `todo_write` 计划、执行中更新状态"约束。

## 4. 测试覆盖（tests/unit/）

- `test_todos.py`：渲染格式与完成计数、空列表、JSON/Python 字符串解析、非法字符串、非列表输入、空内容、非法状态、非对象项、超过 20 项（含恰好 20 项边界）、多个 `in_progress`、校验失败旧列表不变、`TodoItem` 实例输入、schema 注册、dispatch 带与不带 manager、错误观察返回、旧格式 session.json 恢复。
- `test_agent_loop_tools.py`：todo 状态写入 `session.json` 并可重开继续、三轮未更新触发 reminder 且 reminder 不入 ledger、`run()` 间计数重置、多工具批次任一成功 `todo_write` 重置计数、被拒 `todo_write` 不重置计数、console 收到 `[HOOK] ## Current Tasks` 通知。
- `test_hooks.py` / `test_tools.py` / `test_cli.py`：默认 PostToolUse 钩子顺序含 `todo_display_hook`、干净调用零 PostToolUse 事件且仅留权限审计行、激活观察者（出错 / 超长 / 成功显示）仍记录、工具名清单含 `todo_write`、既有权限/文件/图片/JSON 输出行为不回归。

## 5. 边界与约束

- todo 状态属于 session 规划元数据：不写入工作区文件，不进入消息账本（工具观察仍按普通 tool message 记录）。
- `todo_write` 无执行语义，复用现有 dispatch、hooks、权限与 JSONL 消息循环，不新增执行能力。
- reminder 是运行时提示而非用户输入：只影响当次模型请求，不落盘、不跨 `run()` 累积。
- 事件账本降噪：无动作的 hook 不写事件。`log_hook` 恒静默（console-only），`tool_error_hook` / `large_output_hook` / `todo_display_hook` 仅在激活（出错 / 超长输出 / 成功显示）时记录；`check_permission` 对每次工具调用保留一条权限审计事件（含参数预览），`UserPromptSubmit` / `Stop` 照常记录。干净工具调用从 5 条事件降为 1 条。
- 遥感栅格语义仍在 MCP 边界外，本版本不涉及。
