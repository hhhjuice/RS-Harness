# RS-Harness v0.2.1 设计记录：循环简化

状态：已实现
日期：2026-09-12
前置文档：`docs/Harness设计v0.2.0-任务规划.md`

## 1. 动机

v0.2.0 交付后 `agent_loop.py` 膨胀到 560 行：`run` / `_interact` / `_drive_turns` 三层结构中 `_interact` 只剩薄前置；脱敏逻辑散落三处（trace / hooks / agent_loop 各自维护正则与键名集合）；bash 硬拒清单存在两份且不一致；账本每次交互全文件重读约 4 遍。RSH 处于开发阶段、主要使用方式为 CLI 交互模式，无需为不存在的嵌入方维护兼容 API。v0.2.1 在不改变外部行为契约的前提下做一次结构性简化。

## 2. 功能概览

| 项 | 行为 |
| --- | --- |
| 前置清理 | 删 `system_context_version` 版本戳及 `_interact` 入口校验（开发阶段不维护旧会话）；`messages_for_model` assistant 白名单删 legacy `function_call` 字段；`_interact` 内部去重（`UserPromptSubmit` 单次触发、`_append_user_message` 统一 append 错误包装、workdir 检查去 `except _LoopFailure: raise` 空转） |
| 无用 API 删除 | `AgentLoop` facade 类（全仓库零调用）、`Session.append_message`（src 零调用，测试改 `with session.locked(): session.append(...)`）、`run_read` 双签名兼容（"for embedders" 的 limit/workdir 交换舞蹈） |
| 结构合并 | `run()` 吞并 `_interact`：校验、hook、入账、client 构造内联进持锁 try 块；成功路径先赋值 `result` 再 return，保住 `finally` 的 Stop hook 补触发判断；`run` 对外签名不变 |
| 脱敏三合一 | 新增 `runtime/redact.py` 作为唯一定义源：`CREDENTIAL_KEYS`、credential `key=value` 正则、两套 data URI 语义（任意位置 sub / 前缀整值摘要）、`redact_text` / `redact_value` / `secret_redactor` |
| 黑名单统一 | `BASH_DENY_LIST` 落 `tools.py` 单源（7 项超集），`check_permission` 门与 `run_bash` 内层 fence 共用；直调 `dispatch_tool("bash", ...)` 新增拦 `mkfs` / `dd if=`；`DESTRUCTIVE` 警示级留 hooks 走审批 |
| 账本缓存 | `MessageLedger` / `EventLedger` 构造时单遍读入 `_records`，`append` 同步追加，`records()` / `messages()` O(1) 返回新列表壳（内层 dict 共享、调用方只读）；坏行容错统一为跳过（此前 `records()` 遇坏行抛错） |

## 3. 模块变更

- `runtime/redact.py`（新增）：统一脱敏模块；`Redactor` 类型、`identity`、`redact_text`（redactor → data URI sub → credential 正则）、`redact_value`（递归结构，异常映射 `{"type", "message"}`）、`secret_redactor`（空 secret 返回恒等）、`summarize_data_uri`（整值摘要）。
- `runtime/hooks.py`：删 `_sanitize_text` / `_sanitize_value` 与本地正则；`_record` / `tool_error_hook` / `log_hook` 改调 redact；`Redactor` 改从 redact 导入；`HookContext.redact` 字段与 `default_hooks(redactor=)` 接口不变；`check_permission` 改用 `BASH_DENY_LIST`，`from .tools import safe_path` 上提到模块顶部（无环）。
- `agent_loop.py`：560 → 493 行。删 `AgentLoop`、`_redact_secrets`、`_interact`；`run()` 内联交互前置；`_safe_exception_message` 改用 `redact_text`（顺带获得 data URI / credential 正则，行为增强）；registry redactor 改 `secret_redactor(os.getenv(...))`。
- `runtime/session.py`：删 `append_message`。
- `runtime/tools.py`：`run_read` 签名固定 `(path, limit=None, workdir=None, *, allow_workspace_escape=False)`；新增 `BASH_DENY_LIST` 常量。
- `runtime/trace.py`：`_json_safe` 保留序列化职责，credential 键名集合与整值 data URI 摘要委托 redact；两个 Ledger 类改内存缓存，公共读取逻辑抽为 `_read_jsonl` / `_next_seq_after`；`_json_safe` 符号位置不动（`session.py` / `messages.py` import 零改动）。
- 导出：包根 `__init__.py` 删 `AgentLoop`（`runtime/__init__.py` 本就未导出）；redact 模块保持 runtime 内部（`Redactor` 经 hooks 命名空间仍可导入）；包版本升至 `0.2.1`。
- `AGENTS.md`：目录结构同步（`agent_loop.py` 导出列表、`redact.py` 条目）。

## 4. 测试覆盖（tests/unit/）

- `test_redact.py`（新增 8 条）：`redact_text` 四契约（secret 替换 / data URI 任意位置 / `key=value` 正则 / 恒等 redactor 下正则仍生效）、普通文本不动、`redact_value` 递归与异常映射、空 secret 恒等、`summarize_data_uri` 整值边界。
- `test_runtime_session.py`：`append_message` 调用点全部改持锁 `append`；新增列表壳隔离（外层 append 不污染账本）与坏行跳过 + seq 续接两条。
- `test_tools.py`：新增 `mkfs` / `dd if=` 直调拦截断言。
- `test_agent_loop.py`：`run()` 后读账本改 `Session.open` 重开读（跨对象缓存语义，见 §5）。
- 既有脱敏集成测试（工具输出 `[REDACTED]` 进请求与账本、恒等 redactor 下 `tool_error_hook` 仍拦、账本键名/整值、data URI 不落账本）全部作为回归闸门未改动。
- 全量 117 → 132 条通过。

## 5. 边界与约束

- 行为契约不变：`run` 签名、`InteractionResult` 语义、hook 触发顺序与次数、锁 / fsync / 原子写、events 审计降噪（`_SILENT`）、`run(hooks=)` 扩展点均保留。
- **跨对象读语义变化（本版唯一对外可见变化）**：`MessageLedger` 缓存后，一个 `Session` 对象不再看到其他进程或对象对同一账本的追加；需要最新视图必须 `Session.open` 重开。CLI 的 ask 子命令与交互 REPL、`close-session` 本就如此使用。锁内 `reload()` 仍会在每次 `run()` 开始时重建 ledger（读一次磁盘），多进程安全性不变。
- 坏行容错：`_read_jsonl` 对缺 `seq`、非法 JSON、空行统一跳过；`records()` 不再因坏行抛错，与 seq 扫描策略一致。
- `records()` / `messages()` 返回的 message dict 为账本内部对象，调用方必须只读；外层列表为新壳，可安全 append。已审计全部调用方（`_drive_turns`、`messages_for_model`、CLI 历史格式化、测试）均满足。
- 直调 `dispatch_tool("bash", ...)` 的内层 fence 现与 PreToolUse 门同源同集，绕过 hook 的路径不再缺拦 `mkfs` / `dd if=`。
- 遥感栅格语义仍在 MCP 边界外，本版本不涉及。

## 6. 数字

- `agent_loop.py` 560 → 493 行；src 总量净减约 28 行（各处删除合计约 124 行，新增 `redact.py` 96 行对冲）。
- 每次 `run()` 的账本全量读：4 次 → 2 次（`Session.open` 构造 + `locked` 内 `reload` 各一次）；`messages()` O(1)（实测 2000 条账本 50 次读取 1.0ms）。
- bash 硬拒清单：2 份不一致 → 1 份单源 7 项。
