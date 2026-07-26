# own-agent 架构设计

## 项目定位

学习研究型的本地 Python 编码 agent。代码可读性好、模块边界清晰，适合理解 coding agent
的工作原理。

## 核心流程

```
用户输入 → TUI/CLI → Agent Loop → LLM Provider → 工具执行 → 返回结果 → 显示
                    ↑                                    │
                    └──────── 循环直到 finish_reason ─────┘
```

## 模块划分

### 1. `own_agent/cli.py` — 命令行入口
- 使用 `argparse` 解析参数
- 支持三种模式：单轮 (`--message`)、交互式 REPL (`--interactive`)、TUI（默认）
- 支持 `config init/path/show` 子命令

### 2. `own_agent/config.py` — 配置加载
- 读取 `~/.config/own-agent/config.toml` + 项目 `own-agent.toml`
- 支持环境变量覆盖（`OWN_AGENT_API_KEY` 等）
- 输出 `AppConfig` 数据类，供其他模块使用

### 3. `own_agent/providers/` — LLM Provider 抽象层
| 文件 | 职责 |
|---|---|
| `base.py` | `ChatProvider` 抽象基类，定义 `complete()` / `acomplete()` / `astream()` |
| `types.py` | `ChatMessage`, `ChatRequest`, `ChatResponse`, `ToolCall`, `ToolDefinition` 等共享类型 |
| `errors.py` | `ProviderError` + 错误分类（prompt_too_long / rate_limit / auth_error 等） |
| `presets.py` | 预设厂商配置（openai / deepseek / qwen / ollama / anthropic 等） |
| `openai_compatible.py` | OpenAI Chat Completions API 实现 |
| `anthropic.py` | Anthropic Messages API 实现 |

### 4. `own_agent/tools/` — 工具系统
| 文件 | 职责 |
|---|---|
| `types.py` | `Tool`, `ToolResult`, `ToolPermissionSpec` |
| `registry.py` | `ToolRegistry`：注册/查找/执行工具 |
| `builtin.py` | 组装默认工具列表 |
| `think.py` | 无副作用的思考工具 |
| `ls.py` | 列出目录内容 |
| `view.py` | 按行读取文本文件 |
| `grep.py` | 搜索文件内容 |
| `glob.py` | 按 glob 模式查找路径 |
| `write.py` | 写入文本文件 |
| `edit.py` | 替换文本片段 |
| `shell.py` | 执行 shell 命令 |
| `python_exec.py` | 执行 Python 代码 |

每个工具 = `ToolDefinition`（模型可见 schema）+ `ToolExecutor`（本地函数）+ 可选权限声明。

### 5. `own_agent/agent/` — Agent 主循环
| 文件 | 职责 |
|---|---|
| `loop.py` | `AgentLoop`：接收用户消息 → 调用 provider → 解析工具调用 → 执行工具 → 继续循环 |
| `limits.py` | `AgentLoopLimits`：最大工具轮次、最大 token 数等 |

### 6. `own_agent/permissions/` — 权限系统
| 文件 | 职责 |
|---|---|
| `types.py` | `PermissionAction`, `PermissionRequest`, `PermissionDecision` |
| `manager.py` | `PermissionManager`：决定是否允许工具执行 |

支持三种模式：
- `standard`：每个敏感操作询问用户
- `aggressive`：自动允许已知安全操作
- `bypass`：全部自动允许（benchmark 用）

### 7. `own_agent/session/` — 会话管理
| 文件 | 职责 |
|---|---|
| `session.py` | `AgentSession`：当前会话状态 |
| `store.py` | 会话持久化到 JSONL 文件 |

### 8. `own_agent/context/` — 上下文窗口管理
| 文件 | 职责 |
|---|---|
| `manager.py` | `ContextManager`：跟踪 token 使用量，触发上下文压缩 |

### 9. `own_agent/tui/` — TUI 交互界面
使用 **prompt_toolkit + Rich** 构建。
- `app.py`：主 UI 循环
- 输入区：`prompt_toolkit` 的 `PromptSession`
- 输出区：Rich 渲染的消息列表
- 支持流式显示、权限确认、历史记录

## 数据流（一次完整交互）

```
1. 用户输入消息
2. TUI/CLI 调用 AgentLoop.run_user_turn(text)
3. AgentLoop 构建 ChatRequest（消息历史 + 工具 schema）
4. 调用 ChatProvider.complete(request)
5. LLM 返回 ChatResponse（文本 + tool_calls + finish_reason）
6. AgentLoop 检查 finish_reason：
   a. "stop" → 返回响应给用户
   b. "tool_calls" → 逐个调用工具 → 将结果追加到消息历史 → 回到步骤 3
   c. "length" / "error" → 返回错误信息
7. 最终响应返回给 TUI/CLI 展示
```
