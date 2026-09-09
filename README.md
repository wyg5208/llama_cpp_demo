# llama.cpp 本地智能体工作台

一个跑在自己机器上的聊天式 AI 工作台：FastAPI 后端持有 `llama-server` 子进程，浏览器里的原生前端通过 SSE 逐字收答案，模型可以调用检索、文件、长期记忆等工具。所有推理都在本地，除检索后端外不依赖任何云端服务。

**版本 / Version**: v1.1.0 | **更新日期 / Date**: 2026-09-09
**运行环境**: Windows + Python 3.11（仓库内 `.venv`）+ llama.cpp 预编译包（默认 Vulkan 后端）

> 本文件是项目的总览与**版本号唯一来源**。配置项的逐项说明在 `.env.example`，历史变更在 `docs/开发记录/`。

---

## 1. 能力一览

| 能力 | 说明 | 落点 |
|---|---|---|
| 本地对话 | OpenAI 兼容协议流式对话，支持思维链（`enable_thinking` + `think` 工具）与工具调用循环 | `app/llm.py`、`app/main.py` |
| 模型管理 | 平铺扫描模型目录下 `*.gguf`（不递归），按名配对 `mmproj-*` 视觉投影文件，热切换并记住所选 | `app/models.py`、`app/runtime.py` |
| 联网检索 | `auto` 择优：bocha → tavily → bing（RSS，免密钥），三者均可在境内直连 | `app/search.py` |
| 文件读写 | 通过 stdio JSON-RPC 挂载 `@modelcontextprotocol/server-filesystem`，工具级只读/可写分层，允许根目录可配 | `app/mcp.py`、`app/tools.py` |
| 长期记忆 | `remember` / `recall` 读写单文件笔记库 | `app/memory_store.py` |
| 文档入库 | 上传 md / txt / pdf / docx / xlsx / pptx → 纯文本进上下文 | `app/documents.py` |
| 会话归档 | 一个索引 + 每会话一个 JSON 文件（两级，避免打开列表时读入图片 base64） | `app/history.py` |
| 答案导出 | 已渲染 HTML → md / html / csv / pdf / docx（PDF 与 OOXML 均为手写，不引第三方渲染器） | `app/export.py` |
| 机器遥测 | CPU / 内存 / GPU / 显存 / GPU 温度，ctypes 直调 Win32 与 NVML | `app/sysstats.py` |
| 生成文件 | 模型可主动调用 `save_document` 产出文件（Qwen3 的代码块围栏会被解开） | `app/tools.py` |

内置工具均按请求开关装配（`build_tools` 的 `search` / `memory` / `think` / `doc_gen` / `fs_read` / `fs_write` 六个开关）：`web_search`、`remember`、`recall`、`think`、`save_document`；MCP 文件系统工具在检测到 Node.js 与已安装的 server 后追加（实测该包提供 14 个工具，并按只读 / 可写分层）。

## 2. 快速开始

```powershell
# 1) 依赖（requirements.txt 必须保持纯 ASCII，见第 7 节）
py -3.11 -m venv .venv          # 或本机任意一个 Python 3.11：python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2) llama.cpp 运行库：可续传下载 + sha256 校验 + 解压到 runtime\llama-<backend>\
.\.venv\Scripts\python.exe scripts\fetch_runtime.py vulkan        # 或 cuda / cpu

# 3) 可选：MCP 文件系统服务端（需本机 Node.js），装到 runtime\mcp\node_modules\
.\.venv\Scripts\python.exe scripts\fetch_mcp.py

# 4) 配置：复制模板后按注释改，至少要有 MODEL_PATH / MMPROJ_PATH
Copy-Item .env.example .env

# 5) 启动
.\start_app.bat                                                   # 或直接 .\.venv\Scripts\python.exe run.py
```

浏览器打开 **http://127.0.0.1:8123**。`start_app.bat` 只做一件事——转调 `scripts\launcher.ps1`：检查虚拟环境、结束上一实例、等端口释放、起服务并开浏览器。

## 3. 目录结构

```
llama_cpp_demo/
├── run.py                     入口：先 setup_logging 再起 uvicorn（app.main:app）
├── start_app.bat              启动外壳，只能纯 ASCII，逻辑全在 launcher.ps1
├── requirements.txt           运行时依赖（禁止非 ASCII 注释）
├── .env.example               全部配置项的中文说明；.env 的模板，是唯一权威文档
├── app/                       后端，15 个模块
│   ├── config.py              Settings + 字段校验器 + DEFAULT_SYSTEM_PROMPT + 日志装配
│   ├── settings_store.py      运行时可改项的唯一权威（请求体即落盘白名单）
│   ├── main.py                全部 /api/* 路由、SSE 对话、静态挂载
│   ├── runtime.py             持有 llama-server 子进程：拉起 / 等就绪 / 关闭
│   ├── llm.py                 流式客户端与工具调用循环（累积 tool-call 增量）
│   ├── models.py              本地 GGUF 目录清单与 mmproj 配对
│   ├── tools.py               工具装配与策略、非检索类工具执行、MCP schema 转换
│   ├── mcp.py                 纯标准库的 stdio JSON-RPC MCP 客户端
│   ├── search.py              bing / bocha / tavily 三套检索后端
│   ├── documents.py           入站：六种格式 → 文本
│   ├── export.py              出站：HTML → md/html/csv/pdf/docx
│   ├── history.py             会话归档（index + 单文件）
│   ├── memory_store.py        长期记忆 runtime/memory.json
│   ├── sysstats.py            顶栏遥测（Win32 + NVML）
│   └── version.py             读 README 版本行得到应用版本号（按 mtime 缓存，供关于面板用）
├── static/                    原生前端，无构建步骤：index.html + app.js + style.css
├── scripts/                   launcher.ps1 / fetch_runtime.py / fetch_mcp.py
├── tests/                     9 个 unittest 文件，466 个用例
├── docs/开发记录/             迭代记录（v{版本}_{日期}_{描述}.md）与 index.md 索引
└── runtime/                   全部运行时产物，被 .gitignore 忽略
    ├── llama-vulkan/          llama.cpp 预编译包（数百 MB）
    ├── mcp/node_modules/      MCP 服务端依赖
    ├── history/  memory.json  会话与记忆数据
    ├── active_model.json      记住的模型选择
    ├── settings_override.json 界面改写的设置
    └── app.log  llama-server.log
```

## 4. 配置

配置项一律以 `app/config.py` 的 `Settings` 为准，逐项释义与容量测算写在 `.env.example`（`.env` 含密钥，不入库，也不要在文档里引用其内容）。

生效优先级：`runtime/settings_override.json` > 环境变量 / `.env` > `Settings` 默认值。

其中 8 项属于运行时可改（下一句话即生效，无需重启）：`system_prompt`、`temperature`、`top_p`、`max_tokens`、`thinking_max_tokens`、`max_tool_rounds`、`search_provider`、`search_max_results`；其余项需重启。想确认当前总数：

```powershell
.\.venv\Scripts\python.exe -c "from app.config import Settings as S; from app.settings_store import EDITABLE; print(len(S.model_fields), len(EDITABLE))"
```

端口约定：应用 `8123`（`8000` 常被本机 ComfyUI 占用，不要改回去），`llama-server` 子进程 `8081`。想复用外部已启动的 server，设 `RUNTIME_BACKEND=external` 并给 `LLAMA_SERVER_URL`。

## 5. HTTP 接口

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/status` | runtime 子进程状态 |
| GET | `/api/stats` | 机器遥测 |
| GET / POST | `/api/models`、`/api/model` | 列出模型 / 切换模型 |
| GET / PATCH / DELETE | `/api/settings` | 读设置（含掩码密钥）/ 改 8 项 / 清空覆写 |
| GET | `/api/about` | 应用版本、runtime、build_info、模型体积、会话统计 |
| GET / POST | `/api/sessions` | 会话索引 / 新建 |
| GET / PATCH / DELETE | `/api/sessions/{id}` | 单个会话 |
| GET / DELETE | `/api/memory`、`/api/memory/{id}` | 记忆列表 / 删除 |
| POST | `/api/documents` | 上传文档并解析为文本 |
| POST | `/api/export` | 导出会话文件 |
| POST | `/api/chat` | 对话，SSE 流式返回 |
| GET | `/` | 前端首页 |

## 6. 开发与验证

```powershell
$env:PYTHONIOENCODING="utf-8"; .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

当前 466 个用例（含 2 个 skip），耗时 20~35 秒（视磁盘缓存）。跑单个文件：`.\.venv\Scripts\python.exe -m unittest tests.test_export -v`。**本项目不引入 pytest**（标准库足够，且已装依赖越少越好）。版本号四处落点的一致性由 `tests/test_version.py` 负责，改动 `README.md` 顶部版本行后至少跑一次它。

## 7. 仓库约定（红线）

1. `requirements.txt`、`start_app.bat` 必须**纯 ASCII**：pip 按本地 cp936 解码前者，cmd.exe 按字节偏移重读后者，非 ASCII 会让安装失败或把注释当命令执行。
2. `scripts/launcher.ps1` 必须 **UTF-8 带 BOM**：PowerShell 5.1 靠 BOM 判编码，无 BOM 时按 ANSI(cp936) 解码，中文提示会乱码。
3. `runtime/` 与 `.env`、`.venv/`、`__pycache__/` 永不入库；临时脚本、调试产物也一律写在 `runtime/` 下。
4. 易引入大体积传递依赖的包要**精确锁版本**（如 `pymupdf4llm==1.28.2`，避免拖入 41 MB ONNX 布局模型与 numpy/onnxruntime）。
5. 新增 Settings 字段的三处联动：`app/config.py` 字段+校验器 → `.env.example` 说明 → 若需运行时可改再加入 `SettingsPatch`；界面上要改则再动 `static/index.html` 与 `static/app.js`。
6. 文档、注释与提交信息里不写死会漂移的数字（配置项总数等），需要时给实测命令。

## 8. 迭代与版本管理

- 版本号遵循语义化 `MAJOR.MINOR.PATCH`：**修订号 +1** = bug 修复、配置/文案、UI 微调、依赖更新；**次版本号 +1**（修订归零）= 新增功能、新增 `/api` 端点或工具、新增配置项；**主版本号 +1**（其余归零）= 架构重构、runtime 后端策略变更、依赖策略变更等不兼容改动。
- 每次迭代由 `iteration-doc` 智能体生成 `docs/开发记录/v{版本号}_{日期}_{功能描述}.md`，并在 `docs/开发记录/index.md` 顶部追加一行。
- 版本号同步的落点（四处）：本文件顶部 `**版本 / Version**` 行、本文件第 9 节版本历史、迭代记录文档（文件名与「基本信息」）、`docs/开发记录/index.md`。本仓库尚无远程，发布提交上的本地 `git tag vX.Y.Z` 由 `version-release` 负责。
- 顶部 `**版本 / Version**: vX.Y.Z` 这一行的**格式是契约**，不是排版：它是应用版本号的唯一读取源（`app/version.py:read_version()` → `/api/about` 的 `version` 字段 → 侧边栏关于面板首行「应用版本」），四处落点之间的一致性由 `tests/test_version.py` 校验；改动该行的写法会让关于面板显示「未知」并让测试失败。
- 提交由 `version-release` 智能体完成：单行英文祈使句主题，不带 `feat:`/`release:` 前缀。

## 9. 版本历史

### v1.1.0 — 2026-09-09

应用自己的版本号进入运行时：

- ✅ 新增 `app/version.py`：`read_version()` 从本文件顶部版本行正则取版本号，按 `(路径, mtime_ns, size)` 缓存，改 README 免重启即生效
- ✅ 容错不抛异常：文件缺失 / 非 UTF-8 / 版本行形制不匹配一律返回 `None` 并 `log.warning` 一次，面板显示「未知」而非 500
- ✅ `/api/about` 新增 `version` 字段（约定同 SystemStats：取不到时为 `null` 而非缺字段）
- ✅ 侧边栏关于面板首行新增「应用版本」；`v` 前缀只在前端拼，接口回传裸数字便于与 `git tag` 对账
- ✅ 新增 `tests/test_version.py`（11 用例）：README 成为唯一源的读取/缓存行为 + 四处落点一致性 + `/api/about` 契约
- ✅ 刻意不把版本号做成 `Settings` 字段：它不可由用户配置，不该进 `.env` / `.env.example` / `SettingsPatch` 白名单
- ✅ 回归基线：`unittest` 466 用例

### v1.0.0 — 2026-09-09

首个正式版本，功能面收敛完整：

- ✅ 本地对话：SSE 流式、工具调用循环、思维链与 `think` 工具
- ✅ 模型管理：GGUF 平铺扫描 + mmproj 自动配对 + 运行时热切换，Vulkan / CUDA / CPU / external 四种后端
- ✅ 检索：bocha / tavily / bing 三后端与 `auto` 择优
- ✅ MCP 文件读写（纯标准库客户端，Node.js 缺失时优雅降级）
- ✅ 长期记忆、文档上传（6 格式）、会话归档、导出（5 格式）
- ✅ 顶栏机器遥测、设置/关于/帮助面板、滚动日志 `runtime/app.log`
- ✅ 回归基线：`unittest` 455 用例

## 10. 排障

| 现象 | 先看 |
|---|---|
| 页面空白或前端行为没更新 | `Ctrl+F5` 强刷；`/static` 虽带 `Cache-Control: no-cache`，仍可能被会话级缓存欺骗 |
| 启动卡在"等待就绪" | `runtime/llama-server.log`；27B 级别模型加载可达数十秒，`SERVER_STARTUP_TIMEOUT` 默认 300 秒 |
| 回复为空 | 多半是思维链吃满 `max_tokens`：调大 `MAX_TOKENS` 或关掉 `ENABLE_THINKING` |
| 上下文被截断 / OOM | 用 `N_CTX_OVERRIDES` 按模型分别设窗口，显存预算见 `.env.example` 中的实测表 |
| 文件工具灰掉 | 未装 Node.js 或未执行 `scripts/fetch_mcp.py`；`GET /api/status` 里能看到 MCP 明细 |
| 改了 `.env` 不生效 | 检查 `runtime/settings_override.json` 是否覆写了同名字段（DELETE `/api/settings` 可一键清空） |

日志：`runtime/app.log`（1 MB × 3 轮转，每次对话一行）；`GET /api/status`、`GET /api/about` 可确认子进程、build_info 与模型体积。
