# 本机部署说明

已在 Windows 11、Python 3.12、RTX 4060 Laptop 8GB 上完成接口验证。
原始源码仍保存在上一级 PortableLLM-master.zip。

## 日常使用

1. 双击 start_server.bat，等待出现 ready。
2. 浏览器打开 http://127.0.0.1:8000/docs。
3. 展开 POST /generate，点击 Try it out，输入下面的 JSON，再点击 Execute。
4. 用完双击 stop_server.bat。关闭浏览器不会停止后台服务。

```json
{
  "prompt": "请用中文介绍什么是本地大模型。",
  "model": 1,
  "max_tokens": 512
}
```

也可以在项目目录运行：

```powershell
.\.venv\Scripts\python.exe server_control.py start
.\.venv\Scripts\python.exe server_control.py stop
```

当前没有专用聊天界面；/docs 是可直接发请求的接口调试页面。
启动脚本不会自动开机运行，仅启动本项目的后台服务。
停止脚本会核对进程路径、命令行、创建时间，不会按 python.exe 名称批量结束进程。

## 当前模型与配置

- 文本模型：Qwen3.5-4B-Q4_K_M.gguf，下载大小 2,740,937,888 字节。
- 来源：https://huggingface.co/unsloth/Qwen3.5-4B-GGUF
- 固定版本：e87f176479d0855a907a41277aca2f8ee7a09523。
- SHA256：00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4，已核对远端 LFS 哈希。
- 推理：llama-cpp-python 0.3.36，CUDA 12.4，本项目虚拟环境内的 NVIDIA 运行库。
- 上下文：4096；默认最大生成：1024；GPU 层数：-1（尽量全部加载到 GPU）。
- 相似度：all-MiniLM-L6-v2，在 CPU 上按需加载，首次调用较慢。
- 主模型和嵌入模型已下载，run.py 设置 Hugging Face 离线模式。
- 模型文件位于 portablellm/models 和 portablellm/embedding_models。
- 修改 config.json 后先停止再启动服务。
- 只支持本次请求的文本输入，不自动保留聊天历史；本次没有配置图片输入。

## 修复内容

补全并锁定依赖；配置项目内 CUDA DLL 搜索路径；支持 GPU 加载；修正嵌入模型目录并按需加载；
单个生成模型驻留、生成请求加锁；RDF/SPARQL 正确传递模型编号；启动读取 host/port；
限制提示词文件路径；无效请求返回明确错误；修复停止脚本的进程范围。

API 保留 /generate、/rdf、/sparql、/similarity、/dictionary_similarity。
max_tokens 可在生成请求中指定。输出截断返回 400，避免把不完整结果当作完整回答。
RDF/SPARQL 仍然是模型文本生成，本次未加入语法或业务正确性校验。

## 验证与日志

```powershell
.\.venv\Scripts\python.exe verify_local.py
```

结果保存在 logs/verification.json。覆盖中文生成、简单算术、RDF/SPARQL 返回、
相似度、词典匹配、无效模型、模板路径越界、空输入、超长上下文。
这些是部署冒烟检查，不是模型能力评测。
服务日志：logs/server.out.log、logs/server.err.log，每次启动覆盖。
首次部署的 GPU 与速度日志另外保存在 logs/initial-validation.err.log。

## 重建环境（已有环境无需重复执行）

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_models.py
.\.venv\Scripts\python.exe server_control.py start
```

requirements-lock.txt 固定这次测试通过的版本与 CUDA wheel 来源。
使用模型原生聊天模板；没有安装全局 CUDA Toolkit，也没有修改系统显卡驱动。
原项目 build.bat 的 EXE 打包流程不属于本次部署，仍缺少 PortableLLM.spec，请使用上述虚拟环境启动方式。

