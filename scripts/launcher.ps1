<#
    由 start_app.bat 调用：校验环境、结束上一次遗留的实例、等端口真正释放，
    再把浏览器交给一个后台小进程在端口就绪后打开。

    退出码 0 = 可以继续启动应用；非 0 = 应当中止（.bat 会 pause 并停下）。

    本文件必须是 UTF-8 with BOM：PowerShell 5.1 靠 BOM 判定编码，
    没有 BOM 时会按系统 ANSI(cp936) 读，中文虽在本机能碰巧正确，换机器就乱。
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Root
)

$ErrorActionPreference = 'Stop'

try { $Host.UI.RawUI.WindowTitle = 'llama.cpp 本地聊天' } catch { }

function Get-ListeningPids {
    param([int[]]$Ports)
    $found = @()
    foreach ($p in $Ports) {
        $c = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue
        if ($c) { $found += $c.OwningProcess }
    }
    return $found
}

# ------------------------------------------------------------------
# 1. 环境校验
# ------------------------------------------------------------------
$py = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $py)) {
    Write-Host '[错误] 找不到虚拟环境里的 Python：'
    Write-Host ('       ' + $py)
    Write-Host '       请先执行：python -m venv .venv'
    Write-Host '                 .venv\Scripts\pip install -r requirements.txt'
    exit 1
}

# ------------------------------------------------------------------
# 2. 读 .env 里的 HOST / PORT / LLAMA_PORT，读不到就用默认值
#    只取这三个键，其它内容（含 API 密钥）不读进来也不打印。
# ------------------------------------------------------------------
$hostAddr = '127.0.0.1'
$appPort = 8123
$llamaPort = 8081

$envFile = Join-Path $Root '.env'
if (Test-Path -LiteralPath $envFile) {
    foreach ($line in (Get-Content -LiteralPath $envFile -Encoding UTF8)) {
        $t = $line.Trim()
        if (-not $t -or $t.StartsWith('#')) { continue }
        $eq = $t.IndexOf('=')
        if ($eq -lt 1) { continue }
        $key = $t.Substring(0, $eq).Trim().ToUpperInvariant()
        $val = $t.Substring($eq + 1).Trim()
        if (-not $val) { continue }
        $num = 0
        switch ($key) {
            'HOST' { $hostAddr = $val }
            'PORT' { if ([int]::TryParse($val, [ref]$num) -and $num -gt 0 -and $num -lt 65536) { $appPort = $num } }
            'LLAMA_PORT' { if ([int]::TryParse($val, [ref]$num) -and $num -gt 0 -and $num -lt 65536) { $llamaPort = $num } }
        }
    }
}
# 0.0.0.0 / :: 是监听地址，不是可访问地址，打开浏览器要换成回环。
if ($hostAddr -eq '0.0.0.0' -or $hostAddr -eq '::' -or $hostAddr -eq '') { $hostAddr = '127.0.0.1' }

$ports = @($appPort, $llamaPort)
$url = 'http://' + $hostAddr + ':' + $appPort + '/'

Write-Host ''
Write-Host '[1/2] 检测并结束旧进程...'

# ------------------------------------------------------------------
# 3. 找出属于本项目的进程
#
#    两条判据，命中任一条即视为本项目：
#      a) 命令行里含项目根路径，且（是 llama-server，或含 run.py/uvicorn/app.main）
#      b) 正在监听本项目的端口
#
#    为什么要 (b)：Windows 上 .venv\Scripts\python.exe 只是个转发器，它拉起
#    真正的基础解释器来跑 run.py，那个子进程的命令行形如
#    "d:\python\python311\python.exe" run.py —— 不含项目路径，只能靠端口认出来。
#
#    为什么要 (a) 里的 run.py 限定：本机上 ComfyUI 也在跑 python.exe，
#    Ollama 也带一个 llama-server.exe，只按进程名匹配会误杀。
#
#    路径比较一律转小写：.NET 的 String.Contains 区分大小写，而命令行里的盘符
#    是 d: 还是 D: 取决于当初怎么启动的，实测两种都出现过。
# ------------------------------------------------------------------
$names = @('python.exe', 'pythonw.exe', 'llama-server.exe')
$rootLower = $Root.ToLowerInvariant()
$listen = Get-ListeningPids -Ports $ports
$targets = @()

foreach ($p in (Get-CimInstance Win32_Process)) {
    if ($names -notcontains $p.Name) { continue }
    $cl = if ($p.CommandLine) { $p.CommandLine } else { '' }
    $lcl = $cl.ToLowerInvariant()
    $byCmd = $lcl.Contains($rootLower) -and (
        $p.Name -eq 'llama-server.exe' -or
        $lcl.Contains('run.py') -or
        $lcl.Contains('uvicorn') -or
        $lcl.Contains('app.main'))
    $byPort = $listen -contains $p.ProcessId
    if ($byCmd -or $byPort) { $targets += $p }
}

if ($targets.Count -eq 0) {
    Write-Host '      没有旧进程在跑'
}
else {
    foreach ($p in $targets) {
        # /T 会把整条链（venv 转发器 -> 真解释器 -> llama-server）一起带走，
        # 而链上的进程往往都被上面的判据匹配进了 $targets，轮到子进程时它已不在。
        if (-not (Get-Process -Id $p.ProcessId -ErrorAction SilentlyContinue)) {
            Write-Host ('      PID ' + $p.ProcessId + ' 已随父进程一并结束')
            continue
        }
        $cl = if ($p.CommandLine) { $p.CommandLine } else { '(无命令行)' }
        if ($cl.Length -gt 88) { $cl = $cl.Substring(0, 88) + '...' }
        Write-Host ('      结束 PID ' + $p.ProcessId + '  ' + $p.Name)
        Write-Host ('        ' + $cl)
        # 经 cmd 转发：taskkill 的 stderr 就地并进 stdout，PowerShell 不会把它
        # 包成 ErrorRecord（那既会在 $ErrorActionPreference='Stop' 下终止脚本，
        # 也会让消息尾巴多出一串 System.Management.Automation.RemoteException）。
        $out = (cmd /c "taskkill /F /T /PID $($p.ProcessId) 2>&1") -join ' '
        # 判成功看结果不看返回码：/T 结束父进程时子进程会跟着消失，轮到我们时它
        # 常常已经不在了，taskkill 对此返回 128/255 并抱怨「没有找到进程」。
        # 反过来，正在退出的进程也会在进程表里再停留一会儿（实测 llama-server
        # 被杀后 200ms 仍在），所以要等它真的消失，而不是立刻下结论。
        $gone = -not (Get-Process -Id $p.ProcessId -ErrorAction SilentlyContinue)
        for ($w = 0; -not $gone -and $w -lt 15; $w++) {
            Start-Sleep -Milliseconds 200
            $gone = -not (Get-Process -Id $p.ProcessId -ErrorAction SilentlyContinue)
        }
        if (-not $gone) {
            Write-Host ('        [失败] 等了 3 秒进程仍在运行：' + $out)
        }
    }
}

# ------------------------------------------------------------------
# 4. 等端口真正释放
#
#    不能杀完就启动：llama-server 被 taskkill /F 后要几秒才交出 8081，
#    这期间启动的话，新应用的健康检查会被残留的旧服务应答而误报就绪，
#    而真正的新服务还在加载模型（实测 25 秒），期间发消息全部失败；
#    8123 没释放则直接 [Errno 10048] 绑定失败。
# ------------------------------------------------------------------
$deadline = (Get-Date).AddSeconds(30)
$busy = $ports
while ((Get-Date) -lt $deadline) {
    $busy = @()
    foreach ($port in $ports) {
        if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) { $busy += $port }
    }
    if ($busy.Count -eq 0) { break }
    Start-Sleep -Milliseconds 400
}

if ($busy.Count -gt 0) {
    foreach ($port in $busy) {
        $c = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
        $ownerName = '?'
        if ($c) {
            $owner = Get-Process -Id $c.OwningProcess -ErrorAction SilentlyContinue
            if ($owner) { $ownerName = $owner.ProcessName }
        }
        Write-Host ('      [警告] 端口 ' + $port + ' 等了 30 秒仍被 ' + $ownerName + ' 占用，未强制结束')
        Write-Host '             新实例可能绑定失败。请确认它不是本项目残留后手动处理。'
    }
}
elseif ($targets.Count -gt 0) {
    Write-Host '      端口已释放'
}

# ------------------------------------------------------------------
# 5. 浏览器：交给一个隐藏的后台进程，等端口开始监听再打开
#
#    前端首屏请求失败时不会自动重试（pollUntilReady 只在 loading/switching
#    状态下重排），提前打开会一直卡在「无法连接后端服务」。
#
#    用 Start-Process 的参数数组而不是 cmd 的 start：后者会重解析引号，
#    实测能把 Get-NetTCPConnection 劈成两半当成命令执行，还会让 cmd 丢失
#    批处理文件偏移、把 run.py 重跑一遍。
# ------------------------------------------------------------------
$opener = "for(`$i=0;`$i -lt 180;`$i++){ if(Get-NetTCPConnection -LocalPort $appPort -State Listen -ErrorAction SilentlyContinue){ Start-Process '$url'; break }; Start-Sleep -Seconds 1 }"
Start-Process -FilePath 'powershell.exe' `
    -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', $opener) `
    -WindowStyle Hidden

Write-Host ''
Write-Host '[2/2] 启动应用...'
Write-Host ('      地址  ' + $url)
Write-Host '      浏览器会在端口就绪后自动打开；模型加载约需 7~30 秒，'
Write-Host '      显存被 ComfyUI 等占用时更慢，请等窗口出现「模型就绪」再开始提问。'
Write-Host '      停止：在窗口里按 Ctrl+C，会连带关闭 llama-server 子进程。'
Write-Host '      若应用异常退出，窗口会保留方便看日志；若那是因为你又运行了一次'
Write-Host '      本脚本，直接关掉旧窗口即可。'
Write-Host ''

exit 0
