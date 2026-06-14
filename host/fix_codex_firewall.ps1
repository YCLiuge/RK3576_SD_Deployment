$ErrorActionPreference = "Stop"

$remote = "10.138.103.190"
$ports = "22,5555"

$programs = @(
    @{
        Name = "RK3576 SSH-ADB outbound - conda comprehensive python"
        Path = "D:\Anaconda3\envs\comprehensive\python.exe"
    },
    @{
        Name = "RK3576 SSH-ADB outbound - PowerShell"
        Path = "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
    },
    @{
        Name = "RK3576 SSH-ADB outbound - OpenSSH"
        Path = "C:\Program Files\OpenSSH\ssh.exe"
    },
    @{
        Name = "RK3576 SSH-ADB outbound - Codex"
        Path = "C:\Program Files\WindowsApps\OpenAI.Codex_26.609.4994.0_x64__2p2nqsd0c76g0\app\resources\codex.exe"
    }
)

$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()
).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    throw "Please run this script from an elevated Administrator PowerShell."
}

foreach ($program in $programs) {
    $name = $program.Name
    $path = $program.Path

    if (-not (Test-Path -LiteralPath $path)) {
        Write-Warning "Missing program path: $path"
        continue
    }

    $existing = Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue
    if ($existing) {
        Set-NetFirewallRule -DisplayName $name -Enabled True -Action Allow -Profile Any
        Write-Host "Updated: $name"
    } else {
        New-NetFirewallRule `
            -DisplayName $name `
            -Direction Outbound `
            -Action Allow `
            -Enabled True `
            -Profile Any `
            -Program $path `
            -Protocol TCP `
            -RemoteAddress $remote `
            -RemotePort $ports | Out-Null
        Write-Host "Created: $name"
    }
}

Write-Host ""
Write-Host "Firewall rules now matching RK3576:"
Get-NetFirewallRule -DisplayName "RK3576 SSH-ADB outbound*" |
    Select-Object DisplayName, Enabled, Direction, Action, Profile |
    Format-Table -AutoSize

Write-Host ""
Write-Host "Testing board ports:"
Test-NetConnection $remote -Port 22
Test-NetConnection $remote -Port 5555
