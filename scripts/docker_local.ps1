param(
    [ValidateSet('run','setup','build','start','stop','restart','status','logs')]
    [string]$Action = 'run'
)

$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskProject = if ($env:PRAMANA_DOCKER_PROJECT) { $env:PRAMANA_DOCKER_PROJECT } else { 'pramana-local' }
if ($taskProject -notmatch '^[a-z0-9][a-z0-9_-]*$') {
    Write-Host 'PRAMANA_DOCKER_PROJECT must be a lowercase Docker project name.'
    exit 2
}
$taskCompose = @('compose', '--project-name', $taskProject, '--file', (Join-Path $taskRoot 'compose.local.yaml'))

function New-LocalConfiguration {
    $taskConfig = Join-Path $taskRoot '.env'
    if (-not (Test-Path -LiteralPath $taskConfig)) {
        # Create-only copying: an existing config cannot be overwritten.
        [IO.File]::Copy((Join-Path $taskRoot '.env.example'), $taskConfig, $false)
        Write-Host 'Created .env. Add GOOGLE_API_KEY (optional GROQ_API_KEY), save, then run docker-run.cmd.'
    }
}

function Wait-LocalHealth {
    $taskId = (& docker @taskCompose ps --all --quiet pramana 2>$null | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $taskId) {
        throw 'No existing container. Use docker-run.cmd for the first build/run.'
    }
    $taskDeadline = [DateTime]::UtcNow.AddSeconds(120)
    do {
        $taskState = (& docker inspect --format '{{.State.Status}}|{{if .State.Health}}{{.State.Health.Status}}{{end}}' $taskId 2>$null | Out-String).Trim()
        if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect the application container.' }
        if ($taskState -eq 'running|healthy') {
            $taskAddress = (& docker @taskCompose port pramana 8000 2>$null | Out-String).Trim()
            Write-Host "PRAMANA is healthy: http://$taskAddress/"
            Write-Host "Container project: $taskProject. Stop: docker-run.cmd stop. Next start: docker-run.cmd start."
            Write-Host 'No host Python or local model is required. Live APIs need internet/quota.'
            return
        }
        if ($taskState -match '^(exited|dead)' -or $taskState -match 'unhealthy$') {
            throw 'Application startup/health failed. Use docker-run.cmd logs; check API access/quota and configuration.'
        }
        Start-Sleep -Milliseconds 500
    } while ([DateTime]::UtcNow -lt $taskDeadline)
    throw 'Health timeout. Use docker-run.cmd status and docker-run.cmd logs.'
}

Push-Location -LiteralPath $taskRoot
try {
    if ($Action -in @('setup','run','build')) { New-LocalConfiguration }
    if ($Action -eq 'setup') {
        Write-Host 'Edit .env and add your Google API key. Next: docker-run.cmd. Docker Desktop must be running.'
        exit 0
    }
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw 'Docker is not installed/on PATH. Install and start Docker Desktop with Linux containers.'
    }
    $taskEngine = (& docker info --format '{{.OSType}}' 2>$null | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $taskEngine -ne 'linux') {
        throw 'Start Docker Desktop with its Linux engine, then retry. No host Python is needed.'
    }
    & docker compose version | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Docker Compose v2 is required.' }
    if ($Action -in @('run','build')) {
        # Capture resolved configuration in memory only: it contains API keys.
        # Never print this JSON or invoke an unfiltered compose config in logs.
        $taskJson = (& docker @taskCompose config --format json 2>$null | Out-String)
        if ($LASTEXITCODE -ne 0) { throw 'Compose configuration failed. Check .env, port, and Docker Compose version.' }
        $taskConfig = $taskJson | ConvertFrom-Json
        $taskEnvironment = $taskConfig.services.pramana.environment
        if ([string]::IsNullOrWhiteSpace($taskEnvironment.GOOGLE_API_KEY) -and
            [string]::IsNullOrWhiteSpace($taskEnvironment.GROQ_API_KEY)) {
            Write-Host 'Missing API key. Edit .env, set GOOGLE_API_KEY, save, then rerun docker-run.cmd.'
            exit 2
        }
        $taskArgs = @('up','--detach','--wait','--wait-timeout','120')
        if ($Action -eq 'build') { $taskArgs += '--build' }
        & docker @taskCompose @taskArgs
        if ($LASTEXITCODE -ne 0) { throw 'Build/start failed. Check port, resources, internet and API access; run docker-run.cmd logs.' }
        Wait-LocalHealth
    } elseif ($Action -in @('start','restart')) {
        $taskExisting = (& docker @taskCompose ps --all --quiet pramana 2>$null | Out-String).Trim()
        if (-not $taskExisting) { throw 'No existing container. First use docker-run.cmd.' }
        & docker @taskCompose $Action pramana
        if ($LASTEXITCODE -ne 0) { throw 'Container start/restart failed.' }
        Wait-LocalHealth
    } elseif ($Action -eq 'stop') {
        & docker @taskCompose stop pramana
        if ($LASTEXITCODE -ne 0) { throw 'Container stop failed.' }
        Write-Host 'Stopped without deleting the container or state volume. Next: docker-run.cmd start.'
    } elseif ($Action -eq 'status') {
        & docker @taskCompose ps --all
        if ($LASTEXITCODE -ne 0) { throw 'Status check failed.' }
    } else {
        & docker @taskCompose logs --tail 100 pramana
        if ($LASTEXITCODE -ne 0) { throw 'Log read failed.' }
    }
} catch {
    Write-Host ('Docker launcher: ' + $_.Exception.Message)
    exit 1
} finally {
    Pop-Location
}
