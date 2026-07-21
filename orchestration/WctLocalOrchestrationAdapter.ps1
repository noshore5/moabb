[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateSet(
        "InitializeProgram",
        "ValidateIntegration",
        "PrepareWorktree",
        "ConfigureExecution",
        "PreflightExecution",
        "BeforeCleanup"
    )]
    [string]$Operation,
    [Parameter(Mandatory)][hashtable]$Context,
    [string]$LogPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Test-WctPythonModuleName {
    param([string]$Name)
    return $Name -match '^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$'
}

function Invoke-WctPythonProbe {
    param(
        [string]$Python,
        [ValidateSet("dependencies", "sources", "execution")]
        [string]$Mode,
        [string]$Phase,
        [string]$WorkingDirectory
    )
    Write-LocalOrchestrationLog `
        -Phase $Phase `
        -Level "PLAN" `
        -Source "Project:wct-gnn" `
        -Message "Run read-only Python preflight" `
        -LogPath $LogPath
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $locationChanged = $false
    try {
        $preflightScript = Get-LocalOrchestrationPath `
            -Path (Join-Path $PSScriptRoot "WctLocalOrchestrationPreflight.py") `
            -MustExist
        if ($WorkingDirectory) {
            $probeRoot = Get-LocalOrchestrationPath `
                -Path $WorkingDirectory `
                -MustExist
            Push-Location $probeRoot
            $locationChanged = $true
        }
        $probeOutput = & $Python $preflightScript $Mode 2>&1
        $probeExitCode = $LASTEXITCODE
    } finally {
        if ($locationChanged) {
            Pop-Location
        }
        $ErrorActionPreference = $previousPreference
    }
    if ($probeExitCode -ne 0) {
        $detail = ($probeOutput | Out-String).Trim()
        throw "WCT Python preflight failed during $Phase. No dependency or data repair was attempted.`n$detail"
    }
    foreach ($line in @($probeOutput)) {
        $detail = ([string]$line).TrimEnd()
        if ($detail) {
            Write-LocalOrchestrationLog `
                -Phase $Phase `
                -Level "TRACE" `
                -Source "Project:wct-gnn" `
                -Message "Preflight: $detail" `
                -LogPath $LogPath
        }
    }
    Write-LocalOrchestrationLog `
        -Phase $Phase `
        -Level "OK" `
        -Source "Project:wct-gnn" `
        -Message "Python preflight passed" `
        -LogPath $LogPath
}

function Set-WctTemporaryEnvironment {
    param([hashtable]$Values)
    $saved = @{}
    foreach ($key in $Values.Keys) {
        $saved[$key] = [Environment]::GetEnvironmentVariable($key, "Process")
        [Environment]::SetEnvironmentVariable($key, $Values[$key], "Process")
    }
    return $saved
}

function Restore-WctTemporaryEnvironment {
    param([hashtable]$Saved)
    foreach ($key in $Saved.Keys) {
        [Environment]::SetEnvironmentVariable($key, $Saved[$key], "Process")
    }
}

function Invoke-WctInitializeProgram {
    param([hashtable]$Context)

    $environmentRoot = [string]$Context.Settings.EnvironmentRoot
    $dataRoot = [string]$Context.Settings.DataRoot
    $coherentRoot = [string]$Context.Settings.CoherentMultiplexRoot
    $gpuVisibleDevices = [string]$Context.Settings.GpuVisibleDevices
    $python = Join-Path $environmentRoot "Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        throw "Shared WCT environment is missing Scripts\python.exe: $environmentRoot"
    }
    if (-not (Test-Path -LiteralPath (
        Join-Path $coherentRoot "utils\coherence_utils.py"
    ) -PathType Leaf)) {
        throw "CoherentMultiplexRoot is missing utils\coherence_utils.py: $coherentRoot"
    }
    $preflightHome = Join-Path $Context.ProgramRoot `
        ".local-orchestration\initialize-preflight-home"
    $saved = Set-WctTemporaryEnvironment -Values @{
        "_MNE_FAKE_HOME_DIR" = $preflightHome
        "PYTHONDONTWRITEBYTECODE" = "1"
    }
    try {
        Invoke-WctPythonProbe `
            -Python $python `
            -Mode "dependencies" `
            -Phase "InitializeProgram" `
            -WorkingDirectory $Context.ProjectRoot
    } finally {
        Restore-WctTemporaryEnvironment -Saved $saved
    }

    Write-LocalOrchestrationLog `
        -Phase "InitializeProgram" `
        -Level "OK" `
        -Source "Project:wct-gnn" `
        -Message "Validated shared environment and wavelet source; accepted configured EEG data root" `
        -LogPath $LogPath
    return @{
        RuntimeValues = @{
            EnvironmentRoot = $environmentRoot
            PythonExecutable = $python
            DataRoot = $dataRoot
            CoherentMultiplexRoot = $coherentRoot
            GpuVisibleDevices = $gpuVisibleDevices
        }
        ParentCopies = @()
    }
}

function Invoke-WctValidateIntegration {
    param([hashtable]$Context)

    $integration = Get-LocalOrchestrationPath `
        -Path $Context.General.IntegrationWorktree `
        -MustExist
    $python = Get-LocalOrchestrationPath `
        -Path $Context.ProjectRuntime.PythonExecutable `
        -MustExist
    $coherentRoot = Get-LocalOrchestrationPath `
        -Path $Context.ProjectRuntime.CoherentMultiplexRoot `
        -MustExist
    $saved = Set-WctTemporaryEnvironment -Values @{
        "PYTHONPATH" = @($integration, $coherentRoot) -join [IO.Path]::PathSeparator
        "LOCAL_ORCHESTRATION_WORKTREE_ROOT" = $integration
        "WCT_COHERENT_MULTIPLEX_ROOT" = $coherentRoot
        "_MNE_FAKE_HOME_DIR" = Join-Path $Context.ProgramRoot `
            ".local-orchestration\integration-preflight-home"
        "PYTHONDONTWRITEBYTECODE" = "1"
    }
    try {
        Invoke-WctPythonProbe `
            -Python $python `
            -Mode "sources" `
            -Phase "ValidateIntegration" `
            -WorkingDirectory $integration
    } finally {
        Restore-WctTemporaryEnvironment -Saved $saved
    }
    return @{}
}

function Invoke-WctConfigureExecution {
    param([hashtable]$Context)

    $worktree = Get-LocalOrchestrationPath `
        -Path $Context.WorktreeRoot `
        -MustExist
    $requestedResource = [string]$Context.RequestedResource
    if ($requestedResource -and $requestedResource -ne "gpu") {
        throw "WCT managed execution supports only the optional exclusive resource 'gpu', got '$requestedResource'."
    }
    $gpuRequested = $requestedResource -eq "gpu"
    $gpuVisibleDevices = [string]$Context.ProjectRuntime.GpuVisibleDevices
    if ($gpuRequested -and -not $gpuVisibleDevices) {
        throw "WCT GPU execution requires a configured GpuVisibleDevices value."
    }
    $computePolicy = if ($gpuRequested) { "gpu-exclusive" } else { "cpu-only" }
    $cudaVisibleDevices = if ($gpuRequested) { $gpuVisibleDevices } else { "-1" }
    $requestedCommand = [string]$Context.Command
    $commandArguments = @($Context.CommandArguments)
    $pythonModule = $null
    $commandKind = "PythonScript"
    $command = if ($requestedCommand -eq "-m") {
        if ($commandArguments.Count -eq 0) {
            throw "WCT managed Python module execution requires the module name as the first command argument."
        }
        $pythonModule = [string]$commandArguments[0]
        if (-not (Test-WctPythonModuleName -Name $pythonModule)) {
            throw "Invalid Python module name for WCT managed execution: $pythonModule"
        }
        $commandKind = "PythonModule"
        "-m"
    } elseif ([IO.Path]::IsPathRooted($requestedCommand)) {
        Get-LocalOrchestrationPath -Path $requestedCommand -MustExist
    } else {
        Get-LocalOrchestrationPath `
            -Path (Join-Path $worktree $requestedCommand) `
            -MustExist
    }
    if ($commandKind -eq "PythonScript") {
        if (-not (Test-LocalOrchestrationPathWithin `
            -Candidate $command `
            -Root $worktree
        )) {
            throw "WCT managed Python script must be below the assigned worktree: $command"
        }
        if ([IO.Path]::GetExtension($command) -ne ".py") {
            throw "WCT managed execution requires a worktree-local .py script or '-m <module>', got: $command"
        }
    }

    $python = Get-LocalOrchestrationPath `
        -Path $Context.ProjectRuntime.PythonExecutable `
        -MustExist
    $dataRoot = Get-LocalOrchestrationPath `
        -Path $Context.ProjectRuntime.DataRoot `
        -MustExist
    $coherentRoot = Get-LocalOrchestrationPath `
        -Path $Context.ProjectRuntime.CoherentMultiplexRoot `
        -MustExist
    $moabbResults = Join-Path $Context.ExecutionRoot "moabb"
    $mneHome = Join-Path $Context.ExecutionStateRoot "mne-home"
    $mneCache = Join-Path $Context.ExecutionStateRoot "mne-cache"
    $pythonPathParts = @($worktree, $coherentRoot)

    Write-LocalOrchestrationLog `
        -Phase "ConfigureExecution" `
        -Level "OK" `
        -Source "Project:wct-gnn" `
        -Message (
            "Configured data=$dataRoot results=$moabbResults and compute " +
            "policy '$computePolicy'"
        ) `
        -LogPath $LogPath
    $runtimeRecord = @{
        CommandKind = $commandKind
        Python = $python
        DataRoot = $dataRoot
        CoherentMultiplexRoot = $coherentRoot
        MoabbResults = $moabbResults
        MneHome = $mneHome
        MneCache = $mneCache
        ComputePolicy = $computePolicy
        CudaVisibleDevices = $cudaVisibleDevices
        DownloadsDisabled = $true
    }
    if ($pythonModule) {
        $runtimeRecord.PythonModule = $pythonModule
    }

    return @{
        Executable = $python
        PrefixArguments = @()
        # MNE requires an isolated home before import. Its optional cache path
        # is created by MNE only if a command actually uses disk caching.
        Directories = @($moabbResults, $mneHome)
        EnvironmentVariables = @{
            "PYTHONPATH" = $pythonPathParts -join [IO.Path]::PathSeparator
            "WCT_COHERENT_MULTIPLEX_ROOT" = $coherentRoot
            "MNE_DATA" = $dataRoot
            "MNE_DATASETS_BNCI_PATH" = $dataRoot
            "MOABB_RESULTS" = $moabbResults
            "_MNE_FAKE_HOME_DIR" = $mneHome
            "MNE_CACHE_DIR" = $mneCache
            "MOABB_DISABLE_DOWNLOADS" = "1"
            "PYTHONDONTWRITEBYTECODE" = "1"
            "CUDA_VISIBLE_DEVICES" = $cudaVisibleDevices
        }
        RuntimeRecord = $runtimeRecord
        ResolvedCommand = $command
    }
}

function Invoke-WctPreflightExecution {
    param([hashtable]$Context)

    $computePolicy = [string]$Context.ExecutionConfiguration.RuntimeRecord.ComputePolicy
    if ($computePolicy -eq "gpu-exclusive") {
        if (-not $Context.ContainsKey("ResourceLease") -or
            -not $Context.ResourceLease -or
            [string]$Context.ResourceLease.Resource -ne "gpu") {
            throw "WCT GPU preflight requires an acquired managed 'gpu' resource lease."
        }
    }
    $python = Get-LocalOrchestrationPath `
        -Path $Context.ExecutionConfiguration.Executable `
        -MustExist
    Invoke-WctPythonProbe `
        -Python $python `
        -Mode "execution" `
        -Phase "PreflightExecution" `
        -WorkingDirectory $Context.WorktreeRoot
    return @{}
}

function Invoke-WctBeforeCleanup {
    param([hashtable]$Context)

    $worktree = Get-LocalOrchestrationPath `
        -Path $Context.WorktreeRoot `
        -MustExist
    $externalRoots = @(
        $Context.ProjectRuntime.EnvironmentRoot,
        $Context.ProjectRuntime.DataRoot,
        $Context.ProjectRuntime.CoherentMultiplexRoot
    )
    foreach ($root in $externalRoots) {
        $resolved = Get-LocalOrchestrationPath -Path $root -MustExist
        if (Test-LocalOrchestrationPathWithin `
            -Candidate $resolved `
            -Root $worktree `
            -AllowRoot
        ) {
            throw "WCT external/shared root unexpectedly lies inside worker worktree: $resolved"
        }
    }
    Write-LocalOrchestrationLog `
        -Phase "BeforeCleanup" `
        -Level "OK" `
        -Source "Project:wct-gnn" `
        -Message "Verified environment, data, and wavelet roots are outside the worktree; adapter removed nothing" `
        -LogPath $LogPath
    return @{}
}

switch ($Operation) {
    "InitializeProgram" { return (Invoke-WctInitializeProgram -Context $Context) }
    "ValidateIntegration" { return (Invoke-WctValidateIntegration -Context $Context) }
    "PrepareWorktree" { return @{ Copies = @(); References = @() } }
    "ConfigureExecution" { return (Invoke-WctConfigureExecution -Context $Context) }
    "PreflightExecution" { return (Invoke-WctPreflightExecution -Context $Context) }
    "BeforeCleanup" { return (Invoke-WctBeforeCleanup -Context $Context) }
    default { throw "Unsupported WCT adapter operation: $Operation" }
}
