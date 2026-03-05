[CmdletBinding()]
param(
    [string]$PythonExe = "<redacted-path>",
    [string]$LlamaServerBin = "<redacted-path>",
    [string]$LlamaServerBinOverride = "",
    [string]$LlamaModelPath = "<redacted-path>",
    [string]$LlamaTuningArgs = "-fa on -dev Vulkan0 --no-cache-prompt --cache-ram 0",
    [string]$MlcModelRef = "HF://mlc-ai/Qwen2.5-7B-Instruct-q4f16_1-MLC",
    [string]$MlcDevice = "vulkan:0",
    [string]$MlcOpt = "O1",
    [string]$OrtServerUrl = "http://127.0.0.1:8100",
    [string]$OrtModelRef = "ORT://Qwen2.5-7B-Instruct",
    [string]$TokenizerRef = "Qwen/Qwen2.5-7B-Instruct",
    [string]$OrtTokenizerRef = "Qwen/Qwen2.5-7B-Instruct",
    [string]$TokenizerLocalRoot = "<redacted-path>",
    [string]$Workload = "workloads/w1_chat_512_512.yaml",
    [string]$OutRoot = "results/r14_14_4_gpu_grid_r1/",
    [int]$MaxAiperfAttempts = 2,
    [int]$PerProfileTimeoutMinutes = 60,
    [string[]]$PrimaryBackends = @("llama", "mlc", "ort"),
    [string[]]$ExperimentalBackends = @("torch_rocm"),
    [switch]$MainlineClosureFirst,
    [int]$ExperimentalMinimalProfiles = 1,
    [string]$AiperfGpuTelemetryMode = "adaptive",
    [string[]]$InterruptedRunIds = @(),
    [int]$HourlyCheckpointMinutes = 60,
    [int]$AiperfServiceRegistrationTimeoutSec = 90,
    [int]$AiperfServiceRegistrationMaxAttempts = 30,
    [int]$AiperfRequestTimeoutSeconds = 300,
    [int]$AiperfNoProgressTimeoutMinutes = 20,
    [string]$AiperfEndpointType = "chat",
    $AiperfEnableStreaming = $false,
    [int]$OrtNoProgressTimeoutMinutes = 12,
    [int]$PreflightTimeoutSeconds = 300,
    [int]$MlcCompileTimeoutSec = 1200,
    [string]$MlcCompileGlobalCacheRoot = "<redacted-path>",
    [double]$MinSampleRatio = 0.8,
    [double]$MinSystemFreeMemoryGb = 1.0,
    [string]$TorchPythonExe = "<redacted-path>",
    [int]$TorchServerPort = 8110,
    [string]$TorchModelId = "microsoft/Phi-3-mini-4k-instruct",
    [string]$TorchFallbackModelId = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    [string]$TorchModelRoot = "<redacted-path>",
    [int]$LlamaProfileLimit = 5,
    [int]$MlcProfileLimit = 5,
    [int]$TorchProfileLimit = 5,
    [int]$TorchProfileOverrideConcurrency = 0,
    [int]$TorchProfileOverrideRequestCount = 0,
    [int]$TorchProfileOverridePromptTokensMean = 0,
    [int]$TorchProfileOverrideOutputTokensMean = 0,
    [int]$LlamaProfileOverrideConcurrency = 0,
    [int]$LlamaProfileOverrideRequestCount = 0,
    [int]$LlamaProfileOverridePromptTokensMean = 0,
    [int]$LlamaProfileOverrideOutputTokensMean = 0,
    [string]$LlamaProfileIdOverride = "",
    [int]$MlcProfileOverrideConcurrency = 0,
    [int]$MlcProfileOverrideRequestCount = 0,
    [int]$MlcProfileOverridePromptTokensMean = 0,
    [int]$MlcProfileOverrideOutputTokensMean = 0,
    [int]$MlcPrefillChunkSizeOverride = 0,
    [string]$MlcProfileIdOverride = "",
    [int]$OrtProfileOverrideConcurrency = 0,
    [int]$OrtProfileOverrideRequestCount = 0,
    [int]$OrtProfileOverridePromptTokensMean = 0,
    [int]$OrtProfileOverrideOutputTokensMean = 0,
    [string]$OrtProfileIdOverride = "",
    [string]$TorchAttnImplementation = "sdpa",
    [string]$TorchSdpaKernelProfile = "balanced",
    $TorchHipForceDevKernarg = $true,
    [string]$TorchPreferredBlasBackend = "default",
    [string]$TorchPreferredRocmFaBackend = "default",
    [switch]$TorchEnableTunableOp,
    [switch]$TorchTunableOpTuning,
    [string]$TorchTunableOpResultsFile = "",
    [switch]$TorchEnableCompile,
    [string]$TorchCompileMode = "reduce-overhead",
    [switch]$TorchRequireGpu,
    [string]$TorchProfileIdOverride = "",
    [ValidateSet("on_demand", "preflight")]
    [string]$TorchAttachLifecycle = "on_demand",
    [switch]$SkipVisualizationFinalize,
    [switch]$DryRun
)

$scriptLibRoot = Split-Path -Parent $PSCommandPath
$scriptBackendLibRoot = Join-Path $scriptLibRoot "backends"
foreach ($lib in @(
    "common.ps1",
    "meta_io.ps1",
    "preflight.ps1",
    "watchdog.ps1",
    "aiperf.ps1",
    "compare.ps1",
    "visualization.ps1"
)) {
    $path = Join-Path $scriptLibRoot $lib
    if (Test-Path $path) { . $path }
}
foreach ($lib in @("llama.ps1", "mlc.ps1", "ort.ps1", "torch_rocm.ps1")) {
    $path = Join-Path $scriptBackendLibRoot $lib
    if (Test-Path $path) { . $path }
}

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $script:PSNativeCommandUseErrorActionPreference = $false
}

if ($MaxAiperfAttempts -lt 1) {
    throw "MaxAiperfAttempts must be >= 1"
}
if ($PerProfileTimeoutMinutes -lt 1) {
    throw "PerProfileTimeoutMinutes must be >= 1"
}
if ($HourlyCheckpointMinutes -lt 1) {
    throw "HourlyCheckpointMinutes must be >= 1"
}
if ($AiperfServiceRegistrationTimeoutSec -lt 1) {
    throw "AiperfServiceRegistrationTimeoutSec must be >= 1"
}
if ($AiperfServiceRegistrationMaxAttempts -lt 1) {
    throw "AiperfServiceRegistrationMaxAttempts must be >= 1"
}
if ($AiperfRequestTimeoutSeconds -lt 60) {
    throw "AiperfRequestTimeoutSeconds must be >= 60"
}
if ($AiperfNoProgressTimeoutMinutes -lt 5) {
    throw "AiperfNoProgressTimeoutMinutes must be >= 5"
}
$AiperfEndpointType = ([string]$AiperfEndpointType).Trim().ToLower()
if ([string]::IsNullOrWhiteSpace($AiperfEndpointType)) {
    throw "AiperfEndpointType must not be empty"
}
if ($AiperfEnableStreaming -is [bool]) {
    $AiperfEnableStreaming = [bool]$AiperfEnableStreaming
}
else {
    $aiperfEnableStreamingText = ([string]$AiperfEnableStreaming).Trim().ToLower()
    switch ($aiperfEnableStreamingText) {
        "1" { $AiperfEnableStreaming = $true; break }
        "true" { $AiperfEnableStreaming = $true; break }
        "yes" { $AiperfEnableStreaming = $true; break }
        "0" { $AiperfEnableStreaming = $false; break }
        "false" { $AiperfEnableStreaming = $false; break }
        "no" { $AiperfEnableStreaming = $false; break }
        default {
            throw "AiperfEnableStreaming must be a boolean-like value (true/false/1/0)"
        }
    }
}
$TorchHipForceDevKernargBool = $true
if ($TorchHipForceDevKernarg -is [bool]) {
    $TorchHipForceDevKernargBool = [bool]$TorchHipForceDevKernarg
}
else {
    $torchHipKernargText = ([string]$TorchHipForceDevKernarg).Trim().ToLower()
    switch ($torchHipKernargText) {
        "1" { $TorchHipForceDevKernargBool = $true; break }
        "true" { $TorchHipForceDevKernargBool = $true; break }
        "yes" { $TorchHipForceDevKernargBool = $true; break }
        "0" { $TorchHipForceDevKernargBool = $false; break }
        "false" { $TorchHipForceDevKernargBool = $false; break }
        "no" { $TorchHipForceDevKernargBool = $false; break }
        default {
            throw "TorchHipForceDevKernarg must be a boolean-like value (true/false/1/0)"
        }
    }
}
$TorchPreferredBlasBackend = ([string]$TorchPreferredBlasBackend).Trim().ToLower()
if ([string]::IsNullOrWhiteSpace($TorchPreferredBlasBackend)) {
    $TorchPreferredBlasBackend = "default"
}
if ($TorchPreferredBlasBackend -notin @("default", "cublas", "cublaslt", "ck")) {
    throw "TorchPreferredBlasBackend must be one of: default, cublas, cublaslt, ck"
}
$TorchPreferredRocmFaBackend = ([string]$TorchPreferredRocmFaBackend).Trim().ToLower()
if ([string]::IsNullOrWhiteSpace($TorchPreferredRocmFaBackend)) {
    $TorchPreferredRocmFaBackend = "default"
}
if ($TorchPreferredRocmFaBackend -notin @("default", "aotriton", "ck")) {
    throw "TorchPreferredRocmFaBackend must be one of: default, aotriton, ck"
}
if ($OrtNoProgressTimeoutMinutes -lt 1) {
    throw "OrtNoProgressTimeoutMinutes must be >= 1"
}
if ($PreflightTimeoutSeconds -lt 10) {
    throw "PreflightTimeoutSeconds must be >= 10"
}
if ($MlcCompileTimeoutSec -lt 60) {
    throw "MlcCompileTimeoutSec must be >= 60"
}
if ($MinSampleRatio -le 0 -or $MinSampleRatio -gt 1) {
    throw "MinSampleRatio must be in (0, 1]"
}
if ($MinSystemFreeMemoryGb -lt 1) {
    throw "MinSystemFreeMemoryGb must be >= 1"
}
if ($ExperimentalMinimalProfiles -lt 1) {
    throw "ExperimentalMinimalProfiles must be >= 1"
}
if ($LlamaProfileLimit -lt 1 -or $LlamaProfileLimit -gt 5) {
    throw "LlamaProfileLimit must be in [1, 5]"
}
if ($MlcProfileLimit -lt 1 -or $MlcProfileLimit -gt 5) {
    throw "MlcProfileLimit must be in [1, 5]"
}
if ($TorchProfileLimit -lt 1 -or $TorchProfileLimit -gt 5) {
    throw "TorchProfileLimit must be in [1, 5]"
}
$TorchAttachLifecycle = ([string]$TorchAttachLifecycle).Trim().ToLower()
if ([string]::IsNullOrWhiteSpace($TorchAttachLifecycle)) {
    throw "TorchAttachLifecycle must not be empty"
}
if ($TorchAttachLifecycle -notin @("on_demand", "preflight")) {
    throw "TorchAttachLifecycle must be one of: on_demand, preflight"
}
$torchProfileOverrideAllZero = (
    $TorchProfileOverrideConcurrency -eq 0 -and
    $TorchProfileOverrideRequestCount -eq 0 -and
    $TorchProfileOverridePromptTokensMean -eq 0 -and
    $TorchProfileOverrideOutputTokensMean -eq 0
)
$torchProfileOverrideAllEnabled = (
    $TorchProfileOverrideConcurrency -ge 1 -and
    $TorchProfileOverrideRequestCount -ge 1 -and
    $TorchProfileOverridePromptTokensMean -ge 1 -and
    $TorchProfileOverrideOutputTokensMean -ge 1
)
if (-not $torchProfileOverrideAllZero -and -not $torchProfileOverrideAllEnabled) {
    throw "TorchProfileOverride* must be either all 0 (disabled) or all >= 1 (enabled)"
}
$torchProfileOverrideEnabled = [bool]$torchProfileOverrideAllEnabled
$llamaProfileOverrideAllZero = (
    $LlamaProfileOverrideConcurrency -eq 0 -and
    $LlamaProfileOverrideRequestCount -eq 0 -and
    $LlamaProfileOverridePromptTokensMean -eq 0 -and
    $LlamaProfileOverrideOutputTokensMean -eq 0
)
$llamaProfileOverrideAllEnabled = (
    $LlamaProfileOverrideConcurrency -ge 1 -and
    $LlamaProfileOverrideRequestCount -ge 1 -and
    $LlamaProfileOverridePromptTokensMean -ge 1 -and
    $LlamaProfileOverrideOutputTokensMean -ge 1
)
if (-not $llamaProfileOverrideAllZero -and -not $llamaProfileOverrideAllEnabled) {
    throw "LlamaProfileOverride* must be either all 0 (disabled) or all >= 1 (enabled)"
}
$llamaProfileOverrideEnabled = [bool]$llamaProfileOverrideAllEnabled
$mlcProfileOverrideAllZero = (
    $MlcProfileOverrideConcurrency -eq 0 -and
    $MlcProfileOverrideRequestCount -eq 0 -and
    $MlcProfileOverridePromptTokensMean -eq 0 -and
    $MlcProfileOverrideOutputTokensMean -eq 0
)
$mlcProfileOverrideAllEnabled = (
    $MlcProfileOverrideConcurrency -ge 1 -and
    $MlcProfileOverrideRequestCount -ge 1 -and
    $MlcProfileOverridePromptTokensMean -ge 1 -and
    $MlcProfileOverrideOutputTokensMean -ge 1
)
if (-not $mlcProfileOverrideAllZero -and -not $mlcProfileOverrideAllEnabled) {
    throw "MlcProfileOverride* must be either all 0 (disabled) or all >= 1 (enabled)"
}
$mlcProfileOverrideEnabled = [bool]$mlcProfileOverrideAllEnabled
if ($MlcPrefillChunkSizeOverride -ne 0 -and $MlcPrefillChunkSizeOverride -lt 64) {
    throw "MlcPrefillChunkSizeOverride must be 0 (disabled) or >= 64"
}
$mlcPrefillChunkSizeOverrideEnabled = [bool]($MlcPrefillChunkSizeOverride -ge 64)
$ortProfileOverrideAllZero = (
    $OrtProfileOverrideConcurrency -eq 0 -and
    $OrtProfileOverrideRequestCount -eq 0 -and
    $OrtProfileOverridePromptTokensMean -eq 0 -and
    $OrtProfileOverrideOutputTokensMean -eq 0
)
$ortProfileOverrideAllEnabled = (
    $OrtProfileOverrideConcurrency -ge 1 -and
    $OrtProfileOverrideRequestCount -ge 1 -and
    $OrtProfileOverridePromptTokensMean -ge 1 -and
    $OrtProfileOverrideOutputTokensMean -ge 1
)
if (-not $ortProfileOverrideAllZero -and -not $ortProfileOverrideAllEnabled) {
    throw "OrtProfileOverride* must be either all 0 (disabled) or all >= 1 (enabled)"
}
$ortProfileOverrideEnabled = [bool]$ortProfileOverrideAllEnabled
$AiperfGpuTelemetryMode = ([string]$AiperfGpuTelemetryMode).Trim().ToLower()
$validAiperfGpuTelemetryModes = @("adaptive", "strict", "off")
if ($AiperfGpuTelemetryMode -notin $validAiperfGpuTelemetryModes) {
    throw "AiperfGpuTelemetryMode must be one of: $($validAiperfGpuTelemetryModes -join ', ')"
}
$validTorchSdpaProfiles = @("balanced", "flash_only", "mem_efficient_only", "math_only")
if ($TorchSdpaKernelProfile -notin $validTorchSdpaProfiles) {
    throw "TorchSdpaKernelProfile must be one of: $($validTorchSdpaProfiles -join ', ')"
}
$TorchTunableOpResultsFile = ([string]$TorchTunableOpResultsFile).Trim()
if ((-not [bool]$TorchEnableTunableOp) -and [bool]$TorchTunableOpTuning) {
    throw "TorchTunableOpTuning requires TorchEnableTunableOp"
}
if ((-not [bool]$TorchEnableTunableOp) -and -not [string]::IsNullOrWhiteSpace($TorchTunableOpResultsFile)) {
    throw "TorchTunableOpResultsFile requires TorchEnableTunableOp"
}

$outRootResolved = [System.IO.Path]::GetFullPath($OutRoot)
$mlcCompileGlobalCacheRootResolved = [System.IO.Path]::GetFullPath($MlcCompileGlobalCacheRoot)
$torchTunableOpResultsFileResolved = if ([string]::IsNullOrWhiteSpace($TorchTunableOpResultsFile)) { "" } else { [System.IO.Path]::GetFullPath($TorchTunableOpResultsFile) }
$runsRoot = Join-Path $outRootResolved "runs"
$compareRoot = Join-Path $outRootResolved "compare"
$gpuUtilRoot = Join-Path $outRootResolved "gpu_utilization"
$leaderboardPath = Join-Path $outRootResolved "leaderboard.csv"
$matrixReportPath = Join-Path $outRootResolved "matrix_report.md"
$pipelineMetaPath = Join-Path $outRootResolved "pipeline_meta.json"
$perfTimelineRoot = [System.IO.Path]::GetFullPath("reports/perf_timeline")
$perfTimelineReportPath = Join-Path $perfTimelineRoot "timeline_report.md"
$rgpRawRoot = [System.IO.Path]::GetFullPath("reports/rgp_raw")
$rgpSummaryPath = Join-Path $perfTimelineRoot "rgp_summary.csv"
$rgpReportPath = Join-Path $perfTimelineRoot "rgp_report.md"
$torchServerLogDir = Join-Path $outRootResolved "torch_server"
$torchServerStdoutPath = Join-Path $torchServerLogDir "server_stdout.log"
$torchServerStderrPath = Join-Path $torchServerLogDir "server_stderr.log"
$torchServerMetaPath = Join-Path $torchServerLogDir "server_meta.json"
$torchServerUrl = "http://127.0.0.1:$TorchServerPort"
$perProfileTimeoutSec = $PerProfileTimeoutMinutes * 60
$aiperfNoProgressTimeoutSec = $AiperfNoProgressTimeoutMinutes * 60
$ortNoProgressTimeoutSec = $OrtNoProgressTimeoutMinutes * 60
$script:MlcOptEffective = $MlcOpt
$script:LongRunAutoClosed = $false
$script:LastHourlyCheckpointAt = $null
$script:AiperfEnvOriginal = @{}
$script:AiperfEnvApplied = $false
$script:MlcCompileEnvOriginal = $null
$script:MlcCompileGlobalBlocked = $false
$script:MlcCompileGlobalBlocker = $null
$script:MlcCompileGlobalError = $null
$script:ResolvedTokenizerRefMain = $TokenizerRef
$script:ResolvedTokenizerRefOrt = $OrtTokenizerRef
$script:NoProgressAbortCount = 0
$script:AdaptiveProfileDowngradeCount = 0
$script:LlamaKvFallbackCount = 0
$script:GpuTelemetryDegradedCount = 0
$script:GpuTelemetryHardFailCount = 0
$script:LlamaServerBinEffective = if ([string]::IsNullOrWhiteSpace($LlamaServerBinOverride)) {
    [System.IO.Path]::GetFullPath($LlamaServerBin)
}
else {
    [System.IO.Path]::GetFullPath($LlamaServerBinOverride)
}

# Stage A: full sweep (mainline + experimental)
# Stage B: mainline top1 recheck

function Move-ProfileIdFirst {
    param(
        [object[]]$Profiles,
        [string]$ProfileId,
        [string]$BackendName
    )
    if ([string]::IsNullOrWhiteSpace($ProfileId)) {
        return @($Profiles)
    }
    $idNorm = ([string]$ProfileId).Trim().ToLower()
    $match = @($Profiles | Where-Object { ([string]$_.id).Trim().ToLower() -eq $idNorm } | Select-Object -First 1)
    if ($match.Count -eq 0) {
        throw "$BackendName profile id not found for override: $ProfileId"
    }
    $matchedId = ([string]$match[0].id).Trim().ToLower()
    $rest = @($Profiles | Where-Object { ([string]$_.id).Trim().ToLower() -ne $matchedId })
    return @($match[0]) + @($rest)
}

$llamaProfiles = @(
    [ordered]@{ id = "L0"; order = 0; c = 4096; b = 2304; ub = 1152; ngl = 99; ctk = "q8_0"; ctv = "q8_0"; cb = $true; kvu = $true; concurrency = 4; request_count = 10; prompt_tokens_mean = 160; output_tokens_mean = 160; conversation_turn_mean = 1 },
    [ordered]@{ id = "L1"; order = 1; c = 4096; b = 2304; ub = 1152; ngl = 99; ctk = "q8_0"; ctv = "q8_0"; cb = $true; kvu = $true; concurrency = 5; request_count = 8; prompt_tokens_mean = 160; output_tokens_mean = 160; conversation_turn_mean = 1 },
    [ordered]@{ id = "L2"; order = 2; c = 4096; b = 2304; ub = 1152; ngl = 99; ctk = "q8_0"; ctv = "q8_0"; cb = $true; kvu = $true; concurrency = 5; request_count = 8; prompt_tokens_mean = 160; output_tokens_mean = 192; conversation_turn_mean = 1 },
    [ordered]@{ id = "L3"; order = 3; c = 4096; b = 2048; ub = 1024; ngl = 99; ctk = "q8_0"; ctv = "q8_0"; cb = $true; kvu = $true; concurrency = 6; request_count = 8; prompt_tokens_mean = 128; output_tokens_mean = 160; conversation_turn_mean = 1 },
    [ordered]@{ id = "L4"; order = 4; c = 4096; b = 1792; ub = 896; ngl = 99; ctk = "q8_0"; ctv = "q8_0"; cb = $true; kvu = $true; concurrency = 6; request_count = 8; prompt_tokens_mean = 128; output_tokens_mean = 128; conversation_turn_mean = 1 }
)
if ($llamaProfileOverrideEnabled) {
    $llamaProfilesOverridden = @()
    foreach ($profile in @($llamaProfiles)) {
        $profileOverridden = [ordered]@{}
        foreach ($key in $profile.Keys) {
            $profileOverridden[$key] = $profile[$key]
        }
        $profileOverridden["concurrency"] = [int]$LlamaProfileOverrideConcurrency
        $profileOverridden["request_count"] = [int]$LlamaProfileOverrideRequestCount
        $profileOverridden["prompt_tokens_mean"] = [int]$LlamaProfileOverridePromptTokensMean
        $profileOverridden["output_tokens_mean"] = [int]$LlamaProfileOverrideOutputTokensMean
        $llamaProfilesOverridden += $profileOverridden
    }
    $llamaProfiles = @($llamaProfilesOverridden)
}
$llamaProfiles = @(Move-ProfileIdFirst -Profiles $llamaProfiles -ProfileId $LlamaProfileIdOverride -BackendName "llama")
$llamaProfileLimitEffective = [Math]::Min([int]$LlamaProfileLimit, $llamaProfiles.Count)
$llamaProfiles = @($llamaProfiles | Select-Object -First $llamaProfileLimitEffective)

$mlcProfiles = @(
    [ordered]@{ id = "M0"; order = 0; max_num_sequence = 12; max_total_seq_length = 4096; prefill_chunk_size = 384; mode = "server"; concurrency = 4; request_count = 8; prompt_tokens_mean = 256; output_tokens_mean = 256; conversation_turn_mean = 1 },
    [ordered]@{ id = "M1"; order = 1; max_num_sequence = 12; max_total_seq_length = 4096; prefill_chunk_size = 384; mode = "server"; concurrency = 5; request_count = 8; prompt_tokens_mean = 224; output_tokens_mean = 224; conversation_turn_mean = 1 },
    [ordered]@{ id = "M2"; order = 2; max_num_sequence = 12; max_total_seq_length = 4096; prefill_chunk_size = 320; mode = "server"; concurrency = 5; request_count = 8; prompt_tokens_mean = 224; output_tokens_mean = 224; conversation_turn_mean = 1 },
    [ordered]@{ id = "M3"; order = 3; max_num_sequence = 14; max_total_seq_length = 4096; prefill_chunk_size = 320; mode = "server"; concurrency = 5; request_count = 8; prompt_tokens_mean = 192; output_tokens_mean = 224; conversation_turn_mean = 1 },
    [ordered]@{ id = "M4"; order = 4; max_num_sequence = 14; max_total_seq_length = 4096; prefill_chunk_size = 256; mode = "server"; concurrency = 6; request_count = 8; prompt_tokens_mean = 192; output_tokens_mean = 192; conversation_turn_mean = 1 }
)
if ($mlcProfileOverrideEnabled) {
    $mlcProfilesOverridden = @()
    foreach ($profile in @($mlcProfiles)) {
        $profileOverridden = [ordered]@{}
        foreach ($key in $profile.Keys) {
            $profileOverridden[$key] = $profile[$key]
        }
        $profileOverridden["concurrency"] = [int]$MlcProfileOverrideConcurrency
        $profileOverridden["request_count"] = [int]$MlcProfileOverrideRequestCount
        $profileOverridden["prompt_tokens_mean"] = [int]$MlcProfileOverridePromptTokensMean
        $profileOverridden["output_tokens_mean"] = [int]$MlcProfileOverrideOutputTokensMean
        $mlcProfilesOverridden += $profileOverridden
    }
    $mlcProfiles = @($mlcProfilesOverridden)
}
if ($mlcPrefillChunkSizeOverrideEnabled) {
    $mlcProfilesPrefillOverridden = @()
    foreach ($profile in @($mlcProfiles)) {
        $profileOverridden = [ordered]@{}
        foreach ($key in $profile.Keys) {
            $profileOverridden[$key] = $profile[$key]
        }
        $profileOverridden["prefill_chunk_size"] = [int]$MlcPrefillChunkSizeOverride
        $mlcProfilesPrefillOverridden += $profileOverridden
    }
    $mlcProfiles = @($mlcProfilesPrefillOverridden)
}
$mlcProfiles = @(Move-ProfileIdFirst -Profiles $mlcProfiles -ProfileId $MlcProfileIdOverride -BackendName "mlc")
$mlcProfileLimitEffective = [Math]::Min([int]$MlcProfileLimit, $mlcProfiles.Count)
$mlcProfiles = @($mlcProfiles | Select-Object -First $mlcProfileLimitEffective)

$ortProfiles = @(
    [ordered]@{ id = "O0"; order = 0; concurrency = 2; request_count = 6; prompt_tokens_mean = 80; output_tokens_mean = 64; conversation_turn_mean = 1 },
    [ordered]@{ id = "O1"; order = 1; concurrency = 2; request_count = 8; prompt_tokens_mean = 80; output_tokens_mean = 64; conversation_turn_mean = 1 },
    [ordered]@{ id = "O2"; order = 2; concurrency = 2; request_count = 8; prompt_tokens_mean = 96; output_tokens_mean = 64; conversation_turn_mean = 1 },
    [ordered]@{ id = "O3"; order = 3; concurrency = 2; request_count = 8; prompt_tokens_mean = 112; output_tokens_mean = 64; conversation_turn_mean = 1 },
    [ordered]@{ id = "O4"; order = 4; concurrency = 2; request_count = 8; prompt_tokens_mean = 128; output_tokens_mean = 64; conversation_turn_mean = 1 }
)
if ($ortProfileOverrideEnabled) {
    $ortProfilesOverridden = @()
    foreach ($profile in @($ortProfiles)) {
        $profileOverridden = [ordered]@{}
        foreach ($key in $profile.Keys) {
            $profileOverridden[$key] = $profile[$key]
        }
        $profileOverridden["concurrency"] = [int]$OrtProfileOverrideConcurrency
        $profileOverridden["request_count"] = [int]$OrtProfileOverrideRequestCount
        $profileOverridden["prompt_tokens_mean"] = [int]$OrtProfileOverridePromptTokensMean
        $profileOverridden["output_tokens_mean"] = [int]$OrtProfileOverrideOutputTokensMean
        $ortProfilesOverridden += $profileOverridden
    }
    $ortProfiles = @($ortProfilesOverridden)
}
$ortProfiles = @(Move-ProfileIdFirst -Profiles $ortProfiles -ProfileId $OrtProfileIdOverride -BackendName "ort")

$torchProfiles = @(
    [ordered]@{ id = "T0"; order = 0; concurrency = 4; request_count = 8; prompt_tokens_mean = 192; output_tokens_mean = 192; conversation_turn_mean = 1 },
    [ordered]@{ id = "T1"; order = 1; concurrency = 6; request_count = 10; prompt_tokens_mean = 256; output_tokens_mean = 256; conversation_turn_mean = 1 },
    [ordered]@{ id = "T2"; order = 2; concurrency = 8; request_count = 12; prompt_tokens_mean = 320; output_tokens_mean = 320; conversation_turn_mean = 1 },
    [ordered]@{ id = "T3"; order = 3; concurrency = 8; request_count = 14; prompt_tokens_mean = 384; output_tokens_mean = 384; conversation_turn_mean = 1 },
    [ordered]@{ id = "T4"; order = 4; concurrency = 10; request_count = 14; prompt_tokens_mean = 448; output_tokens_mean = 384; conversation_turn_mean = 1 }
)
$torchProfiles = @(Move-ProfileIdFirst -Profiles $torchProfiles -ProfileId $TorchProfileIdOverride -BackendName "torch_rocm")
$torchProfileLimitEffective = [Math]::Min([int]$TorchProfileLimit, $torchProfiles.Count)
$torchProfiles = @($torchProfiles | Select-Object -First $torchProfileLimitEffective)
if ($torchProfileOverrideEnabled) {
    $torchProfilesOverridden = @()
    foreach ($profile in @($torchProfiles)) {
        $profileOverridden = [ordered]@{}
        foreach ($key in $profile.Keys) {
            $profileOverridden[$key] = $profile[$key]
        }
        $profileOverridden["concurrency"] = [int]$TorchProfileOverrideConcurrency
        $profileOverridden["request_count"] = [int]$TorchProfileOverrideRequestCount
        $profileOverridden["prompt_tokens_mean"] = [int]$TorchProfileOverridePromptTokensMean
        $profileOverridden["output_tokens_mean"] = [int]$TorchProfileOverrideOutputTokensMean
        $torchProfilesOverridden += $profileOverridden
    }
    $torchProfiles = @($torchProfilesOverridden)
}

$profilesByBackend = [ordered]@{
    llama = $llamaProfiles
    mlc = $mlcProfiles
    ort = $ortProfiles
    torch_rocm = $torchProfiles
}
$supportedBackends = @("llama", "mlc", "ort", "torch_rocm")

function Normalize-BackendSelection {
    param(
        [string[]]$InputBackends,
        [string]$ParameterName
    )
    $selected = @()
    foreach ($item in @($InputBackends)) {
        if ([string]::IsNullOrWhiteSpace([string]$item)) {
            continue
        }
        $name = [string]$item
        $parts = @($name -split ",")
        foreach ($part in $parts) {
            if ([string]::IsNullOrWhiteSpace([string]$part)) {
                continue
            }
            $selected += ([string]$part).Trim().ToLower()
        }
    }
    $selected = @($selected | Select-Object -Unique)
    foreach ($name in $selected) {
        if ($name -notin $supportedBackends) {
            throw "$ParameterName contains unsupported backend '$name'. Supported values: $($supportedBackends -join ', ')"
        }
    }
    return $selected
}

$primaryBackends = @(Normalize-BackendSelection -InputBackends $PrimaryBackends -ParameterName "PrimaryBackends")
if ($primaryBackends.Count -eq 0) {
    throw "PrimaryBackends must contain at least one backend"
}
$experimentalBackendsRaw = @(Normalize-BackendSelection -InputBackends $ExperimentalBackends -ParameterName "ExperimentalBackends")
$experimentalBackends = @($experimentalBackendsRaw | Where-Object { $_ -notin $primaryBackends })
$mainlineBackendOrder = @($primaryBackends)
$experimentalBackendOrder = @($experimentalBackends)
$experimentalProfilesByBackend = [ordered]@{}
foreach ($backend in $experimentalBackendOrder) {
    $backendProfiles = @($profilesByBackend[$backend])
    if ($MainlineClosureFirst) {
        $take = [Math]::Min([int]$ExperimentalMinimalProfiles, $backendProfiles.Count)
        $experimentalProfilesByBackend[$backend] = @($backendProfiles | Select-Object -First $take)
    }
    else {
        $experimentalProfilesByBackend[$backend] = @($backendProfiles)
    }
}
$selectedBackends = @($primaryBackends + $experimentalBackends)
if ($selectedBackends.Count -eq 0) {
    throw "At least one backend must be selected"
}
$ortBackendSelected = @($selectedBackends) -contains "ort"
$torchBackendSelected = @($selectedBackends) -contains "torch_rocm"

$stageAMainlineProfileTotal = [int](($primaryBackends | ForEach-Object { @($profilesByBackend[$_]).Count } | Measure-Object -Sum).Sum)
$stageAExperimentalProfileTotal = [int](($experimentalBackendOrder | ForEach-Object { @($experimentalProfilesByBackend[$_]).Count } | Measure-Object -Sum).Sum)
$stageAProfileTotal = [int]($stageAMainlineProfileTotal + $stageAExperimentalProfileTotal)
$stageBProfileTotal = [int]$primaryBackends.Count
if ($MainlineClosureFirst) {
    $stageNames = @(
        "stage_a_full_${stageAMainlineProfileTotal}_mainline",
        "stage_b_top1_recheck_${stageBProfileTotal}_mainline",
        "stage_a_experimental_${stageAExperimentalProfileTotal}"
    )
}
else {
    $stageNames = @(
        "stage_a_full_${stageAMainlineProfileTotal}_mainline",
        "stage_a_experimental_${stageAExperimentalProfileTotal}",
        "stage_b_top1_recheck_${stageBProfileTotal}_mainline"
    )
}

$pipelineMeta = [ordered]@{
    schema_version = "gpu_util_uplift_v3"
    generated_at_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    dry_run = [bool]$DryRun
    config = [ordered]@{
        python_exe = $PythonExe
        torch_python_exe = $TorchPythonExe
        llama_server_bin_requested = $LlamaServerBin
        llama_server_bin_override = $LlamaServerBinOverride
        llama_server_bin_effective = $script:LlamaServerBinEffective
        workload = $Workload
        max_attempts = $MaxAiperfAttempts
        per_profile_timeout_minutes = $PerProfileTimeoutMinutes
        per_profile_timeout_sec = $perProfileTimeoutSec
        stages = $stageNames
        profile_count_per_backend = 5
        stage_a_profile_total = $stageAProfileTotal
        stage_a_mainline_profile_total = $stageAMainlineProfileTotal
        stage_a_experimental_profile_total = $stageAExperimentalProfileTotal
        stage_b_profile_total = $stageBProfileTotal
        backends = $selectedBackends
        primary_backends = $primaryBackends
        experimental_backends = $experimentalBackends
        mainline_closure_first = [bool]$MainlineClosureFirst
        experimental_minimal_profiles = [int]$ExperimentalMinimalProfiles
        aiperf_gpu_telemetry_mode = $AiperfGpuTelemetryMode
        interrupted_run_ids = @($InterruptedRunIds)
        hourly_checkpoint_minutes = $HourlyCheckpointMinutes
        aiperf_service_registration_timeout_sec = $AiperfServiceRegistrationTimeoutSec
        aiperf_service_registration_max_attempts = $AiperfServiceRegistrationMaxAttempts
        aiperf_request_timeout_seconds = $AiperfRequestTimeoutSeconds
        aiperf_no_progress_timeout_minutes = $AiperfNoProgressTimeoutMinutes
        aiperf_no_progress_timeout_sec = $aiperfNoProgressTimeoutSec
        aiperf_endpoint_type = $AiperfEndpointType
        aiperf_streaming_enabled = [bool]$AiperfEnableStreaming
        ttft_measurement_mode = if ([bool]$AiperfEnableStreaming) { "streaming_first_token" } else { "non_stream_fallback_possible" }
        ort_no_progress_timeout_minutes = $OrtNoProgressTimeoutMinutes
        ort_no_progress_timeout_sec = $ortNoProgressTimeoutSec
        aiperf_service_connection_probe_timeout_sec = 180
        aiperf_service_profile_start_timeout_sec = 120
        aiperf_service_profile_configure_timeout_sec = 300
        mlc_compile_timeout_sec = $MlcCompileTimeoutSec
        mlc_compile_global_cache_root = $mlcCompileGlobalCacheRootResolved
        min_sample_ratio = $MinSampleRatio
        min_system_free_memory_gb = [double]$MinSystemFreeMemoryGb
        tokenizer_local_root = $TokenizerLocalRoot
        tokenizer_ref_effective = $TokenizerRef
        ort_tokenizer_ref_effective = $OrtTokenizerRef
        ort_preflight_required = [bool]$ortBackendSelected
        ort_url_conflict_check_enabled = [bool]$ortBackendSelected
        ort_tokenizer_cache_required = [bool]$ortBackendSelected
        ort_server_url = $OrtServerUrl
        ort_profile_override_enabled = [bool]$ortProfileOverrideEnabled
        ort_profile_override_concurrency = [int]$OrtProfileOverrideConcurrency
        ort_profile_override_request_count = [int]$OrtProfileOverrideRequestCount
        ort_profile_override_prompt_tokens_mean = [int]$OrtProfileOverridePromptTokensMean
        ort_profile_override_output_tokens_mean = [int]$OrtProfileOverrideOutputTokensMean
        ort_profile_id_override = ([string]$OrtProfileIdOverride).Trim()
        llama_profile_override_enabled = [bool]$llamaProfileOverrideEnabled
        llama_profile_override_concurrency = [int]$LlamaProfileOverrideConcurrency
        llama_profile_override_request_count = [int]$LlamaProfileOverrideRequestCount
        llama_profile_override_prompt_tokens_mean = [int]$LlamaProfileOverridePromptTokensMean
        llama_profile_override_output_tokens_mean = [int]$LlamaProfileOverrideOutputTokensMean
        llama_profile_id_override = ([string]$LlamaProfileIdOverride).Trim()
        mlc_profile_override_enabled = [bool]$mlcProfileOverrideEnabled
        mlc_profile_override_concurrency = [int]$MlcProfileOverrideConcurrency
        mlc_profile_override_request_count = [int]$MlcProfileOverrideRequestCount
        mlc_profile_override_prompt_tokens_mean = [int]$MlcProfileOverridePromptTokensMean
        mlc_profile_override_output_tokens_mean = [int]$MlcProfileOverrideOutputTokensMean
        mlc_prefill_chunk_size_override_enabled = [bool]$mlcPrefillChunkSizeOverrideEnabled
        mlc_prefill_chunk_size_override = [int]$MlcPrefillChunkSizeOverride
        mlc_profile_id_override = ([string]$MlcProfileIdOverride).Trim()
        llama_profile_limit = [int]$llamaProfileLimitEffective
        mlc_profile_limit = [int]$mlcProfileLimitEffective
        torch_server_url = $torchServerUrl
        torch_preflight_required = [bool]$torchBackendSelected
        torch_model_id = $TorchModelId
        torch_fallback_model_id = $TorchFallbackModelId
        torch_profile_limit = [int]$torchProfileLimitEffective
        torch_profile_override_enabled = [bool]$torchProfileOverrideEnabled
        torch_profile_override_concurrency = [int]$TorchProfileOverrideConcurrency
        torch_profile_override_request_count = [int]$TorchProfileOverrideRequestCount
        torch_profile_override_prompt_tokens_mean = [int]$TorchProfileOverridePromptTokensMean
        torch_profile_override_output_tokens_mean = [int]$TorchProfileOverrideOutputTokensMean
        torch_profile_id_override = ([string]$TorchProfileIdOverride).Trim()
        torch_attach_lifecycle = $TorchAttachLifecycle
        torch_attn_implementation = $TorchAttnImplementation
        torch_sdpa_kernel_profile = $TorchSdpaKernelProfile
        torch_hip_force_dev_kernarg = [bool]$TorchHipForceDevKernargBool
        torch_preferred_blas_backend = $TorchPreferredBlasBackend
        torch_preferred_rocm_fa_backend = $TorchPreferredRocmFaBackend
        torch_tunableop_enabled = [bool]$TorchEnableTunableOp
        torch_tunableop_tuning = [bool]$TorchTunableOpTuning
        torch_tunableop_results_file = $torchTunableOpResultsFileResolved
        torch_enable_compile = [bool]$TorchEnableCompile
        torch_compile_mode = $TorchCompileMode
        torch_require_gpu = [bool]$TorchRequireGpu
        skip_visualization_finalize = [bool]$SkipVisualizationFinalize
    }
    paths = [ordered]@{
        out_root = $outRootResolved
        runs = $runsRoot
        compare = $compareRoot
        gpu_utilization = $gpuUtilRoot
        leaderboard = $leaderboardPath
        matrix_report = $matrixReportPath
        pipeline_meta = $pipelineMetaPath
    }
    stage_results = [ordered]@{
        stage_a = @()
        stage_b = @()
    }
    execution = [ordered]@{
        primary_backends = $primaryBackends
        experimental_backends = $experimentalBackends
        hourly_checkpoint_minutes = $HourlyCheckpointMinutes
        hourly_checkpoints = @()
        last_checkpoint_utc = $null
        long_run_auto_closed = $false
        mainline_completed = $false
        mainline_closure_first = [bool]$MainlineClosureFirst
        mainline_stage_a_completed = 0
        mainline_stage_b_completed = 0
        gpu_telemetry_mode = $AiperfGpuTelemetryMode
        gpu_telemetry_degraded_count = 0
        gpu_telemetry_hard_fail_count = 0
        no_progress_abort_count = 0
        adaptive_profile_downgrade_count = 0
        llama_kv_fallback_count = 0
        interrupted_runs = @()
        mlc_compile_preflight = [ordered]@{
            by_profile = [ordered]@{}
            by_key = [ordered]@{}
            blocked_profiles = @()
            global_blocker = $null
            global_error = $null
        }
        torch_gpu_required = [bool]$TorchRequireGpu
        torch_gpu_verified = $null
    }
    progress = [ordered]@{
        state = "initialized"
        stage_a_completed = 0
        stage_a_total = $stageAProfileTotal
        stage_b_completed = 0
        stage_b_total = $stageBProfileTotal
        completed_profiles = 0
        last_run_id = $null
        last_update_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    }
    best = [ordered]@{}
    torch_server = [ordered]@{
        selected_model_id = $null
        fallback_triggered = $null
        fallback_reason_signature = $null
        device = $null
        runtime_device_fallback = $null
        gpu_required_pass = $null
        cache_root = $null
        operator_optimizations = $null
    }
    compare_winners = [ordered]@{}
    visualization = [ordered]@{
        report_path = $perfTimelineReportPath
        generated_at_utc = $null
        ready = $false
        error = $null
        rgp_report_path = $rgpReportPath
        rgp_ready = $false
        rgp_error = $null
    }
    d_drive_gate_pass = $false
    blocker_signature = $null
    artifacts_ready = $false
    steps = @()
}

$allRows = @()
$script:TorchServerProc = $null
$script:TorchServerManaged = $false
$script:MlcCompilePreflightDir = Join-Path $outRootResolved "mlc_compile_preflight"
$script:MlcCompileGlobalCacheRoot = $mlcCompileGlobalCacheRootResolved
$script:MlcLlmHome = Join-Path $script:MlcCompileGlobalCacheRoot "mlc_llm_home"
$script:MlcCompilePreflightByProfileId = @{}
$script:MlcCompileCacheByKey = @{}
function Test-IsDDrivePath {
    param([Parameter(Mandatory = $true)][string]$PathLike)
    $normalized = $PathLike.Replace("\", "/")
    return $normalized.StartsWith("D:/", [System.StringComparison]::OrdinalIgnoreCase)
}

function Assert-DDrivePath {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$PathLike
    )
    if (-not (Test-IsDDrivePath -PathLike $PathLike)) {
        throw "$Name must be on D drive: $PathLike"
    }
}

function Ensure-LocalTokenizerCache {
    param(
        [Parameter(Mandatory = $true)][string]$ModelRef,
        [Parameter(Mandatory = $true)][string]$Alias
    )
    if ($DryRun) {
        return $null
    }
    if ([string]::IsNullOrWhiteSpace($TokenizerLocalRoot)) {
        return $null
    }
    Assert-DDrivePath -Name "TokenizerLocalRoot" -PathLike $TokenizerLocalRoot
    New-Item -ItemType Directory -Path $TokenizerLocalRoot -Force | Out-Null

    $slug = (($ModelRef -replace "[^A-Za-z0-9._-]", "_").Trim("_"))
    if ([string]::IsNullOrWhiteSpace($slug)) {
        $slug = "tokenizer_$Alias"
    }
    $targetDir = Join-Path $TokenizerLocalRoot $slug
    $tokenizerConfig = Join-Path $targetDir "tokenizer_config.json"
    if (Test-Path $tokenizerConfig) {
        Add-Step -Name "preflight-tokenizer-cache-$Alias" -Command "reuse local tokenizer cache: $targetDir" -ReturnCode 0 -Status "executed"
        return $targetDir
    }

    New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
    $scriptPath = Join-Path $targetDir "ensure_local_tokenizer.py"
    $scriptText = @'
import argparse
import traceback
from pathlib import Path

from transformers import AutoTokenizer


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-ref", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        tok = AutoTokenizer.from_pretrained(args.model_ref, trust_remote_code=True)
        tok.save_pretrained(out_dir)
        print(str(out_dir))
        return 0
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
'@
    Set-Content -Path $scriptPath -Value $scriptText -Encoding UTF8
    $rc = Invoke-Step -Name "preflight-tokenizer-cache-$Alias" -Executable $PythonExe -Arguments @(
        $scriptPath,
        "--model-ref", $ModelRef,
        "--out-dir", $targetDir
    ) -TimeoutSeconds $PreflightTimeoutSeconds -AllowFailure
    if ($rc -eq 0 -and (Test-Path $tokenizerConfig)) {
        return $targetDir
    }
    return $null
}

function ConvertTo-WindowsArg {
    param([string]$Arg)
    if ($null -eq $Arg) {
        return '""'
    }
    if ($Arg -notmatch '[\s"]') {
        return $Arg
    }
    $escaped = $Arg -replace '"', '\\"'
    return '"' + $escaped + '"'
}

function Format-Command {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [string[]]$Arguments
    )
    $parts = @($Executable)
    if ($Arguments) {
        $parts += $Arguments
    }
    return (($parts | ForEach-Object { ConvertTo-WindowsArg -Arg $_ }) -join " ")
}

function Get-DescendantProcessIds {
    param([Parameter(Mandatory = $true)][int]$RootProcessId)
    $all = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
    if (@($all).Count -eq 0) {
        return @()
    }
    $childrenByParent = @{}
    foreach ($p in $all) {
        $ppid = [int]$p.ParentProcessId
        if (-not $childrenByParent.ContainsKey($ppid)) {
            $childrenByParent[$ppid] = New-Object System.Collections.Generic.List[int]
        }
        $childrenByParent[$ppid].Add([int]$p.ProcessId)
    }
    $result = New-Object System.Collections.Generic.List[int]
    $stack = New-Object System.Collections.Generic.Stack[int]
    $stack.Push($RootProcessId)
    while ($stack.Count -gt 0) {
        $current = $stack.Pop()
        if (-not $childrenByParent.ContainsKey($current)) {
            continue
        }
        foreach ($cid in $childrenByParent[$current]) {
            $result.Add($cid)
            $stack.Push($cid)
        }
    }
    return @($result)
}

function Stop-ProcessTree {
    param([Parameter(Mandatory = $true)][int]$RootProcessId)
    $descendants = @(Get-DescendantProcessIds -RootProcessId $RootProcessId)
    $allIds = @($descendants + $RootProcessId) | Sort-Object -Descending -Unique
    foreach ($procId in $allIds) {
        try {
            Stop-Process -Id $procId -Force -ErrorAction Stop
        }
        catch {
            # Best-effort cleanup.
        }
    }
}

function Add-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Command,
        [Parameter(Mandatory = $true)][int]$ReturnCode,
        [Parameter(Mandatory = $true)][string]$Status,
        [string]$ErrorMessage = $null
    )
    $entry = [ordered]@{
        name = $Name
        command = $Command
        returncode = $ReturnCode
        status = $Status
        timestamp_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    }
    if (-not [string]::IsNullOrWhiteSpace($ErrorMessage)) {
        $entry.error = $ErrorMessage
    }
    $pipelineMeta.steps += $entry
}

function Add-InterruptedRun {
    param(
        [Parameter(Mandatory = $true)][string]$RunId,
        [Parameter(Mandatory = $true)][string]$Signature,
        [string]$Reason = $null
    )
    if ($null -eq $pipelineMeta.execution.interrupted_runs) {
        $pipelineMeta.execution.interrupted_runs = @()
    }
    $pipelineMeta.execution.interrupted_runs += [ordered]@{
        run_id = $RunId
        signature = $Signature
        reason = $Reason
        state = $pipelineMeta.progress.state
        recorded_at_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    }
}

function Enter-AiperfStabilityEnv {
    $desired = [ordered]@{
        AIPERF_SERVICE_REGISTRATION_TIMEOUT = [string]$AiperfServiceRegistrationTimeoutSec
        AIPERF_SERVICE_REGISTRATION_MAX_ATTEMPTS = [string]$AiperfServiceRegistrationMaxAttempts
        AIPERF_SERVICE_CONNECTION_PROBE_TIMEOUT = "180"
        AIPERF_SERVICE_PROFILE_START_TIMEOUT = "120"
        AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT = "300"
    }
    if (-not $script:AiperfEnvApplied) {
        foreach ($name in $desired.Keys) {
            $existing = Get-Item -Path ("Env:" + $name) -ErrorAction SilentlyContinue
            if ($existing) {
                $script:AiperfEnvOriginal[$name] = [string]$existing.Value
            }
            else {
                $script:AiperfEnvOriginal[$name] = $null
            }
        }
        $script:AiperfEnvApplied = $true
    }
    foreach ($name in $desired.Keys) {
        Set-Item -Path ("Env:" + $name) -Value ([string]$desired[$name])
    }
}

function Exit-AiperfStabilityEnv {
    if (-not $script:AiperfEnvApplied) {
        return
    }
    foreach ($name in $script:AiperfEnvOriginal.Keys) {
        $oldValue = $script:AiperfEnvOriginal[$name]
        if ($null -eq $oldValue) {
            Remove-Item -Path ("Env:" + $name) -ErrorAction SilentlyContinue
        }
        else {
            Set-Item -Path ("Env:" + $name) -Value ([string]$oldValue)
        }
    }
    $script:AiperfEnvApplied = $false
}

function Enter-MlcCompileTimeoutEnv {
    $existing = Get-Item -Path "Env:PERFLAB_MLC_COMPILE_TIMEOUT_SEC" -ErrorAction SilentlyContinue
    if ($existing) {
        $script:MlcCompileEnvOriginal = [string]$existing.Value
    }
    else {
        $script:MlcCompileEnvOriginal = $null
    }
    Set-Item -Path "Env:PERFLAB_MLC_COMPILE_TIMEOUT_SEC" -Value ([string]$MlcCompileTimeoutSec)
}

function Exit-MlcCompileTimeoutEnv {
    if ($null -eq $script:MlcCompileEnvOriginal) {
        Remove-Item -Path "Env:PERFLAB_MLC_COMPILE_TIMEOUT_SEC" -ErrorAction SilentlyContinue
    }
    else {
        Set-Item -Path "Env:PERFLAB_MLC_COMPILE_TIMEOUT_SEC" -Value ([string]$script:MlcCompileEnvOriginal)
    }
}

function Get-ListeningPortOwnerPids {
    param([int[]]$Ports)
    $ownerPids = @()
    if (-not $Ports -or @($Ports).Count -eq 0) {
        return @()
    }
    if (Get-Command -Name Get-NetTCPConnection -ErrorAction SilentlyContinue) {
        foreach ($port in $Ports) {
            try {
                $listeners = @(
                    Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue |
                        Select-Object -ExpandProperty OwningProcess
                )
                if ($listeners.Count -gt 0) {
                    $ownerPids += $listeners
                }
            }
            catch {
                # Fallback to netstat below.
            }
        }
    }
    if (@($ownerPids).Count -eq 0) {
        try {
            $netstat = & netstat -ano -p tcp 2>$null
            foreach ($line in $netstat) {
                if ($line -match "^\s*TCP\s+\S+:(\d+)\s+\S+\s+(LISTENING|侦听)\s+(\d+)\s*$") {
                    $linePort = [int]$matches[1]
                    if ($linePort -in $Ports) {
                        $ownerPids += [int]$matches[3]
                    }
                }
            }
        }
        catch {
            # Best-effort only.
        }
    }
    return @(
        $ownerPids |
            Where-Object { $_ -and ([int]$_) -gt 0 } |
            ForEach-Object { [int]$_ } |
            Sort-Object -Unique
    )
}

function Get-Win32ProcessSnapshotWithTimeout {
    param(
        [string]$Filter = "",
        [int]$TimeoutSeconds = 8
    )
    if ($DryRun) {
        return @()
    }
    if ($TimeoutSeconds -lt 1) {
        $TimeoutSeconds = 1
    }

    $job = $null
    try {
        $job = Start-Job -ScriptBlock {
            param($f)
            if ([string]::IsNullOrWhiteSpace([string]$f)) {
                Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
                    Select-Object ProcessId, Name, CommandLine
            }
            else {
                Get-CimInstance Win32_Process -Filter $f -ErrorAction SilentlyContinue |
                    Select-Object ProcessId, Name, CommandLine
            }
        } -ArgumentList $Filter

        if (-not (Wait-Job -Job $job -Timeout $TimeoutSeconds)) {
            Stop-Job -Job $job -Force -ErrorAction SilentlyContinue
            return @()
        }

        $received = @(Receive-Job -Job $job -ErrorAction SilentlyContinue)
        return @($received)
    }
    catch {
        return @()
    }
    finally {
        if ($null -ne $job) {
            Remove-Job -Job $job -Force -ErrorAction SilentlyContinue
        }
    }
}

function Stop-StaleAiperfProcesses {
    if ($DryRun) {
        return [ordered]@{
            stale_killed = 0
            port_owner_killed = 0
            port_owner_pids = @()
        }
    }
    $staleSnapshot = @(
        Get-Win32ProcessSnapshotWithTimeout -Filter "Name='python.exe'" -TimeoutSeconds 8
    )
    $stale = @(
        $staleSnapshot |
            Where-Object {
                $_.Name -eq "python.exe" -and
                $_.CommandLine -and
                (
                    $_.CommandLine -match "harness/tools/aiperf_windows_wrapper.py" -or
                    $_.CommandLine -match "-m\s+aiperf\s+profile" -or
                    $_.CommandLine -match "--multiprocessing-fork" -or
                    $_.CommandLine -match "multiprocessing\.spawn import spawn_main"
                )
            }
    )
    $killedStale = 0
    foreach ($proc in $stale) {
        try {
            Stop-Process -Id ([int]$proc.ProcessId) -Force -ErrorAction Stop
            $killedStale += 1
        }
        catch {
            # Best-effort cleanup.
        }
    }

    $portOwnerPids = @(Get-ListeningPortOwnerPids -Ports @(5557, 5564))
    $killedPortOwners = 0
    $killedPortOwnerPids = @()
    foreach ($portOwnerProcId in $portOwnerPids) {
        $proc = $null
        try {
            $proc = Get-Process -Id $portOwnerProcId -ErrorAction Stop
        }
        catch {
            continue
        }
        if (-not ($proc.ProcessName -ieq "python")) {
            continue
        }
        try {
            Stop-Process -Id $portOwnerProcId -Force -ErrorAction Stop
            $killedPortOwners += 1
            $killedPortOwnerPids += $portOwnerProcId
        }
        catch {
            # Best-effort cleanup.
        }
    }

    return [ordered]@{
        stale_killed = $killedStale
        port_owner_killed = $killedPortOwners
        port_owner_pids = @($killedPortOwnerPids | Sort-Object -Unique)
    }
}

function Stop-StaleManagedServerPortOwners {
    param(
        [Parameter(Mandatory = $true)][int]$Port
    )
    if ($DryRun) {
        return [ordered]@{
            target_port = $Port
            owner_pids = @()
            killed_count = 0
            killed_pids = @()
            blocked_count = 0
            blocked_pids = @()
        }
    }

    $ownerPids = @(Get-ListeningPortOwnerPids -Ports @($Port))
    $killed = @()
    $blocked = @()
    foreach ($ownerProcId in $ownerPids) {
        $proc = $null
        try {
            $proc = Get-Process -Id ([int]$ownerProcId) -ErrorAction Stop
        }
        catch {
            continue
        }

        $name = [string]$proc.ProcessName
        $killAllowed = $false
        if ($name -ieq "python") {
            $killAllowed = $true
        }
        elseif ($name -ieq "llama-server") {
            $killAllowed = $true
        }

        if (-not $killAllowed) {
            $blocked += [int]$ownerProcId
            continue
        }

        try {
            Stop-Process -Id ([int]$ownerProcId) -Force -ErrorAction Stop
            $killed += [int]$ownerProcId
        }
        catch {
            $blocked += [int]$ownerProcId
        }
    }

    return [ordered]@{
        target_port = $Port
        owner_pids = @($ownerPids)
        killed_count = @($killed).Count
        killed_pids = @($killed | Sort-Object -Unique)
        blocked_count = @($blocked | Sort-Object -Unique).Count
        blocked_pids = @($blocked | Sort-Object -Unique)
    }
}

function Write-PipelineMeta {
    if ($DryRun) {
        return
    }
    New-Item -ItemType Directory -Path (Split-Path -Parent $pipelineMetaPath) -Force | Out-Null
    ($pipelineMeta | ConvertTo-Json -Depth 50) | Set-Content -Path $pipelineMetaPath -Encoding UTF8
}

function Build-HourlyCheckpointPayload {
    param([string]$Reason = "interval")
    $backendSummary = @()
    foreach ($backend in $selectedBackends) {
        $rows = @($allRows | Where-Object { $_.backend -eq $backend })
        $successCount = @($rows | Where-Object { $_.status -eq "success" }).Count
        $failedCount = @($rows | Where-Object { $_.status -eq "failed" }).Count
        $dryRunCount = @($rows | Where-Object { $_.status -eq "dry-run" }).Count
        $backendSummary += [ordered]@{
            backend = $backend
            total = @($rows).Count
            success = $successCount
            failed = $failedCount
            dry_run = $dryRunCount
        }
    }

    $failedRows = @($allRows | Where-Object { $_.status -eq "failed" })
    $blockerTop = ""
    if (@($failedRows).Count -gt 0) {
        $grouped = @(
            $failedRows |
                Group-Object -Property blocker_signature |
                Sort-Object -Property Count -Descending |
                Select-Object -First 1
        )
        if (@($grouped).Count -gt 0 -and -not [string]::IsNullOrWhiteSpace([string]$grouped[0].Name)) {
            $blockerTop = [string]$grouped[0].Name
        }
    }

    $recentRuns = @(
        $allRows |
            Select-Object -Last 3 |
            ForEach-Object {
                [ordered]@{
                    backend = $_.backend
                    profile_id = $_.profile_id
                    stage = $_.stage
                    status = $_.status
                    attempt = $_.attempt
                    tps_mean = $_.tps_mean
                    ttft_p50_ms = $_.ttft_p50_ms
                    blocker_signature = $_.blocker_signature
                }
            }
    )

    return [ordered]@{
        generated_at_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
        reason = $Reason
        state = $pipelineMeta.progress.state
        progress = [ordered]@{
            stage_a = "{0}/{1}" -f $pipelineMeta.progress.stage_a_completed, $pipelineMeta.progress.stage_a_total
            stage_b = "{0}/{1}" -f $pipelineMeta.progress.stage_b_completed, $pipelineMeta.progress.stage_b_total
            completed_profiles = $pipelineMeta.progress.completed_profiles
            last_run_id = $pipelineMeta.progress.last_run_id
        }
        backend_summary = $backendSummary
        blocker_top = $blockerTop
        recent_runs = $recentRuns
    }
}

function Write-HourlyCheckpoint {
    param(
        [switch]$Force,
        [string]$Reason = "interval"
    )
    if ($DryRun) {
        return
    }
    $now = Get-Date
    if (-not $Force) {
        if ($null -eq $script:LastHourlyCheckpointAt) {
            return
        }
        if (($now - $script:LastHourlyCheckpointAt).TotalMinutes -lt $HourlyCheckpointMinutes) {
            return
        }
    }

    $payload = Build-HourlyCheckpointPayload -Reason $Reason
    $ts = $now.ToUniversalTime().ToString("yyyyMMdd_HHmmss")
    $mdPath = Join-Path $outRootResolved ("hourly_report_{0}.md" -f $ts)
    $jsonPath = Join-Path $outRootResolved ("hourly_report_{0}.json" -f $ts)

    ($payload | ConvertTo-Json -Depth 20) | Set-Content -Path $jsonPath -Encoding UTF8
    $lines = @()
    $lines += "# Hourly Checkpoint"
    $lines += ""
    $lines += ("generated_at_utc: {0}" -f $payload.generated_at_utc)
    $lines += ("reason: {0}" -f $payload.reason)
    $lines += ("state: {0}" -f $payload.state)
    $lines += ("stage_a: {0}" -f $payload.progress.stage_a)
    $lines += ("stage_b: {0}" -f $payload.progress.stage_b)
    $lines += ("completed_profiles: {0}" -f $payload.progress.completed_profiles)
    $lines += ("last_run_id: {0}" -f $payload.progress.last_run_id)
    $lines += ""
    $lines += "## Backend Summary"
    foreach ($item in @($payload.backend_summary)) {
        $lines += ("- {0}: total={1}, success={2}, failed={3}, dry_run={4}" -f $item.backend, $item.total, $item.success, $item.failed, $item.dry_run)
    }
    $lines += ""
    $lines += ("blocker_top: {0}" -f $payload.blocker_top)
    $lines += ""
    $lines += "## Recent Runs"
    foreach ($run in @($payload.recent_runs)) {
        $lines += ("- {0}/{1}/{2}: status={3}, attempt={4}, tps={5}, ttft={6}, blocker={7}" -f $run.backend, $run.profile_id, $run.stage, $run.status, $run.attempt, $run.tps_mean, $run.ttft_p50_ms, $run.blocker_signature)
    }
    Set-Content -Path $mdPath -Value ($lines -join "`n") -Encoding UTF8

    $entry = [ordered]@{
        generated_at_utc = $payload.generated_at_utc
        reason = $Reason
        state = $payload.state
        stage_a = $payload.progress.stage_a
        stage_b = $payload.progress.stage_b
        json_path = $jsonPath
        md_path = $mdPath
    }
    $pipelineMeta.execution.hourly_checkpoints += $entry
    $pipelineMeta.execution.last_checkpoint_utc = $payload.generated_at_utc
    $pipelineMeta.execution.long_run_auto_closed = [bool]$script:LongRunAutoClosed
    $script:LastHourlyCheckpointAt = $now
}

function Write-ProgressCheckpoint {
    param(
        [Parameter(Mandatory = $true)][string]$State,
        [string]$LastRunId = $null
    )
    if ($DryRun) {
        return
    }
    $pipelineMeta.progress.state = $State
    if (-not [string]::IsNullOrWhiteSpace($LastRunId)) {
        $pipelineMeta.progress.last_run_id = $LastRunId
    }
    $pipelineMeta.progress.last_update_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    if ($null -ne $allRows -and @($allRows).Count -gt 0) {
        Write-Leaderboard -Rows @($allRows)
    }
    Write-HourlyCheckpoint
    Write-PipelineMeta
}

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Executable,
        [string[]]$Arguments,
        [int]$TimeoutSeconds = 0,
        [switch]$AllowFailure
    )
    $commandText = Format-Command -Executable $Executable -Arguments $Arguments
    if ($DryRun) {
        Write-Host "[dry-run] $commandText"
        Add-Step -Name $Name -Command $commandText -ReturnCode 0 -Status "dry-run"
        return 0
    }

    if ($TimeoutSeconds -gt 0) {
        $stdoutPath = [System.IO.Path]::GetTempFileName()
        $stderrPath = [System.IO.Path]::GetTempFileName()
        try {
            $argsText = if ($Arguments) { (($Arguments | ForEach-Object { ConvertTo-WindowsArg -Arg $_ }) -join " ") } else { "" }
            if ([string]::IsNullOrWhiteSpace($argsText)) {
                $proc = Start-Process -FilePath $Executable -PassThru -NoNewWindow -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
            }
            else {
                $proc = Start-Process -FilePath $Executable -ArgumentList $argsText -PassThru -NoNewWindow -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
            }
            $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
            $nextHeartbeatAt = (Get-Date).AddSeconds(60)
            $finished = $false
            while ((Get-Date) -lt $deadline) {
                if ($proc.WaitForExit(1000)) {
                    $finished = $true
                    break
                }
                if ((Get-Date) -ge $nextHeartbeatAt) {
                    $pipelineMeta.progress.last_update_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
                    Write-PipelineMeta
                    $nextHeartbeatAt = (Get-Date).AddSeconds(60)
                }
            }
            if (-not $finished) {
                Stop-ProcessTree -RootProcessId $proc.Id
                Add-Step -Name $Name -Command $commandText -ReturnCode 124 -Status "timeout" -ErrorMessage "$Name timed out after $TimeoutSeconds sec"
                return 124
            }
            $rc = [int]$proc.ExitCode
            if (Test-Path $stdoutPath) {
                Get-Content -Path $stdoutPath -Encoding UTF8 -ErrorAction SilentlyContinue | Out-Host
            }
            if (Test-Path $stderrPath) {
                Get-Content -Path $stderrPath -Encoding UTF8 -ErrorAction SilentlyContinue | Out-Host
            }
        }
        finally {
            if (Test-Path $stdoutPath) { Remove-Item $stdoutPath -Force -ErrorAction SilentlyContinue }
            if (Test-Path $stderrPath) { Remove-Item $stderrPath -Force -ErrorAction SilentlyContinue }
        }
    }
    else {
        $oldEa = $ErrorActionPreference
        $hasNativePref = $null -ne (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue)
        if ($hasNativePref) {
            $oldNativePref = $PSNativeCommandUseErrorActionPreference
            $script:PSNativeCommandUseErrorActionPreference = $false
        }
        $ErrorActionPreference = "Continue"
        try {
            # Keep command output visible, but do not let it pollute function return value.
            & $Executable @Arguments | Out-Host
            $rc = if ($null -eq $LASTEXITCODE) { 0 } else { [int]$LASTEXITCODE }
        }
        finally {
            $ErrorActionPreference = $oldEa
            if ($hasNativePref) {
                $script:PSNativeCommandUseErrorActionPreference = $oldNativePref
            }
        }
    }
    Add-Step -Name $Name -Command $commandText -ReturnCode $rc -Status "executed"
    if ($rc -ne 0 -and -not $AllowFailure) {
        throw "$Name failed with exit code $rc"
    }
    return $rc
}

function Read-JsonFile {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-Content -Path $Path -Raw -Encoding UTF8 | ConvertFrom-Json)
}

function Read-JsonFileWithRetry {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [int]$MaxAttempts = 5,
        [int]$DelayMs = 200
    )
    $attempt = 0
    while ($attempt -lt $MaxAttempts) {
        $attempt += 1
        try {
            return Read-JsonFile -Path $Path
        }
        catch {
            if ($attempt -ge $MaxAttempts) {
                throw
            }
            Start-Sleep -Milliseconds $DelayMs
        }
    }
    throw "failed to read json with retry: $Path"
}

function Get-FieldValue {
    param(
        [Parameter(Mandatory = $false)]$Object,
        [Parameter(Mandatory = $true)][string]$Name
    )
    if ($null -eq $Object) {
        return $null
    }
    if ($Object -is [System.Collections.IDictionary]) {
        return $Object[$Name]
    }
    $prop = $Object.PSObject.Properties[$Name]
    if ($null -ne $prop) {
        return $prop.Value
    }
    return $null
}

function Test-ServerReachable {
    param([Parameter(Mandatory = $true)][string]$BaseUrl)
    $targets = @("$($BaseUrl.TrimEnd('/'))/health", "$($BaseUrl.TrimEnd('/'))/v1/models")
    foreach ($url in $targets) {
        try {
            $resp = Invoke-WebRequest -Method Get -Uri $url -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
            if ($resp.StatusCode -eq 200) {
                return $true
            }
        }
        catch {
            # Some Windows PowerShell builds can throw a null-reference even when the
            # endpoint is reachable. Fall back to curl status probing before failing.
            try {
                $curlCmd = Get-Command "curl.exe" -ErrorAction SilentlyContinue
                if ($null -ne $curlCmd) {
                    $httpCode = & curl.exe -sS -o NUL -w "%{http_code}" --connect-timeout 5 --max-time 8 $url 2>$null
                    if (($httpCode | Out-String).Trim() -eq "200") {
                        return $true
                    }
                }
            }
            catch {
                continue
            }
        }
    }
    return $false
}

function Resolve-RunFailureSignature {
    param(
        [Parameter(Mandatory = $true)][string]$Message,
        [string]$Backend = ""
    )
    $text = ($Message | Out-String).ToLower()
    if ($text -match "codec can't encode" -or $text -match "illegal multibyte sequence" -or $text -match "gbk") { return "tool_output_encoding_failure" }
    if ($text -match "unrecognized arguments:\s*--opt") { return "mlc_opt_unsupported" }
    if (
        $text -match "llama_kv_cache_type_unsupported" -or
        $text -match "invalid cache type" -or
        $text -match "unsupported cache type" -or
        $text -match "argument -ctk" -or
        $text -match "argument -ctv" -or
        $text -match "unrecognized arguments?:\s*-ctk" -or
        $text -match "unrecognized arguments?:\s*-ctv"
    ) { return "llama_kv_cache_type_unsupported" }
    if ($text -match "tool_run_meta") { return "tool_run_meta_invalid" }
    if ($text -match "tool_returncode_nonzero_with_zero_process_exit") { return "tool_returncode_nonzero_with_zero_process_exit" }
    if (
        (
            $text -match "error while attempting to bind on address" -or
            $text -match "only one usage of each socket address" -or
            $text -match "errno 10048" -or
            $text -match "\[errno 10048\]"
        ) -and (
            $text -match "127\.0\.0\.1:82\d{2}" -or
            $text -match "127\.0\.0\.1:83\d{2}" -or
            $text -match "830\d" -or
            $text -match "820\d"
        )
    ) { return "managed_server_port_conflict" }
    if (
        (
            $text -match "address in use" -or
            $text -match "already in use" -or
            $text -match "eaddrinuse"
        ) -and (
            $text -match "5557" -or
            $text -match "5564" -or
            $text -match "tcp://127\.0\.0\.1"
        )
    ) { return "aiperf_zmq_port_conflict" }
    if ($text -match "tcp://127\.0\.0\.1:5557" -or $text -match "tcp://127\.0\.0\.1:5564") { return "aiperf_zmq_port_conflict" }
    if (
        $text -match "died before registering" -or
        $text -match "registration timeout" -or
        $text -match "service registration timeout"
    ) { return "aiperf_service_registration_timeout" }
    if (
        $text -match "failed to perform operation 'configure profiling'" -or
        $text -match "lifecycleoperationerror"
    ) { return "aiperf_tool_failure" }
    if ($text -match "aiperf_no_progress_timeout") { return "aiperf_no_progress_timeout" }
    if ($text -match "ort_long_wait_low_progress") { return "ort_long_wait_low_progress" }
    if ($text -match "timed out after" -or $text -match "timeout after") { return "profile_timeout" }
    if ($text -match "context size has been exceeded" -or $text -match "failed to find free space in the kv cache") { return "context_window_exceeded" }
    if ($text -match "insufficient_samples_after_report") { return "insufficient_samples_after_report" }
    if ($text -match "metrics_missing_after_servebench") { return "metrics_missing_after_servebench" }
    if ($text -match "summary_missing_after_report") { return "summary_missing_after_report" }
    if ($text -match "tokenizer_local_cache_failure") { return "tokenizer_local_cache_failure" }
    if ($text -match "vuid-standalonespirv-memorysemantics-10866" -or $text -match "opcontrolbarrier") { return "mlc_vulkan_compile_vuid_10866" }
    if ($text -match "torch_oom_fallback_failed") { return "torch_oom_fallback_failed" }
    if ($text -match "torch_gpu_required_but_cpu_fallback" -or $text -match "torch_gpu_required_meta_missing") { return "torch_gpu_required_but_cpu_fallback" }
    if ($text -match "torch_model_download_failure") { return "torch_model_download_failure" }
    if ($text -match "compare_artifacts") { return "compare_artifacts_gate_failure" }
    if (
        ($text -match "gpu" -and $text -match "counter") -or
        $text -match "gpu telemetry" -or
        $text -match "gpu sampler" -or
        $text -match "failed to collect gpu" -or
        $text -match "telemetry collection failed" -or
        $text -match "pdh" -or
        $text -match "rocm-smi" -or
        $text -match "nvidia-smi"
    ) { return "gpu_sampler_unavailable" }
    if ($text -match "aiperf" -or $text -match "profile_export" -or $text -match "tool command failed" -or $text -match "start profiling" -or $text -match "configure profiling") { return "aiperf_tool_failure" }
    if ($text -match "health" -or $text -match "server" -or $text -match "connect" -or $text -match "timeout" -or $text -match "connectionrefusederror" -or $text -match "serverdisconnectederror" -or $text -match "clientconnectorerror") {
        if ($Backend -eq "torch_rocm") {
            return "torch_server_startup_failure"
        }
        return "server_startup_failure"
    }
    if ($text -match "adapt" -or $text -match "replay") { return "replay_adapt_failure" }
    if ($text -match "download" -or $text -match "huggingface") { return "torch_model_download_failure" }
    return "unknown_failure"
}

function Get-SystemFreeMemoryGb {
    if ($DryRun) {
        return [double]$MinSystemFreeMemoryGb
    }
    try {
        $counter = Get-Counter '\Memory\Available MBytes' -ErrorAction Stop
        $samples = @($counter.CounterSamples)
        if ($samples.Count -lt 1) {
            return -1.0
        }
        $freeMb = [double]$samples[0].CookedValue
        return [Math]::Round(($freeMb / 1024.0), 3)
    }
    catch {
        return -1.0
    }
}

function Assert-MinSystemFreeMemory {
    param(
        [Parameter(Mandatory = $true)][string]$Context
    )
    $freeGb = Get-SystemFreeMemoryGb
    $thresholdGb = [double]$MinSystemFreeMemoryGb
    $stepName = "memory-guard-$Context"
    $stepCommand = "ensure free RAM >= ${thresholdGb}GB"
    $stepStatus = if ($DryRun) { "dry-run" } else { "executed" }
    if ($freeGb -lt 0) {
        Add-Step -Name $stepName -Command $stepCommand -ReturnCode 0 -Status $stepStatus -ErrorMessage "skip memory guard: probe unavailable"
        return
    }
    if ($freeGb -lt $thresholdGb) {
        $err = "low free RAM: ${freeGb}GB < ${thresholdGb}GB at $Context"
        Add-Step -Name $stepName -Command $stepCommand -ReturnCode 9 -Status "executed" -ErrorMessage $err
        throw $err
    }
    Add-Step -Name $stepName -Command $stepCommand -ReturnCode 0 -Status $stepStatus -ErrorMessage "free_ram_gb=$freeGb"
}

function Test-GpuTelemetryFailure {
    param(
        [string]$Signature,
        [string]$ErrorText
    )
    if ([string]::IsNullOrWhiteSpace([string]$Signature) -and [string]::IsNullOrWhiteSpace([string]$ErrorText)) {
        return $false
    }
    $sig = [string]$Signature
    if ($sig -eq "gpu_sampler_unavailable") {
        return $true
    }
    $text = ([string]$ErrorText).ToLower()
    return (
        $text -match "gpu telemetry" -or
        $text -match "gpu sampler" -or
        $text -match "failed to collect gpu" -or
        $text -match "telemetry collection failed" -or
        $text -match "pdh" -or
        $text -match "rocm-smi" -or
        $text -match "nvidia-smi" -or
        ($text -match "gpu" -and $text -match "counter")
    )
}

function New-ProfileClone {
    param([Parameter(Mandatory = $true)]$Profile)
    $clone = [ordered]@{}
    if ($Profile -is [System.Collections.IDictionary]) {
        foreach ($key in $Profile.Keys) {
            $clone[[string]$key] = $Profile[$key]
        }
        return $clone
    }
    foreach ($prop in $Profile.PSObject.Properties) {
        $clone[[string]$prop.Name] = $prop.Value
    }
    return $clone
}

function Set-ProfileField {
    param(
        [Parameter(Mandatory = $true)]$Profile,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)]$Value
    )
    if ($Profile -is [System.Collections.IDictionary]) {
        $Profile[$Name] = $Value
        return
    }
    $prop = $Profile.PSObject.Properties[$Name]
    if ($null -ne $prop) {
        $prop.Value = $Value
        return
    }
    Add-Member -InputObject $Profile -NotePropertyName $Name -NotePropertyValue $Value -Force
}

function Invoke-AdaptiveRetryTuning {
    param(
        [Parameter(Mandatory = $true)][string]$RunId,
        [Parameter(Mandatory = $true)][string]$Backend,
        [Parameter(Mandatory = $true)][string]$FailureSignature,
        [string]$FailureText = "",
        [Parameter(Mandatory = $true)][int]$Attempt,
        [Parameter(Mandatory = $true)][int]$MaxAttempts,
        [Parameter(Mandatory = $true)][ref]$Concurrency,
        [Parameter(Mandatory = $true)][ref]$RequestCount,
        [Parameter(Mandatory = $true)][ref]$PromptTokens,
        [Parameter(Mandatory = $true)][ref]$OutputTokens
    )
    if ($Attempt -ge $MaxAttempts) {
        return $false
    }

    $text = ($FailureText | Out-String).ToLower()
    $oldConc = [int]$Concurrency.Value
    $oldReq = [int]$RequestCount.Value
    $oldPrompt = [int]$PromptTokens.Value
    $oldOutput = [int]$OutputTokens.Value
    $newConc = $oldConc
    $newReq = $oldReq
    $newPrompt = $oldPrompt
    $newOutput = $oldOutput
    $reason = $null

    if (
        $FailureSignature -eq "context_window_exceeded" -or
        $text -match "context size has been exceeded" -or
        $text -match "failed to find free space in the kv cache"
    ) {
        $newConc = [Math]::Max(1, $oldConc - 1)
        $newReq = [Math]::Max(4, [int][Math]::Floor($oldReq * 0.8))
        $newPrompt = [Math]::Max(128, [int][Math]::Floor($oldPrompt * 0.85))
        $newOutput = [Math]::Max(128, [int][Math]::Floor($oldOutput * 0.7))
        $reason = "context_window_exceeded_guard"
    }
    elseif ($FailureSignature -eq "insufficient_samples_after_report") {
        $newConc = [Math]::Max(1, $oldConc - 1)
        $newReq = [Math]::Max(4, [int][Math]::Floor($oldReq * 0.7))
        $newPrompt = [Math]::Max(128, [int][Math]::Floor($oldPrompt * 0.8))
        $newOutput = [Math]::Max(128, [int][Math]::Floor($oldOutput * 0.6))
        $reason = "insufficient_samples_guard_aggressive"
    }
    elseif (
        $FailureSignature -eq "tool_returncode_nonzero_with_zero_process_exit" -or
        $text -match "profile export has no valid metric rows"
    ) {
        $newConc = [Math]::Max(1, $oldConc - 1)
        $newReq = [Math]::Max(4, [int][Math]::Floor($oldReq * 0.65))
        $newPrompt = [Math]::Max(128, [int][Math]::Floor($oldPrompt * 0.75))
        $newOutput = [Math]::Max(128, [int][Math]::Floor($oldOutput * 0.55))
        $reason = "tool_returncode_guard"
    }
    elseif ($FailureSignature -eq "aiperf_no_progress_timeout" -or $FailureSignature -eq "profile_timeout") {
        $newConc = [Math]::Max(1, $oldConc - 1)
        $newReq = [Math]::Max(4, [int][Math]::Floor($oldReq * 0.75))
        $newPrompt = [Math]::Max(128, [int][Math]::Floor($oldPrompt * 0.85))
        $newOutput = [Math]::Max(128, [int][Math]::Floor($oldOutput * 0.75))
        $reason = "timeout_guard"
    }

    $changed = $false
    if ($newConc -ne $oldConc) { $Concurrency.Value = $newConc; $changed = $true }
    if ($newReq -ne $oldReq) { $RequestCount.Value = $newReq; $changed = $true }
    if ($newPrompt -ne $oldPrompt) { $PromptTokens.Value = $newPrompt; $changed = $true }
    if ($newOutput -ne $oldOutput) { $OutputTokens.Value = $newOutput; $changed = $true }

    if (-not $changed) {
        return $false
    }

    $msg = "reason=$reason backend=$Backend conc:$oldConc->$newConc req:$oldReq->$newReq prompt:$oldPrompt->$newPrompt output:$oldOutput->$newOutput"
    Add-Step -Name "adaptive-retry-$RunId-attempt-$Attempt" -Command "auto tune profile for retry" -ReturnCode 0 -Status "executed" -ErrorMessage $msg
    $script:AdaptiveProfileDowngradeCount = [int]$script:AdaptiveProfileDowngradeCount + 1
    Write-Host "[adaptive-retry] $RunId $msg"
    return $true
}

function Invoke-RetryTuning {
    param(
        [Parameter(Mandatory = $true)][string]$RunId,
        [Parameter(Mandatory = $true)][string]$Backend,
        [Parameter(Mandatory = $true)][string]$ProfileId,
        [Parameter(Mandatory = $true)][string]$FailureSignature,
        [string]$FailureText = "",
        [Parameter(Mandatory = $true)][int]$Attempt,
        [Parameter(Mandatory = $true)][int]$MaxAttempts,
        [Parameter(Mandatory = $true)][ref]$Concurrency,
        [Parameter(Mandatory = $true)][ref]$RequestCount,
        [Parameter(Mandatory = $true)][ref]$PromptTokens,
        [Parameter(Mandatory = $true)][ref]$OutputTokens
    )
    if ($Attempt -ge $MaxAttempts) {
        return $false
    }

    $failureTextLower = ($FailureText | Out-String).ToLower()
    $isInsufficientSampleFailure =
        ($FailureSignature -eq "insufficient_samples_after_report") -or
        ($failureTextLower -match "insufficient_samples_after_report")
    if ($Backend -eq "mlc" -and $Attempt -eq 1 -and ($ProfileId -eq "M3" -or $ProfileId -eq "M4") -and $isInsufficientSampleFailure) {
        $oldReq = [int]$RequestCount.Value
        $oldOutput = [int]$OutputTokens.Value
        $newReq = [Math]::Max(4, $oldReq - 2)
        $newOutput = [Math]::Max(128, $oldOutput - 64)
        $changed = $false
        if ($newReq -ne $oldReq) {
            $RequestCount.Value = $newReq
            $changed = $true
        }
        if ($newOutput -ne $oldOutput) {
            $OutputTokens.Value = $newOutput
            $changed = $true
        }
        if ($changed) {
            $msg = "reason=mlc_high_profile_second_attempt_guard backend=$Backend profile=$ProfileId req:$oldReq->$newReq output:$oldOutput->$newOutput"
            Add-Step -Name "adaptive-retry-$RunId-attempt-$Attempt" -Command "auto tune profile for retry" -ReturnCode 0 -Status "executed" -ErrorMessage $msg
            $script:AdaptiveProfileDowngradeCount = [int]$script:AdaptiveProfileDowngradeCount + 1
            Write-Host "[adaptive-retry] $RunId $msg"
            return $true
        }
    }

    return (Invoke-AdaptiveRetryTuning -RunId $RunId -Backend $Backend -FailureSignature $FailureSignature -FailureText $FailureText -Attempt $Attempt -MaxAttempts $MaxAttempts -Concurrency $Concurrency -RequestCount $RequestCount -PromptTokens $PromptTokens -OutputTokens $OutputTokens)
}

function Invoke-LlamaKvFallback {
    param(
        [Parameter(Mandatory = $true)][string]$RunId,
        [Parameter(Mandatory = $true)][string]$Backend,
        [Parameter(Mandatory = $true)][string]$FailureSignature,
        [string]$FailureText = "",
        [Parameter(Mandatory = $true)][int]$Attempt,
        [Parameter(Mandatory = $true)][int]$MaxAttempts,
        [Parameter(Mandatory = $true)][ref]$CacheTypeK,
        [Parameter(Mandatory = $true)][ref]$CacheTypeV
    )
    if ($Backend -ne "llama" -or $Attempt -ge $MaxAttempts) {
        return $false
    }

    $oldK = [string]$CacheTypeK.Value
    $oldV = [string]$CacheTypeV.Value
    $oldKLower = $oldK.ToLowerInvariant()
    $oldVLower = $oldV.ToLowerInvariant()
    if ($oldKLower -eq "f16" -and $oldVLower -eq "f16") {
        return $false
    }

    $text = ($FailureText | Out-String).ToLower()
    $unsupported = $FailureSignature -eq "llama_kv_cache_type_unsupported" -or
        $text -match "invalid cache type" -or
        $text -match "unsupported cache type" -or
        $text -match "argument -ctk" -or
        $text -match "argument -ctv" -or
        $text -match "unrecognized arguments?:\s*-ctk" -or
        $text -match "unrecognized arguments?:\s*-ctv"
    if (-not $unsupported) {
        return $false
    }

    $CacheTypeK.Value = "f16"
    $CacheTypeV.Value = "f16"
    $msg = "reason=llama_kv_cache_type_unsupported backend=llama ctk:$oldK->f16 ctv:$oldV->f16"
    Add-Step -Name "adaptive-retry-$RunId-attempt-$Attempt" -Command "fallback llama kv cache type to f16/f16" -ReturnCode 0 -Status "executed" -ErrorMessage $msg
    $script:LlamaKvFallbackCount = [int]$script:LlamaKvFallbackCount + 1
    $pipelineMeta.execution.llama_kv_fallback_count = [int]$script:LlamaKvFallbackCount
    Write-Host "[adaptive-retry] $RunId $msg"
    return $true
}

function Probe-MlcOptSupport {
    if ($DryRun) {
        Add-Step -Name "preflight-mlc-help" -Command "$PythonExe -m mlc_llm serve --help" -ReturnCode 0 -Status "dry-run"
        return $true
    }

    $stdoutPath = [System.IO.Path]::GetTempFileName()
    $stderrPath = [System.IO.Path]::GetTempFileName()
    try {
        $proc = Start-Process -FilePath $PythonExe -ArgumentList @("-m", "mlc_llm", "serve", "--help") -NoNewWindow -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
        $waitMs = [Math]::Min(([int64]$PreflightTimeoutSeconds * 1000), [int64][int]::MaxValue)
        $finished = $proc.WaitForExit([int]$waitMs)
        if (-not $finished) {
            Stop-ProcessTree -RootProcessId $proc.Id
            Add-Step -Name "preflight-mlc-help" -Command "$PythonExe -m mlc_llm serve --help" -ReturnCode 124 -Status "timeout" -ErrorMessage "preflight-mlc-help timed out after $PreflightTimeoutSeconds sec"
            throw "preflight-mlc-help timed out after $PreflightTimeoutSeconds sec"
        }
        $rc = [int]$proc.ExitCode
        $stdoutText = if (Test-Path $stdoutPath) { Get-Content -Path $stdoutPath -Raw -Encoding UTF8 } else { "" }
        $stderrText = if (Test-Path $stderrPath) { Get-Content -Path $stderrPath -Raw -Encoding UTF8 } else { "" }
        Add-Step -Name "preflight-mlc-help" -Command "$PythonExe -m mlc_llm serve --help" -ReturnCode $rc -Status "executed"
        if ($rc -ne 0) {
            throw "preflight-mlc-help failed with exit code $rc"
        }
        $all = ($stdoutText + "`n" + $stderrText).ToLower()
        return $all.Contains("--opt")
    }
    finally {
        if (Test-Path $stdoutPath) { Remove-Item $stdoutPath -Force -ErrorAction SilentlyContinue }
        if (Test-Path $stderrPath) { Remove-Item $stderrPath -Force -ErrorAction SilentlyContinue }
    }
}

function Get-RunFailureEvidence {
    param(
        [Parameter(Mandatory = $true)][string]$RunDir,
        [Parameter(Mandatory = $true)][string]$Backend
    )

    $rawDir = Join-Path $RunDir "raw_tool_output"
    $toolStderrPath = Join-Path $rawDir "tool_stderr.log"
    $serverStderrPath = Join-Path $rawDir "server_stderr.log"
    $aiperfLogPath = Join-Path $rawDir "aiperf/logs/aiperf.log"
    $toolMetaPath = Join-Path $rawDir "tool_run_meta.json"

    $toolStderr = if (Test-Path $toolStderrPath) { (Get-Content -Path $toolStderrPath -Tail 200 -Encoding UTF8 -ErrorAction SilentlyContinue) -join "`n" } else { "" }
    $serverStderr = if (Test-Path $serverStderrPath) { (Get-Content -Path $serverStderrPath -Tail 200 -Encoding UTF8 -ErrorAction SilentlyContinue) -join "`n" } else { "" }
    $aiperfLog = if (Test-Path $aiperfLogPath) { (Get-Content -Path $aiperfLogPath -Tail 200 -Encoding UTF8 -ErrorAction SilentlyContinue) -join "`n" } else { "" }
    $metaText = ""
    if (Test-Path $toolMetaPath) {
        try {
            $meta = Read-JsonFileWithRetry -Path $toolMetaPath -MaxAttempts 6 -DelayMs 250
            $metaRc = [string]$meta.returncode
            $metaTimeout = [string](Get-FieldValue -Object $meta -Name "timed_out")
            $metaText = "tool_run_meta.returncode=$metaRc timed_out=$metaTimeout"
        }
        catch {
            $metaText = "tool_run_meta_invalid"
        }
    }
    $combined = @($toolStderr, $serverStderr, $aiperfLog, $metaText) -join "`n"
    $sig = Resolve-RunFailureSignature -Message $combined -Backend $Backend
    $short = $combined.Trim()
    if ($short.Length -gt 1200) {
        $short = $short.Substring(0, 1200)
    }
    return [ordered]@{
        signature = $sig
        error = if ([string]::IsNullOrWhiteSpace($short)) { "servebench failed without logs (run_dir=$RunDir)" } else { $short }
    }
}

function Build-AiperfToolRunCmd {
    param(
        [Parameter(Mandatory = $true)][string]$RunnerPython,
        [Parameter(Mandatory = $true)][string]$ServerUrl,
        [Parameter(Mandatory = $true)][string]$TokenizerModel,
        [Parameter(Mandatory = $true)][int]$Concurrency,
        [Parameter(Mandatory = $true)][int]$RequestCount,
        [Parameter(Mandatory = $true)][int]$PromptTokens,
        [Parameter(Mandatory = $true)][int]$OutputTokens,
        [Parameter(Mandatory = $true)][int]$ConversationTurnMean,
        [Parameter(Mandatory = $true)][string]$EndpointType,
        [Parameter(Mandatory = $true)][bool]$EnableStreaming,
        [Parameter(Mandatory = $true)][int]$RequestTimeoutSeconds,
        [Parameter(Mandatory = $true)][bool]$DisableGpuTelemetry
    )
    $telemetryArgs = if ($DisableGpuTelemetry) { "--no-server-metrics --no-gpu-telemetry" } else { "--no-server-metrics" }
    $streamingArg = if ($EnableStreaming) { "--streaming" } else { "" }
    return (
        '{0} harness/tools/aiperf_windows_wrapper.py profile {1} --url {2} --workload {3} --endpoint-type {4} {5} --concurrency {6} --request-count {7} --prompt-input-tokens-mean {8} --output-tokens-mean {9} --conversation-turn-mean {10} --conversation-turn-stddev 0 --request-timeout-seconds {11} {12}' -f
        $RunnerPython, $TokenizerModel, $ServerUrl, $Workload, $EndpointType, $streamingArg, $Concurrency, $RequestCount, $PromptTokens, $OutputTokens, $ConversationTurnMean, $RequestTimeoutSeconds, $telemetryArgs
    )
}

function Build-ServebenchArgs {
    param(
        [Parameter(Mandatory = $true)][string]$Backend,
        [Parameter(Mandatory = $true)]$Profile,
        [Parameter(Mandatory = $true)][string]$RunDir,
        [Parameter(Mandatory = $true)][string]$Stage,
        [Parameter(Mandatory = $true)][bool]$DisableGpuTelemetry
    )

    $toolPython = $PythonExe
    $toolTokenizer = switch ($Backend) {
        "ort" { $script:ResolvedTokenizerRefOrt }
        "torch_rocm" { $TorchModelId }
        default { $script:ResolvedTokenizerRefMain }
    }

    $stageOffset = if ($Stage -eq "stage_b_recheck") { 100 } else { 0 }
    $managedPort = $null
    if ($Backend -eq "llama") {
        $managedPort = 8200 + $stageOffset + [int]$Profile.order
    }
    elseif ($Backend -eq "mlc") {
        $managedPort = 8300 + $stageOffset + [int]$Profile.order
    }
    $toolServerUrl = if ($null -ne $managedPort) {
        "http://127.0.0.1:$managedPort"
    }
    elseif ($Backend -eq "ort") {
        $OrtServerUrl
    }
    else {
        $torchServerUrl
    }
    $toolCmd = Build-AiperfToolRunCmd -RunnerPython $toolPython -ServerUrl $toolServerUrl -TokenizerModel $toolTokenizer -Concurrency ([int]$Profile.concurrency) -RequestCount ([int]$Profile.request_count) -PromptTokens ([int]$Profile.prompt_tokens_mean) -OutputTokens ([int]$Profile.output_tokens_mean) -ConversationTurnMean ([int]$Profile.conversation_turn_mean) -EndpointType $AiperfEndpointType -EnableStreaming ([bool]$AiperfEnableStreaming) -RequestTimeoutSeconds $AiperfRequestTimeoutSeconds -DisableGpuTelemetry $DisableGpuTelemetry
    $base = @(
        "harness/benchctl.py", "servebench",
        "--mode", "replay",
        "--workload", $Workload,
        "--tool", "aiperf",
        "--tool-run-cmd", $toolCmd,
        "--out", $RunDir,
        "--server-health-timeout-sec", "240",
        "--server-health-interval-sec", "1"
    )

    switch ($Backend) {
        "llama" {
            $port = [int]$managedPort
            $ctk = if ($null -ne $Profile.ctk -and -not [string]::IsNullOrWhiteSpace([string]$Profile.ctk)) { [string]$Profile.ctk } else { "f16" }
            $ctv = if ($null -ne $Profile.ctv -and -not [string]::IsNullOrWhiteSpace([string]$Profile.ctv)) { [string]$Profile.ctv } else { "f16" }
            $serverExtra = "-ngl $($Profile.ngl) -c $($Profile.c) -b $($Profile.b) -ub $($Profile.ub) -ctk $ctk -ctv $ctv"
            if (($null -ne $Profile.cb) -and [bool]$Profile.cb) {
                $serverExtra = ($serverExtra + " -cb").Trim()
            }
            if (($null -ne $Profile.kvu) -and [bool]$Profile.kvu) {
                $serverExtra = ($serverExtra + " -kvu").Trim()
            }
            if (-not [string]::IsNullOrWhiteSpace($LlamaTuningArgs)) {
                $serverExtra = ($serverExtra + " " + $LlamaTuningArgs).Trim()
            }
            return @(
                $base + @(
                    "--backend", "llama_cpp",
                    "--backend-version", "local",
                    "--server-mode", "managed",
                    "--server-bin", $script:LlamaServerBinEffective,
                    "--server-host", "127.0.0.1",
                    "--server-port", [string]$port,
                    "--model", $LlamaModelPath,
                    "--server-extra-args", $serverExtra
                )
            )
        }
        "mlc" {
            $port = [int]$managedPort
            $mlcArgs = @(
                "--backend", "mlc_llm",
                "--backend-version", "nightly",
                "--server-mode", "managed",
                "--server-bin", $PythonExe,
                "--server-bin-args", "-m mlc_llm serve",
                "--server-host", "127.0.0.1",
                "--server-port", [string]$port,
                "--model", $MlcModelRef,
                "--mlc-mode", [string]$Profile.mode,
                "--mlc-max-num-sequence", [string]$Profile.max_num_sequence,
                "--mlc-max-total-seq-length", [string]$Profile.max_total_seq_length,
                "--mlc-prefill-chunk-size", [string]$Profile.prefill_chunk_size,
                "--mlc-device", $MlcDevice
            )
            if ($script:MlcOptEffective -and $script:MlcOptEffective.Trim() -ne "") {
                $mlcArgs += @("--mlc-opt", $script:MlcOptEffective)
            }
            $prebuiltModelLib = [string]$script:MlcCompilePreflightByProfileId[$Profile.id]
            if (-not [string]::IsNullOrWhiteSpace($prebuiltModelLib)) {
                $mlcArgs += @("--mlc-model-lib", $prebuiltModelLib)
            }
            return @($base + $mlcArgs)
        }
        "ort" {
            return @(
                $base + @(
                    "--backend", "ort_dml",
                    "--backend-version", "attach",
                    "--server-mode", "attach",
                    "--server-url", $OrtServerUrl,
                    "--model", $OrtModelRef
                )
            )
        }
        "torch_rocm" {
            return @(
                $base + @(
                    "--backend", "torch_rocm",
                    "--backend-version", "rocm",
                    "--server-mode", "attach",
                    "--server-url", $torchServerUrl,
                    "--model", $TorchModelId
                )
            )
        }
    }

    throw "unsupported backend: $Backend"
}
function Capture-GpuUtilization {
    param(
        [Parameter(Mandatory = $true)][string]$Backend,
        [Parameter(Mandatory = $true)][string]$ProfileId,
        [Parameter(Mandatory = $true)][string]$Stage,
        [Parameter(Mandatory = $true)][string]$RunDir
    )
    $utilPath = Join-Path $gpuUtilRoot ("{0}_{1}_{2}.json" -f $Backend, $ProfileId, $Stage)
    if ($DryRun) {
        return [ordered]@{
            gpu_util_avg = 0.0
            sampler_status = "dry-run"
            sampler_error = $null
            artifact = $utilPath
        }
    }

    $avg = 0.0
    $samplerStatus = "ok"
    $samplerError = $null
    $samplerMode = "post_run_counter"
    $runtimeCounterPath = Join-Path $RunDir "raw_tool_output/gpu_runtime_counter.json"

    if (Test-Path $runtimeCounterPath) {
        try {
            $runtimeCounter = Read-JsonFile -Path $runtimeCounterPath
            $sampleCount = [int](Get-FieldValue -Object $runtimeCounter -Name "sample_count")
            if ($sampleCount -gt 0) {
                $preferred = [double](Get-FieldValue -Object $runtimeCounter -Name "preferred_util")
                $avg = [Math]::Round($preferred, 3)
                $samplerMode = "runtime_counter"
            }
            else {
                throw "runtime counter sample_count is zero"
            }
        }
        catch {
            $samplerStatus = "fallback_zero"
            $samplerError = "runtime counter parse failure: $($_.Exception.Message)"
        }
    }

    if ($samplerStatus -eq "ok" -and $samplerMode -eq "post_run_counter") {
        try {
            $counter = Get-Counter '\GPU Engine(*)\Utilization Percentage' -MaxSamples 6
            $allSamples = @($counter.CounterSamples | ForEach-Object { [double]$_.CookedValue } | Where-Object { $_ -ge 0 -and $_ -le 100 })
            if ($allSamples.Count -lt 1) {
                throw "GPU counter returned no usable samples"
            }
            $activeSamples = @($allSamples | Where-Object { $_ -gt 0.1 })
            if ($activeSamples.Count -gt 0) {
                $avg = [Math]::Round((($activeSamples | Measure-Object -Average).Average), 3)
            }
            else {
                $avg = [Math]::Round((($allSamples | Measure-Object -Average).Average), 3)
            }
        }
        catch {
            $avg = 0.0
            $samplerStatus = "fallback_zero"
            $samplerError = $_.Exception.Message
        }
    }

    $payload = [ordered]@{
        backend = $Backend
        profile_id = $ProfileId
        stage = $Stage
        run_dir = $RunDir
        gpu_util_avg = $avg
        sampler_status = $samplerStatus
        sampler_mode = $samplerMode
        sampler_error = $samplerError
        runtime_counter_artifact = if (Test-Path $runtimeCounterPath) { $runtimeCounterPath } else { $null }
        sampled_at_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    }
    New-Item -ItemType Directory -Path $gpuUtilRoot -Force | Out-Null
    ($payload | ConvertTo-Json -Depth 10) | Set-Content -Path $utilPath -Encoding UTF8

    return [ordered]@{
        gpu_util_avg = $avg
        sampler_status = $samplerStatus
        sampler_error = $samplerError
        artifact = $utilPath
    }
}

function Ensure-RunArtifacts {
    param([Parameter(Mandatory = $true)][string]$RunDir)
    $servebenchCheck = Ensure-ServebenchArtifacts -RunDir $RunDir
    if (-not $servebenchCheck.ok) {
        return $servebenchCheck
    }
    $summaryPath = Join-Path $RunDir "summary.csv"
    if (-not (Test-Path $summaryPath)) {
        return [ordered]@{ ok = $false; signature = "summary_missing_after_report"; error = "summary_missing_after_report: $summaryPath" }
    }
    return [ordered]@{ ok = $true; signature = $null; error = $null }
}

function Ensure-ServebenchArtifacts {
    param([Parameter(Mandatory = $true)][string]$RunDir)
    $toolMetaPath = Join-Path $RunDir "raw_tool_output/tool_run_meta.json"
    if (-not (Test-Path $toolMetaPath)) {
        return [ordered]@{ ok = $false; signature = "tool_run_meta_missing"; error = "tool_run_meta_missing: $toolMetaPath" }
    }

    try {
        $toolMeta = Read-JsonFile -Path $toolMetaPath
        $toolRc = [int]$toolMeta.returncode
    }
    catch {
        return [ordered]@{ ok = $false; signature = "tool_run_meta_invalid"; error = "tool_run_meta_invalid: $toolMetaPath" }
    }

    if ($toolRc -ne 0) {
        return [ordered]@{
            ok = $false
            signature = "tool_returncode_nonzero_with_zero_process_exit"
            error = "tool_returncode_nonzero_with_zero_process_exit: $toolMetaPath returncode=$toolRc"
        }
    }

    $metricsPath = Join-Path $RunDir "metrics.jsonl"
    if (-not (Test-Path $metricsPath)) {
        return [ordered]@{ ok = $false; signature = "metrics_missing_after_servebench"; error = "metrics_missing_after_servebench: $metricsPath" }
    }
    return [ordered]@{ ok = $true; signature = $null; error = $null }
}

function Read-SummaryMetrics {
    param([Parameter(Mandatory = $true)][string]$RunDir)
    $summaryPath = Join-Path $RunDir "summary.csv"
    $rows = @(Import-Csv -Path $summaryPath)
    if ($rows.Count -lt 1) {
        throw "summary.csv empty: $summaryPath"
    }
    $serving = @($rows | Where-Object { $_.track -eq "serving" })
    $row = if ($serving.Count -gt 0) { $serving[0] } else { $rows[0] }
    $sampleCount = 0
    try {
        $sampleCount = [int]$row.samples
    }
    catch {
        $sampleCount = 0
    }
    return [ordered]@{
        samples = $sampleCount
        ttft_p50_ms = [double]$row.ttft_p50_ms
        ttft_p95_ms = [double]$row.ttft_p95_ms
        itl_p50_ms = [double]$row.itl_p50_ms
        itl_p95_ms = [double]$row.itl_p95_ms
        tps_mean = [double]$row.tps_mean
        rps_mean = [double]$row.rps_mean
    }
}

function Get-MlcCompileKey {
    param([Parameter(Mandatory = $true)]$Profile)
    return "{0}|{1}|{2}|{3}|{4}" -f $MlcModelRef, $MlcDevice, $Profile.max_num_sequence, $Profile.max_total_seq_length, $Profile.prefill_chunk_size
}

function Get-ShortSha256 {
    param(
        [Parameter(Mandatory = $true)][string]$Text,
        [int]$Length = 16
    )
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($Text)
        $hash = $sha.ComputeHash($bytes)
        $hex = (($hash | ForEach-Object { $_.ToString("x2") }) -join "")
        if ($Length -gt 0 -and $Length -lt $hex.Length) {
            return $hex.Substring(0, $Length)
        }
        return $hex
    }
    finally {
        $sha.Dispose()
    }
}

function Invoke-MlcCompilePreflight {
    param([Parameter(Mandatory = $true)]$Profile)
    $compileKey = Get-MlcCompileKey -Profile $Profile
    if ($script:MlcCompileCacheByKey.ContainsKey($compileKey)) {
        $cached = $script:MlcCompileCacheByKey[$compileKey]
        if ($cached.ok -and -not [string]::IsNullOrWhiteSpace([string]$cached.model_lib)) {
            $script:MlcCompilePreflightByProfileId[$Profile.id] = [string]$cached.model_lib
        }
        return $cached
    }

    $hash = Get-ShortSha256 -Text $compileKey -Length 16
    $globalCacheDir = Join-Path $script:MlcCompileGlobalCacheRoot $hash
    $globalCacheModelLib = Join-Path $globalCacheDir "model_o0.dll"
    $globalCacheMeta = Join-Path $globalCacheDir "compile_preflight_meta.json"
    if ((Test-Path $globalCacheModelLib) -and (Test-Path $globalCacheMeta)) {
        $globalMeta = $null
        try {
            $globalMeta = Read-JsonFileWithRetry -Path $globalCacheMeta -MaxAttempts 4 -DelayMs 200
        }
        catch {
            $globalMeta = $null
        }
        $globalUsedBypass = $false
        if ($null -ne $globalMeta) {
            $globalUsedBypass = [bool](Get-FieldValue -Object $globalMeta -Name "used_validation_bypass")
        }
        $globalResult = [ordered]@{
            ok = $true
            compile_key = $compileKey
            compile_dir = $globalCacheDir
            model_lib = $globalCacheModelLib
            used_validation_bypass = $globalUsedBypass
            degraded_release = $globalUsedBypass
            compile_attempts = @()
            signature = $null
            error = $null
            cached = $true
            cache_scope = "global"
        }
        $script:MlcCompileCacheByKey[$compileKey] = $globalResult
        $script:MlcCompilePreflightByProfileId[$Profile.id] = [string]$globalCacheModelLib
        return $globalResult
    }
    $compileDir = Join-Path $script:MlcCompilePreflightDir $hash
    $outputPath = Join-Path $compileDir "model_o0.dll"
    $metaOut = Join-Path $compileDir "compile_preflight_meta.json"
    $probeScript = Join-Path $compileDir "mlc_compile_preflight_probe.py"

    if ($DryRun) {
        $dryResult = [ordered]@{
            ok = $true
            compile_key = $compileKey
            compile_dir = $compileDir
            model_lib = $outputPath
            used_validation_bypass = $false
            degraded_release = $false
            compile_attempts = @()
            signature = $null
            error = $null
            cached = $false
            cache_scope = "local"
        }
        $script:MlcCompileCacheByKey[$compileKey] = $dryResult
        $script:MlcCompilePreflightByProfileId[$Profile.id] = [string]$outputPath
        return $dryResult
    }

    New-Item -ItemType Directory -Path $compileDir -Force | Out-Null
    $probeText = @'
import argparse
import json
import sys
import traceback
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-exe", required=True)
    parser.add_argument("--model-ref", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--max-num-sequence", type=int, required=True)
    parser.add_argument("--max-total-seq-length", type=int, required=True)
    parser.add_argument("--prefill-chunk-size", type=int, required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--compile-dir", required=True)
    parser.add_argument("--meta-out", required=True)
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    harness_root = (repo_root / "harness").resolve()
    if str(harness_root) not in sys.path:
        sys.path.insert(0, str(harness_root))
    from harness import bench_ops as ops

    cmd = [
        args.python_exe,
        "-m",
        "mlc_llm",
        "compile",
        args.model_ref,
        "--device",
        args.device,
        "--opt",
        "O0",
        "--output",
        args.output_path,
    ]
    overrides = (
        f"max_batch_size={args.max_num_sequence};"
        f"context_window_size={args.max_total_seq_length};"
        f"prefill_chunk_size={args.prefill_chunk_size}"
    )
    if overrides:
        cmd.extend(["--overrides", overrides])

    payload: dict[str, object] = {
        "success": False,
        "command": cmd,
        "error_signature": "",
    }
    try:
        result = ops.run_mlc_compile_with_retry(
            command=cmd,
            compile_dir=Path(args.compile_dir),
        )
        payload.update(result)
        payload["success"] = bool(result.get("success", False))
        payload["effective_output_path"] = result.get("effective_output_path") or args.output_path
    except Exception as exc:  # pragma: no cover - probe error path
        message = str(exc)
        payload["success"] = False
        payload["error"] = message
        payload["traceback"] = traceback.format_exc()
        payload["error_signature"] = (
            "mlc_vulkan_compile_vuid_10866"
            if ops.is_mlc_jit_compile_failure(message)
            else "mlc_compile_preflight_failed"
        )

    out_path = Path(args.meta_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if payload.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
'@
    Set-Content -Path $probeScript -Value $probeText -Encoding UTF8

    $probeArgs = @(
        $probeScript,
        "--python-exe", $PythonExe,
        "--model-ref", $MlcModelRef,
        "--device", $MlcDevice,
        "--max-num-sequence", [string]$Profile.max_num_sequence,
        "--max-total-seq-length", [string]$Profile.max_total_seq_length,
        "--prefill-chunk-size", [string]$Profile.prefill_chunk_size,
        "--output-path", $outputPath,
        "--compile-dir", $compileDir,
        "--meta-out", $metaOut,
        "--repo-root", (Get-Location).Path
    )
    $rc = Invoke-Step -Name ("mlc-compile-preflight-{0}" -f $Profile.id) -Executable $PythonExe -Arguments $probeArgs -TimeoutSeconds $MlcCompileTimeoutSec -AllowFailure

    $meta = $null
    if (Test-Path $metaOut) {
        try {
            $meta = Read-JsonFileWithRetry -Path $metaOut -MaxAttempts 8 -DelayMs 300
        }
        catch {
            $meta = $null
        }
    }

    $ok = $false
    $sig = "mlc_compile_preflight_failed"
    $err = "mlc compile preflight failed for profile=$($Profile.id)"
    $modelLib = $outputPath
    $usedBypass = $false
    $degraded = $false
    $attempts = @()
    if ($null -ne $meta) {
        $ok = [bool](Get-FieldValue -Object $meta -Name "success")
        $modelLibCandidate = [string](Get-FieldValue -Object $meta -Name "effective_output_path")
        if (-not [string]::IsNullOrWhiteSpace($modelLibCandidate)) {
            $modelLib = $modelLibCandidate
        }
        $usedBypass = [bool](Get-FieldValue -Object $meta -Name "used_validation_bypass")
        $degraded = $usedBypass
        $attemptsRaw = Get-FieldValue -Object $meta -Name "attempts"
        if ($null -ne $attemptsRaw) {
            $attempts = @($attemptsRaw)
        }
        $metaSig = [string](Get-FieldValue -Object $meta -Name "error_signature")
        if (-not [string]::IsNullOrWhiteSpace($metaSig)) {
            $sig = $metaSig
        }
        $metaErr = [string](Get-FieldValue -Object $meta -Name "error")
        if (-not [string]::IsNullOrWhiteSpace($metaErr)) {
            $err = $metaErr
        }
    }

    if (-not $ok) {
        $stderrLogs = @(
            (Join-Path $compileDir "compile_o0_default_stderr.log")
            (Join-Path $compileDir "compile_o0_disable_shader_validation_stderr.log")
        )
        $tailText = ""
        foreach ($logPath in $stderrLogs) {
            if (Test-Path $logPath) {
                $tail = (Get-Content -Path $logPath -Tail 40 -Encoding UTF8 -ErrorAction SilentlyContinue) -join "`n"
                if (-not [string]::IsNullOrWhiteSpace($tail)) {
                    $tailText += "`n[$logPath]`n$tail"
                }
            }
        }
        $combined = "$err`n$tailText"
        $resolved = Resolve-RunFailureSignature -Message $combined -Backend "mlc"
        if (-not [string]::IsNullOrWhiteSpace([string]$resolved) -and $resolved -ne "unknown_failure") {
            $sig = $resolved
        }
        $err = $combined.Trim()
        if ([string]::IsNullOrWhiteSpace($err)) {
            $err = "mlc compile preflight failed for profile=$($Profile.id) (returncode=$rc)"
        }
    }

    $result = [ordered]@{
        ok = $ok
        compile_key = $compileKey
        compile_dir = $compileDir
        model_lib = $modelLib
        used_validation_bypass = $usedBypass
        degraded_release = $degraded
        compile_attempts = $attempts
        signature = if ($ok) { $null } else { $sig }
        error = if ($ok) { $null } else { $err }
        cached = $false
        cache_scope = "local"
    }
    if ($ok) {
        try {
            New-Item -ItemType Directory -Path $globalCacheDir -Force | Out-Null
            $copySource = $null
            if (Test-Path $outputPath) {
                $copySource = $outputPath
            }
            elseif (Test-Path $modelLib) {
                $copySource = $modelLib
            }
            if ($null -ne $copySource -and -not [string]::IsNullOrWhiteSpace([string]$copySource)) {
                Copy-Item -Path $copySource -Destination $globalCacheModelLib -Force
                $result.model_lib = $globalCacheModelLib
            }
            if (Test-Path $metaOut) {
                Copy-Item -Path $metaOut -Destination $globalCacheMeta -Force
            }
            $result.cache_scope = "global"
            $result.global_cache_model_lib = $globalCacheModelLib
        }
        catch {
            # Global cache copy is best-effort; keep local model lib on failure.
        }
    }
    $script:MlcCompileCacheByKey[$compileKey] = $result
    if ($ok) {
        $script:MlcCompilePreflightByProfileId[$Profile.id] = [string]$result.model_lib
    }
    return $result
}

function Get-AiperfProgressState {
    param([Parameter(Mandatory = $true)][string]$RunDir)

    $logPath = Join-Path $RunDir "raw_tool_output/aiperf/logs/aiperf.log"
    if (-not (Test-Path $logPath)) {
        return [ordered]@{
            has_log = $false
            log_path = $logPath
            log_last_write = $null
            sent = $null
            completed = $null
            in_flight = $null
            progress_key = ""
        }
    }

    $logItem = Get-Item -LiteralPath $logPath -ErrorAction SilentlyContinue
    $tailLines = @()
    try {
        $tailLines = @(Get-Content -LiteralPath $logPath -Tail 120 -Encoding UTF8 -ErrorAction SilentlyContinue)
    }
    catch {
        $tailLines = @()
    }

    $sent = $null
    $completed = $null
    $inFlight = $null
    for ($idx = $tailLines.Count - 1; $idx -ge 0; $idx--) {
        $line = [string]$tailLines[$idx]
        if ($line -match "sent\s*=\s*(\d+)\s*,\s*completed\s*=\s*(\d+)\s*,\s*in_flight\s*=\s*(\d+)") {
            $sent = [int]$matches[1]
            $completed = [int]$matches[2]
            $inFlight = [int]$matches[3]
            break
        }
    }

    $progressKey = ""
    if ($null -ne $sent -and $null -ne $completed -and $null -ne $inFlight) {
        $progressKey = "{0}/{1}/{2}" -f $sent, $completed, $inFlight
    }

    return [ordered]@{
        has_log = $true
        log_path = $logPath
        log_last_write = if ($null -ne $logItem) { $logItem.LastWriteTime } else { $null }
        sent = $sent
        completed = $completed
        in_flight = $inFlight
        progress_key = $progressKey
    }
}

function Invoke-ServebenchWithWatchdog {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$RunDir,
        [Parameter(Mandatory = $true)][int]$TimeoutSeconds,
        [Parameter(Mandatory = $true)][int]$NoProgressTimeoutSeconds
    )
    if ($DryRun) {
        return (Invoke-Step -Name $Name -Executable $PythonExe -Arguments $Arguments -TimeoutSeconds $TimeoutSeconds -AllowFailure)
    }

    $commandText = Format-Command -Executable $PythonExe -Arguments $Arguments
    $stdoutPath = [System.IO.Path]::GetTempFileName()
    $stderrPath = [System.IO.Path]::GetTempFileName()
    $gpuAllAvgSamples = New-Object 'System.Collections.Generic.List[double]'
    $gpuActiveAvgSamples = New-Object 'System.Collections.Generic.List[double]'
    $gpuPeakSamples = New-Object 'System.Collections.Generic.List[double]'
    try {
        $argsText = if ($Arguments) { (($Arguments | ForEach-Object { ConvertTo-WindowsArg -Arg $_ }) -join " ") } else { "" }
        if ([string]::IsNullOrWhiteSpace($argsText)) {
            $proc = Start-Process -FilePath $PythonExe -PassThru -NoNewWindow -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
        }
        else {
            $proc = Start-Process -FilePath $PythonExe -ArgumentList $argsText -PassThru -NoNewWindow -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
        }

        $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
        $nextHeartbeatAt = (Get-Date).AddSeconds(60)
        $nextProgressProbeAt = (Get-Date)
        $nextGpuProbeAt = (Get-Date)
        $lastProgressAt = Get-Date
        $lastProgressKey = ""
        $lastProgressHuman = "n/a"
        $hasSeenAiperfLog = $false
        $lastCompleted = -1
        $hasObservedCounters = $false
        $finished = $false

        while ((Get-Date) -lt $deadline) {
            if ($proc.WaitForExit(1000)) {
                $finished = $true
                break
            }

            $now = Get-Date
            if ($now -ge $nextGpuProbeAt) {
                try {
                    $counter = Get-Counter '\GPU Engine(*)\Utilization Percentage'
                    $engineValues = @($counter.CounterSamples | ForEach-Object { [double]$_.CookedValue } | Where-Object { $_ -ge 0 -and $_ -le 100 })
                    if ($engineValues.Count -gt 0) {
                        $allAvg = [double](($engineValues | Measure-Object -Average).Average)
                        $activeValues = @($engineValues | Where-Object { $_ -gt 0.1 })
                        $activeAvg = if ($activeValues.Count -gt 0) { [double](($activeValues | Measure-Object -Average).Average) } else { 0.0 }
                        $peak = [double](($engineValues | Measure-Object -Maximum).Maximum)
                        $gpuAllAvgSamples.Add($allAvg)
                        $gpuActiveAvgSamples.Add($activeAvg)
                        $gpuPeakSamples.Add($peak)
                    }
                }
                catch {
                    # Runtime GPU sampling is best-effort and should not fail profiling.
                }
                $nextGpuProbeAt = (Get-Date).AddSeconds(2)
            }

            if ($now -ge $nextProgressProbeAt) {
                $progress = Get-AiperfProgressState -RunDir $RunDir
                if ([bool]$progress.has_log) {
                    $hasSeenAiperfLog = $true
                    $sentValue = if ($null -ne $progress.sent) { [int]$progress.sent } else { $null }
                    $completedValue = if ($null -ne $progress.completed) { [int]$progress.completed } else { $null }
                    $inFlightValue = if ($null -ne $progress.in_flight) { [int]$progress.in_flight } else { $null }
                    if ($null -ne $sentValue -and $null -ne $completedValue -and $null -ne $inFlightValue) {
                        $hasObservedCounters = $true
                        $progressHuman = "sent=$sentValue, completed=$completedValue, in_flight=$inFlightValue"
                        if ($lastCompleted -lt 0) {
                            # First valid counter sample marks profiling as started.
                            $lastCompleted = $completedValue
                            $lastProgressAt = $now
                            $lastProgressKey = "{0}/{1}/{2}" -f $sentValue, $completedValue, $inFlightValue
                            $lastProgressHuman = $progressHuman
                        }
                        elseif ($completedValue -gt $lastCompleted) {
                            # Only treat completed growth as real progress.
                            $lastCompleted = $completedValue
                            $lastProgressAt = $now
                            $lastProgressKey = "{0}/{1}/{2}" -f $sentValue, $completedValue, $inFlightValue
                            $lastProgressHuman = $progressHuman
                        }
                        else {
                            $lastProgressHuman = "$progressHuman (completed_not_advancing)"
                        }
                    }
                }
                $nextProgressProbeAt = (Get-Date).AddSeconds(10)
            }

            if ($now -ge $nextHeartbeatAt) {
                $pipelineMeta.progress.last_update_utc = $now.ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
                Write-HourlyCheckpoint
                Write-PipelineMeta
                Write-Host ("[watchdog] {0}: {1}" -f $Name, $lastProgressHuman)
                $nextHeartbeatAt = (Get-Date).AddSeconds(60)
            }

            if ($hasSeenAiperfLog) {
                if (($now - $lastProgressAt).TotalSeconds -ge $NoProgressTimeoutSeconds) {
                    Stop-ProcessTree -RootProcessId $proc.Id
                    $timeoutReason = if ($hasObservedCounters) { "completed metric stalled" } else { "no valid progress counters observed" }
                    $timeoutMessage = "aiperf_no_progress_timeout after ${NoProgressTimeoutSeconds}s ($timeoutReason; $lastProgressHuman; log=$(Join-Path $RunDir 'raw_tool_output/aiperf/logs/aiperf.log'))"
                    Add-Step -Name $Name -Command $commandText -ReturnCode 125 -Status "timeout" -ErrorMessage $timeoutMessage
                    return 125
                }
            }
        }

        if (-not $finished) {
            Stop-ProcessTree -RootProcessId $proc.Id
            Add-Step -Name $Name -Command $commandText -ReturnCode 124 -Status "timeout" -ErrorMessage "$Name timed out after $TimeoutSeconds sec"
            return 124
        }

        $rc = [int]$proc.ExitCode
        if (Test-Path $stdoutPath) {
            Get-Content -Path $stdoutPath -Encoding UTF8 -ErrorAction SilentlyContinue | Out-Host
        }
        if (Test-Path $stderrPath) {
            Get-Content -Path $stderrPath -Encoding UTF8 -ErrorAction SilentlyContinue | Out-Host
        }

        Add-Step -Name $Name -Command $commandText -ReturnCode $rc -Status "executed"
        return $rc
    }
    finally {
        if ($gpuAllAvgSamples.Count -gt 0 -or $gpuActiveAvgSamples.Count -gt 0 -or $gpuPeakSamples.Count -gt 0) {
            try {
                $rawToolDir = Join-Path $RunDir "raw_tool_output"
                New-Item -ItemType Directory -Path $rawToolDir -Force | Out-Null
                $runtimeCounterPath = Join-Path $rawToolDir "gpu_runtime_counter.json"

                $allAvgMean = if ($gpuAllAvgSamples.Count -gt 0) { [Math]::Round(([double](($gpuAllAvgSamples | Measure-Object -Average).Average)), 3) } else { 0.0 }
                $activeAvgMean = if ($gpuActiveAvgSamples.Count -gt 0) { [Math]::Round(([double](($gpuActiveAvgSamples | Measure-Object -Average).Average)), 3) } else { 0.0 }
                $peakMax = if ($gpuPeakSamples.Count -gt 0) { [Math]::Round(([double](($gpuPeakSamples | Measure-Object -Maximum).Maximum)), 3) } else { 0.0 }
                $preferredUtil = if ($activeAvgMean -gt 0) { $activeAvgMean } else { $allAvgMean }

                $runtimePayload = [ordered]@{
                    sample_count = [int]([Math]::Max($gpuAllAvgSamples.Count, [Math]::Max($gpuActiveAvgSamples.Count, $gpuPeakSamples.Count)))
                    avg_all = $allAvgMean
                    avg_active = $activeAvgMean
                    peak = $peakMax
                    preferred_util = [Math]::Round([double]$preferredUtil, 3)
                    sampled_at_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
                }
                ($runtimePayload | ConvertTo-Json -Depth 10) | Set-Content -Path $runtimeCounterPath -Encoding UTF8
            }
            catch {
                # Best-effort artifact only.
            }
        }
        if (Test-Path $stdoutPath) { Remove-Item $stdoutPath -Force -ErrorAction SilentlyContinue }
        if (Test-Path $stderrPath) { Remove-Item $stderrPath -Force -ErrorAction SilentlyContinue }
    }
}

function Invoke-ProfileWithRetry {
    param(
        [Parameter(Mandatory = $true)][string]$Backend,
        [Parameter(Mandatory = $true)]$Profile,
        [Parameter(Mandatory = $true)][string]$Stage
    )

    $runId = "{0}-{1}-{2}" -f $Backend, $Profile.id, $Stage
    $runDir = Join-Path $runsRoot $runId
    $attempt = 0
    $lastSignature = $null
    $lastError = $null
    $baseProfile = New-ProfileClone -Profile $Profile
    $effectiveConcurrency = [int]$baseProfile.concurrency
    $effectiveRequestCount = [int]$baseProfile.request_count
    $effectivePromptTokens = [int]$baseProfile.prompt_tokens_mean
    $effectiveOutputTokens = [int]$baseProfile.output_tokens_mean
    $effectiveConversationTurn = [int]$baseProfile.conversation_turn_mean
    $baseCacheTypeK = Get-FieldValue -Object $baseProfile -Name "ctk"
    $baseCacheTypeV = Get-FieldValue -Object $baseProfile -Name "ctv"
    $baseCb = Get-FieldValue -Object $baseProfile -Name "cb"
    $baseKvu = Get-FieldValue -Object $baseProfile -Name "kvu"
    $effectiveCacheTypeK = if (-not [string]::IsNullOrWhiteSpace([string]$baseCacheTypeK)) { [string]$baseCacheTypeK } else { "f16" }
    $effectiveCacheTypeV = if (-not [string]::IsNullOrWhiteSpace([string]$baseCacheTypeV)) { [string]$baseCacheTypeV } else { "f16" }
    $effectiveCb = if ($null -ne $baseCb) { [bool]$baseCb } else { $false }
    $effectiveKvu = if ($null -ne $baseKvu) { [bool]$baseKvu } else { $false }
    $effectiveDisableGpuTelemetry = ($AiperfGpuTelemetryMode -eq "off")
    $telemetryDegraded = [bool]$effectiveDisableGpuTelemetry

    while ($attempt -lt $MaxAiperfAttempts) {
        $attempt += 1
        Assert-MinSystemFreeMemory -Context "$runId-attempt-$attempt"
        $profileForAttempt = New-ProfileClone -Profile $baseProfile
        $profileForAttempt.concurrency = $effectiveConcurrency
        $profileForAttempt.request_count = $effectiveRequestCount
        $profileForAttempt.prompt_tokens_mean = $effectivePromptTokens
        $profileForAttempt.output_tokens_mean = $effectiveOutputTokens
        $profileForAttempt.conversation_turn_mean = $effectiveConversationTurn
        Set-ProfileField -Profile $profileForAttempt -Name "ctk" -Value $effectiveCacheTypeK
        Set-ProfileField -Profile $profileForAttempt -Name "ctv" -Value $effectiveCacheTypeV
        Set-ProfileField -Profile $profileForAttempt -Name "cb" -Value $effectiveCb
        Set-ProfileField -Profile $profileForAttempt -Name "kvu" -Value $effectiveKvu

        if ($Backend -eq "torch_rocm") {
            $script:TorchServerProc = Start-TorchAttachServer
        }
        elseif (
            $TorchAttachLifecycle -eq "on_demand" -and
            $script:TorchServerManaged -and
            $null -ne $script:TorchServerProc
        ) {
            Stop-TorchAttachServer -Process $script:TorchServerProc
            $stopStatus = if ($DryRun) { "dry-run" } else { "executed" }
            Add-Step -Name "stop-torch-attach-server" -Command "stop managed attach server before non-torch backend=$Backend" -ReturnCode 0 -Status $stopStatus
        }

        Enter-AiperfStabilityEnv
        $cleanupResult = Stop-StaleAiperfProcesses
        $killedAiperf = [int]$cleanupResult.stale_killed
        $killedPortOwners = [int]$cleanupResult.port_owner_killed
        $killedPortOwnerPidText = if (@($cleanupResult.port_owner_pids).Count -gt 0) { (@($cleanupResult.port_owner_pids) -join ",") } else { "" }
        $cleanupStatus = if ($DryRun) { "dry-run" } else { "executed" }
        Add-Step -Name "cleanup-aiperf-$runId-attempt-$attempt" -Command "stop stale aiperf python processes and clear port owners on 5557/5564" -ReturnCode 0 -Status $cleanupStatus -ErrorMessage "killed_stale=$killedAiperf; killed_port_owners=$killedPortOwners; port_owner_pids=$killedPortOwnerPidText"

        if (-not $DryRun -and (Test-Path $runDir)) {
            Remove-Item -Path $runDir -Recurse -Force
        }

        $serveArgs = Build-ServebenchArgs -Backend $Backend -Profile $profileForAttempt -RunDir $runDir -Stage $Stage -DisableGpuTelemetry $effectiveDisableGpuTelemetry
        $managedPort = $null
        for ($i = 0; $i -lt ($serveArgs.Count - 1); $i++) {
            if ($serveArgs[$i] -eq "--server-port") {
                $managedPort = [int]$serveArgs[$i + 1]
                break
            }
        }
        if ($null -ne $managedPort) {
            $managedCleanup = Stop-StaleManagedServerPortOwners -Port ([int]$managedPort)
            $managedCleanupStatus = if ($DryRun) { "dry-run" } else { "executed" }
            $managedKilledText = if (@($managedCleanup.killed_pids).Count -gt 0) { (@($managedCleanup.killed_pids) -join ",") } else { "" }
            $managedBlockedText = if (@($managedCleanup.blocked_pids).Count -gt 0) { (@($managedCleanup.blocked_pids) -join ",") } else { "" }
            Add-Step -Name "cleanup-managed-port-$runId-attempt-$attempt" -Command "clear managed server listener on port $managedPort" -ReturnCode 0 -Status $managedCleanupStatus -ErrorMessage "owner_pids=$(@($managedCleanup.owner_pids).Count); killed=$($managedCleanup.killed_count); killed_pids=$managedKilledText; blocked=$($managedCleanup.blocked_count); blocked_pids=$managedBlockedText"
            if ([int]$managedCleanup.blocked_count -gt 0) {
                $lastSignature = "managed_server_port_conflict"
                $lastError = "managed port $managedPort occupied by non-killable pid(s): $managedBlockedText"
                if ($attempt -lt $MaxAiperfAttempts) {
                    Write-Host "retrying $runId after managed port conflict on $managedPort"
                    continue
                }
                break
            }
        }
        $noProgressTimeoutSec = if ($Backend -eq "ort") { $ortNoProgressTimeoutSec } else { $aiperfNoProgressTimeoutSec }
        $serveRc = Invoke-ServebenchWithWatchdog -Name "servebench-$runId-attempt-$attempt" -Arguments $serveArgs -RunDir $runDir -TimeoutSeconds $perProfileTimeoutSec -NoProgressTimeoutSeconds $noProgressTimeoutSec
        if ($serveRc -ne 0) {
            $evidence = Get-RunFailureEvidence -RunDir $runDir -Backend $Backend
            if ($serveRc -eq 124) {
                $lastSignature = "profile_timeout"
                $lastError = "servebench timed out after ${perProfileTimeoutSec}s; $($evidence.error)"
                $script:LongRunAutoClosed = $true
                $pipelineMeta.execution.long_run_auto_closed = $true
            }
            elseif ($serveRc -eq 125) {
                $script:NoProgressAbortCount = [int]$script:NoProgressAbortCount + 1
                $pipelineMeta.execution.no_progress_abort_count = [int]$script:NoProgressAbortCount
                if ($Backend -eq "ort") {
                    $lastSignature = "ort_long_wait_low_progress"
                    $lastError = "ort_long_wait_low_progress after ${noProgressTimeoutSec}s; $($evidence.error)"
                }
                else {
                    $lastSignature = "aiperf_no_progress_timeout"
                    $lastError = "aiperf_no_progress_timeout after ${noProgressTimeoutSec}s; $($evidence.error)"
                }
                Add-InterruptedRun -RunId $runId -Signature $lastSignature -Reason $lastError
                $script:LongRunAutoClosed = $true
                $pipelineMeta.execution.long_run_auto_closed = $true
            }
            else {
                $lastSignature = if ($evidence.signature) { [string]$evidence.signature } else { Resolve-RunFailureSignature -Message "servebench failed with code $serveRc" -Backend $Backend }
                $lastError = if ($evidence.error) { "servebench failed with code $serveRc; $($evidence.error)" } else { "servebench failed with code $serveRc" }
            }
            if ($attempt -lt $MaxAiperfAttempts) {
                if (
                    $AiperfGpuTelemetryMode -eq "adaptive" -and
                    -not $effectiveDisableGpuTelemetry -and
                    (Test-GpuTelemetryFailure -Signature $lastSignature -ErrorText $lastError)
                ) {
                    $effectiveDisableGpuTelemetry = $true
                    $telemetryDegraded = $true
                    $script:GpuTelemetryDegradedCount = [int]$script:GpuTelemetryDegradedCount + 1
                    $pipelineMeta.execution.gpu_telemetry_degraded_count = [int]$script:GpuTelemetryDegradedCount
                    Write-Host "retrying $runId with GPU telemetry disabled after failed attempt $attempt"
                    continue
                }
                if (Invoke-LlamaKvFallback -RunId $runId -Backend $Backend -FailureSignature ([string]$lastSignature) -FailureText ([string]$lastError) -Attempt $attempt -MaxAttempts $MaxAiperfAttempts -CacheTypeK ([ref]$effectiveCacheTypeK) -CacheTypeV ([ref]$effectiveCacheTypeV)) {
                    Write-Host "retrying $runId with llama kv fallback after failed attempt $attempt"
                    continue
                }
                [void](Invoke-RetryTuning -RunId $runId -Backend $Backend -ProfileId ([string]$Profile.id) -FailureSignature ([string]$lastSignature) -FailureText ([string]$lastError) -Attempt $attempt -MaxAttempts $MaxAiperfAttempts -Concurrency ([ref]$effectiveConcurrency) -RequestCount ([ref]$effectiveRequestCount) -PromptTokens ([ref]$effectivePromptTokens) -OutputTokens ([ref]$effectiveOutputTokens))
                Write-Host "retrying $runId after failed attempt $attempt"
                continue
            }
            break
        }

        if (-not $DryRun) {
            $servebenchCheck = Ensure-ServebenchArtifacts -RunDir $runDir
            if (-not $servebenchCheck.ok) {
                $evidence = Get-RunFailureEvidence -RunDir $runDir -Backend $Backend
                $checkSignature = [string]$servebenchCheck.signature
                $evidenceSignature = [string]$evidence.signature
                if (
                    [string]::IsNullOrWhiteSpace($checkSignature) -or
                    (
                        $checkSignature -eq "tool_returncode_nonzero_with_zero_process_exit" -and
                        -not [string]::IsNullOrWhiteSpace($evidenceSignature) -and
                        $evidenceSignature -ne "tool_returncode_nonzero_with_zero_process_exit"
                    )
                ) {
                    $lastSignature = $evidenceSignature
                }
                else {
                    $lastSignature = $checkSignature
                }
                $lastError = if ($evidence.error) {
                    "$($servebenchCheck.error); $($evidence.error)"
                }
                else {
                    [string]$servebenchCheck.error
                }
                if ($attempt -lt $MaxAiperfAttempts) {
                    if (
                        $AiperfGpuTelemetryMode -eq "adaptive" -and
                        -not $effectiveDisableGpuTelemetry -and
                        (Test-GpuTelemetryFailure -Signature $lastSignature -ErrorText $lastError)
                    ) {
                        $effectiveDisableGpuTelemetry = $true
                        $telemetryDegraded = $true
                        $script:GpuTelemetryDegradedCount = [int]$script:GpuTelemetryDegradedCount + 1
                        $pipelineMeta.execution.gpu_telemetry_degraded_count = [int]$script:GpuTelemetryDegradedCount
                        Write-Host "retrying $runId with GPU telemetry disabled after failed attempt $attempt"
                        continue
                    }
                    if (Invoke-LlamaKvFallback -RunId $runId -Backend $Backend -FailureSignature ([string]$lastSignature) -FailureText ([string]$lastError) -Attempt $attempt -MaxAttempts $MaxAiperfAttempts -CacheTypeK ([ref]$effectiveCacheTypeK) -CacheTypeV ([ref]$effectiveCacheTypeV)) {
                        Write-Host "retrying $runId with llama kv fallback after failed attempt $attempt"
                        continue
                    }
                    [void](Invoke-RetryTuning -RunId $runId -Backend $Backend -ProfileId ([string]$Profile.id) -FailureSignature ([string]$lastSignature) -FailureText ([string]$lastError) -Attempt $attempt -MaxAttempts $MaxAiperfAttempts -Concurrency ([ref]$effectiveConcurrency) -RequestCount ([ref]$effectiveRequestCount) -PromptTokens ([ref]$effectivePromptTokens) -OutputTokens ([ref]$effectiveOutputTokens))
                    Write-Host "retrying $runId after failed attempt $attempt"
                    continue
                }
                break
            }
        }

        $validateRc = Invoke-Step -Name "validate-$runId-attempt-$attempt" -Executable $PythonExe -Arguments @("harness/benchctl.py", "validate", "--input", (Join-Path $runDir "metrics.jsonl")) -AllowFailure
        if ($validateRc -ne 0) {
            $lastSignature = "replay_adapt_failure"
            $lastError = "validate failed with code $validateRc"
            if ($attempt -lt $MaxAiperfAttempts) {
                [void](Invoke-RetryTuning -RunId $runId -Backend $Backend -ProfileId ([string]$Profile.id) -FailureSignature ([string]$lastSignature) -FailureText ([string]$lastError) -Attempt $attempt -MaxAttempts $MaxAiperfAttempts -Concurrency ([ref]$effectiveConcurrency) -RequestCount ([ref]$effectiveRequestCount) -PromptTokens ([ref]$effectivePromptTokens) -OutputTokens ([ref]$effectiveOutputTokens))
                Write-Host "retrying $runId after failed attempt $attempt"
                continue
            }
            break
        }

        $reportRc = Invoke-Step -Name "report-$runId-attempt-$attempt" -Executable $PythonExe -Arguments @("harness/benchctl.py", "report", "--input", $runDir, "--out", $runDir) -AllowFailure
        if ($reportRc -ne 0) {
            $lastSignature = "replay_adapt_failure"
            $lastError = "report failed with code $reportRc"
            if ($attempt -lt $MaxAiperfAttempts) {
                [void](Invoke-RetryTuning -RunId $runId -Backend $Backend -ProfileId ([string]$Profile.id) -FailureSignature ([string]$lastSignature) -FailureText ([string]$lastError) -Attempt $attempt -MaxAttempts $MaxAiperfAttempts -Concurrency ([ref]$effectiveConcurrency) -RequestCount ([ref]$effectiveRequestCount) -PromptTokens ([ref]$effectivePromptTokens) -OutputTokens ([ref]$effectiveOutputTokens))
                Write-Host "retrying $runId after failed attempt $attempt"
                continue
            }
            break
        }

        if ($DryRun) {
            return [ordered]@{
                backend = $Backend
                profile_id = $Profile.id
                stage = $Stage
                status = "dry-run"
                attempt = $attempt
                run_dir = $runDir
                samples = [int]$profileForAttempt.request_count
                ttft_p50_ms = [double]$profileForAttempt.prompt_tokens_mean
                itl_p50_ms = 10.0
                tps_mean = [double]$profileForAttempt.concurrency
                rps_mean = [Math]::Round(([double]$profileForAttempt.concurrency / 10.0), 3)
                gpu_util_avg = 0.0
                gpu_util_artifact = Join-Path $gpuUtilRoot ("{0}_{1}_{2}.json" -f $Backend, $Profile.id, $Stage)
                telemetry_effective = if ($effectiveDisableGpuTelemetry) { "disabled" } else { "enabled" }
                telemetry_degraded = [bool]$telemetryDegraded
                sampler_status = "dry-run"
                blocker_signature = $null
                error = $null
                score = 0.0
            }
        }

        $artifactCheck = Ensure-RunArtifacts -RunDir $runDir
        if (-not $artifactCheck.ok) {
            $lastSignature = [string]$artifactCheck.signature
            $lastError = [string]$artifactCheck.error
            if ($attempt -lt $MaxAiperfAttempts) {
                if (Invoke-LlamaKvFallback -RunId $runId -Backend $Backend -FailureSignature ([string]$lastSignature) -FailureText ([string]$lastError) -Attempt $attempt -MaxAttempts $MaxAiperfAttempts -CacheTypeK ([ref]$effectiveCacheTypeK) -CacheTypeV ([ref]$effectiveCacheTypeV)) {
                    Write-Host "retrying $runId with llama kv fallback after failed attempt $attempt"
                    continue
                }
                [void](Invoke-RetryTuning -RunId $runId -Backend $Backend -ProfileId ([string]$Profile.id) -FailureSignature ([string]$lastSignature) -FailureText ([string]$lastError) -Attempt $attempt -MaxAttempts $MaxAiperfAttempts -Concurrency ([ref]$effectiveConcurrency) -RequestCount ([ref]$effectiveRequestCount) -PromptTokens ([ref]$effectivePromptTokens) -OutputTokens ([ref]$effectiveOutputTokens))
                Write-Host "retrying $runId after failed attempt $attempt"
                continue
            }
            break
        }

        try {
            $summary = Read-SummaryMetrics -RunDir $runDir
        }
        catch {
            $lastSignature = "summary_missing_after_report"
            $lastError = $_.Exception.Message
            if ($attempt -lt $MaxAiperfAttempts) {
                [void](Invoke-RetryTuning -RunId $runId -Backend $Backend -ProfileId ([string]$Profile.id) -FailureSignature ([string]$lastSignature) -FailureText ([string]$lastError) -Attempt $attempt -MaxAttempts $MaxAiperfAttempts -Concurrency ([ref]$effectiveConcurrency) -RequestCount ([ref]$effectiveRequestCount) -PromptTokens ([ref]$effectivePromptTokens) -OutputTokens ([ref]$effectiveOutputTokens))
                Write-Host "retrying $runId after failed attempt $attempt"
                continue
            }
            break
        }

        $expectedSamples = [int]$profileForAttempt.request_count
        $requiredSamples = [int][Math]::Ceiling([double]$expectedSamples * [double]$MinSampleRatio)
        $actualSamples = [int]$summary.samples
        if ($actualSamples -lt $requiredSamples) {
            $sampleEvidence = Get-RunFailureEvidence -RunDir $runDir -Backend $Backend
            $sampleSig = if ($sampleEvidence.signature) { [string]$sampleEvidence.signature } else { "" }
            if ($sampleSig -and $sampleSig -ne "unknown_failure") {
                $lastSignature = $sampleSig
            }
            else {
                $lastSignature = "insufficient_samples_after_report"
            }
            $lastError = "insufficient_samples_after_report: actual=$actualSamples required=$requiredSamples expected=$expectedSamples run=$runDir; evidence=$($sampleEvidence.error)"
            if ($attempt -lt $MaxAiperfAttempts) {
                [void](Invoke-RetryTuning -RunId $runId -Backend $Backend -ProfileId ([string]$Profile.id) -FailureSignature ([string]$lastSignature) -FailureText ([string]$lastError) -Attempt $attempt -MaxAttempts $MaxAiperfAttempts -Concurrency ([ref]$effectiveConcurrency) -RequestCount ([ref]$effectiveRequestCount) -PromptTokens ([ref]$effectivePromptTokens) -OutputTokens ([ref]$effectiveOutputTokens))
                Write-Host "retrying $runId after failed attempt $attempt (samples=$actualSamples required=$requiredSamples)"
                continue
            }
            break
        }

        $gpuSnapshot = Capture-GpuUtilization -Backend $Backend -ProfileId $Profile.id -Stage $Stage -RunDir $runDir
        if ($gpuSnapshot.sampler_status -ne "ok" -and -not $pipelineMeta.blocker_signature) {
            $pipelineMeta.blocker_signature = "gpu_sampler_unavailable"
        }

        return [ordered]@{
            backend = $Backend
            profile_id = $Profile.id
            stage = $Stage
            status = if ($DryRun) { "dry-run" } else { "success" }
            attempt = $attempt
            run_dir = $runDir
            samples = [int]$summary.samples
            ttft_p50_ms = [double]$summary.ttft_p50_ms
            itl_p50_ms = [double]$summary.itl_p50_ms
            tps_mean = [double]$summary.tps_mean
            rps_mean = [double]$summary.rps_mean
            gpu_util_avg = [double]$gpuSnapshot.gpu_util_avg
            gpu_util_artifact = [string]$gpuSnapshot.artifact
            telemetry_effective = if ($effectiveDisableGpuTelemetry) { "disabled" } else { "enabled" }
            telemetry_degraded = [bool]$telemetryDegraded
            sampler_status = [string]$gpuSnapshot.sampler_status
            blocker_signature = $null
            error = $null
            score = 0.0
        }
    }

    if (Test-GpuTelemetryFailure -Signature $lastSignature -ErrorText $lastError) {
        $script:GpuTelemetryHardFailCount = [int]$script:GpuTelemetryHardFailCount + 1
        $pipelineMeta.execution.gpu_telemetry_hard_fail_count = [int]$script:GpuTelemetryHardFailCount
    }

    return [ordered]@{
        backend = $Backend
        profile_id = $Profile.id
        stage = $Stage
        status = "failed"
        attempt = $attempt
        run_dir = $runDir
        samples = 0
        ttft_p50_ms = 0.0
        itl_p50_ms = 0.0
        tps_mean = 0.0
        rps_mean = 0.0
        gpu_util_avg = 0.0
        gpu_util_artifact = $null
        telemetry_effective = if ($effectiveDisableGpuTelemetry) { "disabled" } else { "enabled" }
        telemetry_degraded = [bool]$telemetryDegraded
        sampler_status = "n/a"
        blocker_signature = if ($lastSignature) { $lastSignature } else { "unknown_failure" }
        error = if ($lastError) { $lastError } else { "profile failed without explicit error" }
        score = 0.0
    }
}

function Rank-BackendRows {
    param(
        [Parameter(Mandatory = $true)][object[]]$Rows,
        [Parameter(Mandatory = $true)][string]$Backend,
        [Parameter(Mandatory = $true)][string]$Stage
    )

    $okStatuses = if ($DryRun) { @("success", "dry-run") } else { @("success") }
    $subset = @($Rows | Where-Object { $_.backend -eq $Backend -and $_.stage -eq $Stage -and $okStatuses -contains $_.status })
    if ($subset.Count -eq 0) {
        return @()
    }

    $maxGpu = [double](($subset | Measure-Object -Property gpu_util_avg -Maximum).Maximum)
    $maxTps = [double](($subset | Measure-Object -Property tps_mean -Maximum).Maximum)

    foreach ($row in $subset) {
        $gpuNorm = if ($maxGpu -gt 0) { [double]$row.gpu_util_avg / $maxGpu } else { 0.0 }
        $tpsNorm = if ($maxTps -gt 0) { [double]$row.tps_mean / $maxTps } else { 0.0 }
        $row.score = 0.6 * $gpuNorm + 0.4 * $tpsNorm
    }

    return @(
        $subset |
            Sort-Object @{Expression = { [int][bool](Get-FieldValue -Object $_ -Name "telemetry_degraded") }; Descending = $false }, @{Expression = { $_.tps_mean }; Descending = $true }, @{Expression = { $_.score }; Descending = $true }, @{Expression = { $_.rps_mean }; Descending = $true }, @{Expression = { $_.ttft_p50_ms }; Descending = $false }
    )
}

function Write-Leaderboard {
    param([object[]]$Rows = @())
    if ($DryRun) {
        return
    }
    New-Item -ItemType Directory -Path (Split-Path -Parent $leaderboardPath) -Force | Out-Null
    $records = @()
    foreach ($row in $Rows) {
        $records += [pscustomobject]@{
            backend = $row.backend
            profile_id = $row.profile_id
            stage = $row.stage
            status = $row.status
            attempt = $row.attempt
            samples = $row.samples
            ttft_p50_ms = $row.ttft_p50_ms
            itl_p50_ms = $row.itl_p50_ms
            tps_mean = $row.tps_mean
            rps_mean = $row.rps_mean
            gpu_util_avg = $row.gpu_util_avg
            score = $row.score
            telemetry_effective = [string](Get-FieldValue -Object $row -Name "telemetry_effective")
            telemetry_degraded = [bool](Get-FieldValue -Object $row -Name "telemetry_degraded")
            sampler_status = [string](Get-FieldValue -Object $row -Name "sampler_status")
            blocker_signature = $row.blocker_signature
            error = $row.error
            run_dir = $row.run_dir
            gpu_util_artifact = $row.gpu_util_artifact
        }
    }
    $records | Export-Csv -Path $leaderboardPath -NoTypeInformation -Encoding UTF8
}
function Invoke-ComparePair {
    param(
        [Parameter(Mandatory = $true)][string]$PairName,
        [Parameter(Mandatory = $true)][string]$BaselineRun,
        [Parameter(Mandatory = $true)][string]$CandidateRun,
        [Parameter(Mandatory = $true)][string]$BaselineLabel,
        [Parameter(Mandatory = $true)][string]$CandidateLabel
    )

    $pairDir = Join-Path $compareRoot $PairName
    $args = @(
        "harness/benchctl.py", "compare-runs",
        "--baseline-run", $BaselineRun,
        "--candidate-run", $CandidateRun,
        "--baseline-label", $BaselineLabel,
        "--candidate-label", $CandidateLabel,
        "--out", $pairDir
    )
    $null = Invoke-Step -Name "compare-$PairName" -Executable $PythonExe -Arguments $args

    if ($DryRun) {
        return [pscustomobject]@{
            winner = "dry-run"
            vote_counts = [pscustomobject]@{
                candidate = 0
                baseline = 0
                tie = 0
            }
        }
    }

    $summaryPath = Join-Path $pairDir "comparison_summary.csv"
    $reportPath = Join-Path $pairDir "comparison_report.md"
    $metaPath = Join-Path $pairDir "compare_meta.json"
    if (-not (Test-Path $summaryPath) -or -not (Test-Path $reportPath) -or -not (Test-Path $metaPath)) {
        throw "compare_artifacts_gate_failure: $PairName missing compare triplets"
    }

    $meta = Read-JsonFile -Path $metaPath
    return [pscustomobject]@{
        winner = [string]$meta.overall.winner
        vote_counts = $meta.overall.vote_counts
    }
}

function Build-MatrixReport {
    param([Parameter(Mandatory = $true)][hashtable]$PairResults)
    if ($DryRun) {
        return
    }

    $lines = @()
    $lines += "# GPU Util Uplift Matrix"
    $lines += ""
    $lines += "Generated at: $((Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'))"
    $lines += ""
    $lines += "Mainline backends: $($mainlineBackendOrder -join ', ')"
    if ($experimentalBackendOrder.Count -gt 0) {
        $lines += "Experimental backends: $($experimentalBackendOrder -join ', ')"
    }
    else {
        $lines += "Experimental backends: (none)"
    }
    $lines += ""
    $lines += "| Pair | Winner | Candidate Votes | Baseline Votes | Tie Votes |"
    $lines += "| --- | --- | ---: | ---: | ---: |"
    foreach ($pair in @($PairResults.Keys | Sort-Object)) {
        $p = $PairResults[$pair]
        $winner = Get-FieldValue -Object $p -Name "winner"
        $voteCounts = Get-FieldValue -Object $p -Name "vote_counts"
        $candidateVotes = Get-FieldValue -Object $voteCounts -Name "candidate"
        $baselineVotes = Get-FieldValue -Object $voteCounts -Name "baseline"
        $tieVotes = Get-FieldValue -Object $voteCounts -Name "tie"
        $lines += "| $pair | $winner | $candidateVotes | $baselineVotes | $tieVotes |"
    }
    Set-Content -Path $matrixReportPath -Value ($lines -join "`n") -Encoding UTF8
}

function Get-TorchAttachServerHealth {
    param([Parameter(Mandatory = $true)][string]$BaseUrl)
    try {
        $health = Invoke-RestMethod -Uri ($BaseUrl.TrimEnd("/") + "/health") -Method GET -TimeoutSec 5
    }
    catch {
        return $null
    }
    if ($null -eq $health) {
        return $null
    }
    return [ordered]@{
        selected_model_id = if ($null -eq $health.selected_model_id) { $null } else { [string]$health.selected_model_id }
        fallback_triggered = if ($null -eq $health.fallback_triggered) { $null } else { [bool]$health.fallback_triggered }
        runtime_device_fallback = if ($null -eq $health.runtime_device_fallback) { $null } else { [bool]$health.runtime_device_fallback }
        device = if ($null -eq $health.device) { $null } else { [string]$health.device }
    }
}

function Start-TorchAttachServer {
    if ($DryRun) {
        Add-Step -Name "start-torch-attach-server" -Command "dry-run torch attach startup" -ReturnCode 0 -Status "dry-run"
        $script:TorchServerManaged = $false
        return $null
    }

    if ($script:TorchServerManaged -and $null -ne $script:TorchServerProc) {
        try {
            if (-not $script:TorchServerProc.HasExited -and (Test-ServerReachable -BaseUrl $torchServerUrl)) {
                Add-Step -Name "start-torch-attach-server" -Command "reuse managed attach server at $torchServerUrl" -ReturnCode 0 -Status "executed"
                return $script:TorchServerProc
            }
        }
        catch {
            # Best-effort fast-path; fall through to normal startup flow.
        }
    }

    if (Test-ServerReachable -BaseUrl $torchServerUrl) {
        Add-Step -Name "start-torch-attach-server" -Command "reuse existing attach server at $torchServerUrl" -ReturnCode 0 -Status "executed"
        $script:TorchServerManaged = $false
        $healthMeta = Get-TorchAttachServerHealth -BaseUrl $torchServerUrl
        if ($null -ne $healthMeta) {
            $pipelineMeta.torch_server.selected_model_id = $healthMeta.selected_model_id
            $pipelineMeta.torch_server.fallback_triggered = $healthMeta.fallback_triggered
            $pipelineMeta.torch_server.fallback_reason_signature = $null
            $pipelineMeta.torch_server.device = $healthMeta.device
            $pipelineMeta.torch_server.runtime_device_fallback = $healthMeta.runtime_device_fallback
            $pipelineMeta.torch_server.cache_root = $null
            $pipelineMeta.torch_server.operator_optimizations = $null
        }
        try {
            $models = Invoke-RestMethod -Uri ($torchServerUrl.TrimEnd("/") + "/v1/models") -Method GET -TimeoutSec 5
            if ($models -and $models.data -and $models.data.Count -gt 0) {
                $pipelineMeta.torch_server.selected_model_id = [string]$models.data[0].id
            }
        }
        catch {
            # Best-effort metadata only.
        }
        if ($TorchRequireGpu) {
            if ($null -eq $healthMeta) {
                throw "torch_gpu_required_meta_missing: $torchServerUrl/health"
            }
            $device = [string]$healthMeta.device
            $runtimeFallback = [bool]$healthMeta.runtime_device_fallback
            $fallbackTriggered = [bool]$healthMeta.fallback_triggered
            if ($device -ne "cuda" -or $runtimeFallback -or $fallbackTriggered) {
                $pipelineMeta.execution.torch_gpu_verified = $false
                $pipelineMeta.torch_server.gpu_required_pass = $false
                throw "torch_gpu_required_but_cpu_fallback: device=$device runtime_device_fallback=$runtimeFallback fallback_triggered=$fallbackTriggered reason=$($pipelineMeta.torch_server.fallback_reason_signature)"
            }
            $pipelineMeta.execution.torch_gpu_verified = $true
            $pipelineMeta.torch_server.gpu_required_pass = $true
        }
        else {
            $pipelineMeta.execution.torch_gpu_verified = $null
            $pipelineMeta.torch_server.gpu_required_pass = $null
        }
        return $null
    }

    New-Item -ItemType Directory -Path $torchServerLogDir -Force | Out-Null
    if (Test-Path $torchServerStdoutPath) { Remove-Item $torchServerStdoutPath -Force }
    if (Test-Path $torchServerStderrPath) { Remove-Item $torchServerStderrPath -Force }
    if (Test-Path $torchServerMetaPath) { Remove-Item $torchServerMetaPath -Force }

    # Ensure HuggingFace cache roots are inherited by the child process before import-time config.
    $hfHome = Join-Path $TorchModelRoot "hf_cache"
    $hfHubCache = Join-Path $hfHome "hub"
    $hfTransformersCache = Join-Path $hfHome "transformers"
    Assert-DDrivePath -Name "HF_HOME" -PathLike $hfHome
    Assert-DDrivePath -Name "HUGGINGFACE_HUB_CACHE" -PathLike $hfHubCache
    Assert-DDrivePath -Name "TRANSFORMERS_CACHE" -PathLike $hfTransformersCache
    New-Item -ItemType Directory -Path $hfHubCache, $hfTransformersCache -Force | Out-Null
    $env:HF_HOME = $hfHome
    $env:HUGGINGFACE_HUB_CACHE = $hfHubCache
    $env:HF_HUB_CACHE = $hfHubCache
    $env:TRANSFORMERS_CACHE = $hfTransformersCache
    $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
    $env:PYTHONNOUSERSITE = "1"
    if (-not [string]::IsNullOrWhiteSpace($torchTunableOpResultsFileResolved)) {
        $tunableParent = Split-Path -Parent $torchTunableOpResultsFileResolved
        if (-not [string]::IsNullOrWhiteSpace($tunableParent)) {
            New-Item -ItemType Directory -Path $tunableParent -Force | Out-Null
        }
    }

    $hipForceDevKernargArg = if ([bool]$TorchHipForceDevKernargBool) { "1" } else { "0" }
    $args = @(
        "-s",
        "scripts/start_torch_rocm_attach_server.py",
        "--model-id", $TorchModelId,
        "--fallback-model-id", $TorchFallbackModelId,
        "--model-root", $TorchModelRoot,
        "--host", "127.0.0.1",
        "--port", [string]$TorchServerPort,
        "--attn-implementation", $TorchAttnImplementation,
        "--sdpa-kernel-profile", $TorchSdpaKernelProfile,
        "--hip-force-dev-kernarg", $hipForceDevKernargArg,
        "--preferred-blas-backend", $TorchPreferredBlasBackend,
        "--preferred-rocm-fa-backend", $TorchPreferredRocmFaBackend,
        "--torch-compile-mode", $TorchCompileMode,
        "--meta-out", $torchServerMetaPath,
        "--no-allow-auto-download"
    )
    if ([bool]$TorchEnableTunableOp) {
        $args += "--enable-tunableop"
    }
    else {
        $args += "--disable-tunableop"
    }
    if ([bool]$TorchTunableOpTuning) {
        $args += "--tunableop-tuning"
    }
    else {
        $args += "--no-tunableop-tuning"
    }
    if (-not [string]::IsNullOrWhiteSpace($torchTunableOpResultsFileResolved)) {
        $args += @("--tunableop-results-file", $torchTunableOpResultsFileResolved)
    }
    if ($TorchEnableCompile) {
        $args += "--enable-torch-compile"
    }
    else {
        $args += "--disable-torch-compile"
    }

    $argLine = ($args | ForEach-Object { ConvertTo-WindowsArg -Arg $_ }) -join " "
    $proc = Start-Process -FilePath $TorchPythonExe -ArgumentList $argLine -PassThru -NoNewWindow -RedirectStandardOutput $torchServerStdoutPath -RedirectStandardError $torchServerStderrPath
    Add-Step -Name "start-torch-attach-server" -Command (Format-Command -Executable $TorchPythonExe -Arguments $args) -ReturnCode 0 -Status "executed"

    $deadline = (Get-Date).AddSeconds(600)
    while ((Get-Date) -lt $deadline) {
        if ($proc.HasExited) {
            $stderrTail = if (Test-Path $torchServerStderrPath) { (Get-Content $torchServerStderrPath -Tail 80 -ErrorAction SilentlyContinue) -join "`n" } else { "stderr missing" }
            throw "torch_server_startup_failure: process exited code $($proc.ExitCode). $stderrTail"
        }
        if (Test-ServerReachable -BaseUrl $torchServerUrl) {
            break
        }
        Start-Sleep -Milliseconds 500
    }

    if (-not (Test-ServerReachable -BaseUrl $torchServerUrl)) {
        throw "torch_server_startup_failure: server not reachable at $torchServerUrl"
    }

    if (Test-Path $torchServerMetaPath) {
        $meta = Read-JsonFile -Path $torchServerMetaPath
        $pipelineMeta.torch_server.selected_model_id = $meta.selected_model_id
        $pipelineMeta.torch_server.fallback_triggered = [bool]$meta.fallback_triggered
        $pipelineMeta.torch_server.fallback_reason_signature = $meta.fallback_reason_signature
        $pipelineMeta.torch_server.device = $meta.device
        $pipelineMeta.torch_server.runtime_device_fallback = [bool]$meta.runtime_device_fallback
        $pipelineMeta.torch_server.cache_root = $meta.cache_root
        $pipelineMeta.torch_server.operator_optimizations = $meta.operator_optimizations
    }

    $healthMeta = Get-TorchAttachServerHealth -BaseUrl $torchServerUrl
    if ($null -ne $healthMeta) {
        if (-not $pipelineMeta.torch_server.selected_model_id) {
            $pipelineMeta.torch_server.selected_model_id = $healthMeta.selected_model_id
        }
        if ($null -eq $pipelineMeta.torch_server.fallback_triggered) {
            $pipelineMeta.torch_server.fallback_triggered = $healthMeta.fallback_triggered
        }
        if ($null -eq $pipelineMeta.torch_server.runtime_device_fallback) {
            $pipelineMeta.torch_server.runtime_device_fallback = $healthMeta.runtime_device_fallback
        }
        if (-not $pipelineMeta.torch_server.device) {
            $pipelineMeta.torch_server.device = $healthMeta.device
        }
    }
    elseif ($TorchRequireGpu -and -not (Test-Path $torchServerMetaPath)) {
        throw "torch_gpu_required_meta_missing: $torchServerMetaPath and $torchServerUrl/health"
    }

    if ($TorchRequireGpu) {
        $device = [string]$pipelineMeta.torch_server.device
        $runtimeFallback = [bool]$pipelineMeta.torch_server.runtime_device_fallback
        $fallbackTriggered = [bool]$pipelineMeta.torch_server.fallback_triggered
        if ($device -ne "cuda" -or $runtimeFallback -or $fallbackTriggered) {
            $pipelineMeta.execution.torch_gpu_verified = $false
            $pipelineMeta.torch_server.gpu_required_pass = $false
            throw "torch_gpu_required_but_cpu_fallback: device=$device runtime_device_fallback=$runtimeFallback fallback_triggered=$fallbackTriggered reason=$($pipelineMeta.torch_server.fallback_reason_signature)"
        }
        $pipelineMeta.execution.torch_gpu_verified = $true
        $pipelineMeta.torch_server.gpu_required_pass = $true
    }
    else {
        $pipelineMeta.execution.torch_gpu_verified = $null
        $pipelineMeta.torch_server.gpu_required_pass = $null
    }

    $script:TorchServerManaged = $true
    $script:TorchServerProc = $proc
    return $proc
}

function Stop-TorchAttachServer {
    param([System.Diagnostics.Process]$Process)
    if ($null -eq $Process) {
        return
    }
    try {
        if (-not $Process.HasExited) {
            $Process.Kill()
            $Process.WaitForExit(5000) | Out-Null
        }
    }
    catch {
        # ignore cleanup errors
    }
    finally {
        $script:TorchServerManaged = $false
        $script:TorchServerProc = $null
    }
}

$exitCode = 0
try {
    Assert-DDrivePath -Name "OutRoot" -PathLike $outRootResolved
    Assert-DDrivePath -Name "TorchModelRoot" -PathLike $TorchModelRoot
    Assert-DDrivePath -Name "MlcCompileGlobalCacheRoot" -PathLike $script:MlcCompileGlobalCacheRoot
    Assert-DDrivePath -Name "MlcLlmHome" -PathLike $script:MlcLlmHome
    if (-not [string]::IsNullOrWhiteSpace($torchTunableOpResultsFileResolved)) {
        Assert-DDrivePath -Name "TorchTunableOpResultsFile" -PathLike $torchTunableOpResultsFileResolved
    }
    if ($ortBackendSelected) {
        if ($torchServerUrl.TrimEnd("/") -ieq $OrtServerUrl.TrimEnd("/")) {
            throw "server_startup_failure: Torch and ORT attach URLs must differ (torch=$torchServerUrl, ort=$OrtServerUrl)"
        }
    }
    else {
        $ortUrlCheckStatus = if ($DryRun) { "dry-run" } else { "executed" }
        Add-Step -Name "preflight-ort-url-check" -Command "skip because ort backend not selected (skip ORT URL conflict check)" -ReturnCode 0 -Status $ortUrlCheckStatus
    }
    if ($DryRun) {
        Add-Step -Name "d-drive-gate" -Command "validate OutRoot/TorchModelRoot/MlcCompileGlobalCacheRoot on D drive" -ReturnCode 0 -Status "dry-run"
        $pipelineMeta.d_drive_gate_pass = $true
    }
    else {
        Add-Step -Name "d-drive-gate" -Command "validate OutRoot/TorchModelRoot/MlcCompileGlobalCacheRoot on D drive" -ReturnCode 0 -Status "executed"
        $pipelineMeta.d_drive_gate_pass = $true
    }
    $aiperfTtftModeStatus = if ($DryRun) { "dry-run" } else { "executed" }
    $aiperfStreamingText = if ($AiperfEnableStreaming) { "true" } else { "false" }
    $aiperfTtftModeCommand = "aiperf ttft mode: endpoint=$AiperfEndpointType streaming=$aiperfStreamingText"
    Add-Step -Name "preflight-aiperf-ttft-mode" -Command $aiperfTtftModeCommand -ReturnCode 0 -Status $aiperfTtftModeStatus
    $llamaProfileOverrideStatus = if ($DryRun) { "dry-run" } else { "executed" }
    if ($llamaProfileOverrideEnabled) {
        $llamaProfileOverrideCommand = "apply llama profile override: c=$LlamaProfileOverrideConcurrency req=$LlamaProfileOverrideRequestCount prompt=$LlamaProfileOverridePromptTokensMean output=$LlamaProfileOverrideOutputTokensMean"
    }
    else {
        $llamaProfileOverrideCommand = "skip llama profile override: disabled (all zeros)"
    }
    Add-Step -Name "preflight-llama-profile-override" -Command $llamaProfileOverrideCommand -ReturnCode 0 -Status $llamaProfileOverrideStatus
    if ([string]::IsNullOrWhiteSpace($LlamaProfileIdOverride)) {
        $llamaProfileIdOverrideCommand = "skip llama profile id override: disabled (empty)"
    }
    else {
        $llamaProfileIdOverrideCommand = "apply llama profile id override: $LlamaProfileIdOverride"
    }
    Add-Step -Name "preflight-llama-profile-id-override" -Command $llamaProfileIdOverrideCommand -ReturnCode 0 -Status $llamaProfileOverrideStatus
    $mlcProfileOverrideStatus = if ($DryRun) { "dry-run" } else { "executed" }
    if ($mlcProfileOverrideEnabled) {
        $mlcProfileOverrideCommand = "apply mlc profile override: c=$MlcProfileOverrideConcurrency req=$MlcProfileOverrideRequestCount prompt=$MlcProfileOverridePromptTokensMean output=$MlcProfileOverrideOutputTokensMean"
    }
    else {
        $mlcProfileOverrideCommand = "skip mlc profile override: disabled (all zeros)"
    }
    Add-Step -Name "preflight-mlc-profile-override" -Command $mlcProfileOverrideCommand -ReturnCode 0 -Status $mlcProfileOverrideStatus
    if ([string]::IsNullOrWhiteSpace($MlcProfileIdOverride)) {
        $mlcProfileIdOverrideCommand = "skip mlc profile id override: disabled (empty)"
    }
    else {
        $mlcProfileIdOverrideCommand = "apply mlc profile id override: $MlcProfileIdOverride"
    }
    Add-Step -Name "preflight-mlc-profile-id-override" -Command $mlcProfileIdOverrideCommand -ReturnCode 0 -Status $mlcProfileOverrideStatus
    if ($mlcPrefillChunkSizeOverrideEnabled) {
        $mlcPrefillOverrideCommand = "apply mlc prefill_chunk_size override: prefill=$MlcPrefillChunkSizeOverride"
    }
    else {
        $mlcPrefillOverrideCommand = "skip mlc prefill_chunk_size override: disabled (0)"
    }
    Add-Step -Name "preflight-mlc-prefill-override" -Command $mlcPrefillOverrideCommand -ReturnCode 0 -Status $mlcProfileOverrideStatus
    $ortProfileOverrideStatus = if ($DryRun) { "dry-run" } else { "executed" }
    if ($ortProfileOverrideEnabled) {
        $ortProfileOverrideCommand = "apply ort profile override: c=$OrtProfileOverrideConcurrency req=$OrtProfileOverrideRequestCount prompt=$OrtProfileOverridePromptTokensMean output=$OrtProfileOverrideOutputTokensMean"
    }
    else {
        $ortProfileOverrideCommand = "skip ort profile override: disabled (all zeros)"
    }
    Add-Step -Name "preflight-ort-profile-override" -Command $ortProfileOverrideCommand -ReturnCode 0 -Status $ortProfileOverrideStatus
    if ([string]::IsNullOrWhiteSpace($OrtProfileIdOverride)) {
        $ortProfileIdOverrideCommand = "skip ort profile id override: disabled (empty)"
    }
    else {
        $ortProfileIdOverrideCommand = "apply ort profile id override: $OrtProfileIdOverride"
    }
    Add-Step -Name "preflight-ort-profile-id-override" -Command $ortProfileIdOverrideCommand -ReturnCode 0 -Status $ortProfileOverrideStatus
    $torchProfileOverrideStatus = if ($DryRun) { "dry-run" } else { "executed" }
    if ($torchProfileOverrideEnabled) {
        $torchProfileOverrideCommand = "apply torch profile override: c=$TorchProfileOverrideConcurrency req=$TorchProfileOverrideRequestCount prompt=$TorchProfileOverridePromptTokensMean output=$TorchProfileOverrideOutputTokensMean"
    }
    else {
        $torchProfileOverrideCommand = "skip torch profile override: disabled (all zeros)"
    }
    Add-Step -Name "preflight-torch-profile-override" -Command $torchProfileOverrideCommand -ReturnCode 0 -Status $torchProfileOverrideStatus
    if ([string]::IsNullOrWhiteSpace($TorchProfileIdOverride)) {
        $torchProfileIdOverrideCommand = "skip torch profile id override: disabled (empty)"
    }
    else {
        $torchProfileIdOverrideCommand = "apply torch profile id override: $TorchProfileIdOverride"
    }
    Add-Step -Name "preflight-torch-profile-id-override" -Command $torchProfileIdOverrideCommand -ReturnCode 0 -Status $torchProfileOverrideStatus
    if ([bool]$TorchEnableTunableOp) {
        $tunableFileText = if ([string]::IsNullOrWhiteSpace($torchTunableOpResultsFileResolved)) { "(auto)" } else { $torchTunableOpResultsFileResolved }
        $torchHipOpsCommand = "apply torch HIP ops: HIP_FORCE_DEV_KERNARG=$([int][bool]$TorchHipForceDevKernargBool) blas=$TorchPreferredBlasBackend rocm_fa=$TorchPreferredRocmFaBackend tunableop=on tuning=$([int][bool]$TorchTunableOpTuning) file=$tunableFileText"
    }
    else {
        $torchHipOpsCommand = "apply torch HIP ops: HIP_FORCE_DEV_KERNARG=$([int][bool]$TorchHipForceDevKernargBool) blas=$TorchPreferredBlasBackend rocm_fa=$TorchPreferredRocmFaBackend tunableop=off"
    }
    Add-Step -Name "preflight-torch-hip-ops" -Command $torchHipOpsCommand -ReturnCode 0 -Status $torchProfileOverrideStatus

    if (-not $DryRun) {
        New-Item -ItemType Directory -Path $outRootResolved, $runsRoot, $compareRoot, $gpuUtilRoot, $torchServerLogDir, $script:MlcCompileGlobalCacheRoot, $script:MlcLlmHome -Force | Out-Null
    }
    Write-ProgressCheckpoint -State "running_preflight"

    $env:PERFLAB_TOOL_TIMEOUT_SEC = [string]$perProfileTimeoutSec
    $env:MLC_LLM_HOME = $script:MlcLlmHome
    $env:MLC_DOWNLOAD_CACHE_POLICY = "ON"
    Enter-AiperfStabilityEnv
    Enter-MlcCompileTimeoutEnv
    $aiperfEnvStatus = if ($DryRun) { "dry-run" } else { "executed" }
    Add-Step -Name "preflight-aiperf-service-env" -Command "set AIPERF service timeout env defaults" -ReturnCode 0 -Status $aiperfEnvStatus
    Add-Step -Name "preflight-mlc-compile-timeout-env" -Command "set PERFLAB_MLC_COMPILE_TIMEOUT_SEC=$MlcCompileTimeoutSec" -ReturnCode 0 -Status $aiperfEnvStatus
    Add-Step -Name "preflight-mlc-cache-env" -Command "set MLC_LLM_HOME=$script:MlcLlmHome; MLC_DOWNLOAD_CACHE_POLICY=ON" -ReturnCode 0 -Status $aiperfEnvStatus
    $pipelineMeta.config["mlc_llm_home"] = $script:MlcLlmHome

    Write-Host "[preflight] python start"
    Invoke-Step -Name "preflight-python" -Executable $PythonExe -Arguments @("-V") -TimeoutSeconds $PreflightTimeoutSeconds
    Write-Host "[preflight] python done"
    Write-Host "[preflight] llama-server start"
    if ($DryRun) {
        Add-Step -Name "preflight-llama-server" -Command "$script:LlamaServerBinEffective --version" -ReturnCode 0 -Status "dry-run"
    }
    else {
        if (-not (Test-Path $script:LlamaServerBinEffective)) {
            throw "server_startup_failure: llama server binary not found: $script:LlamaServerBinEffective"
        }
        Invoke-Step -Name "preflight-llama-server" -Executable $script:LlamaServerBinEffective -Arguments @("--version") -TimeoutSeconds $PreflightTimeoutSeconds
    }
    Write-Host "[preflight] llama-server done"
    Write-Host "[preflight] torch-python start"
    Invoke-Step -Name "preflight-torch-python" -Executable $TorchPythonExe -Arguments @("-V") -TimeoutSeconds $PreflightTimeoutSeconds
    Write-Host "[preflight] torch-python done"
    Write-Host "[preflight] mlc-help start"
    $mlcOptSupported = Probe-MlcOptSupport
    Write-Host "[preflight] mlc-help done"
    if (-not $mlcOptSupported -and $MlcOpt -and $MlcOpt.Trim() -ne "") {
        $script:MlcOptEffective = ""
        $downgradeStatus = if ($DryRun) { "dry-run" } else { "executed" }
        Add-Step -Name "preflight-mlc-opt-downgrade" -Command "disable --mlc-opt because mlc_llm serve --help does not support --opt" -ReturnCode 0 -Status $downgradeStatus
    } else {
        $script:MlcOptEffective = $MlcOpt
    }
    $pipelineMeta.config["mlc_opt_requested"] = $MlcOpt
    $pipelineMeta.config["mlc_opt_effective"] = $script:MlcOptEffective
    $pipelineMeta.config["mlc_opt_supported"] = [bool]$mlcOptSupported

    $localTokenizerMain = Ensure-LocalTokenizerCache -ModelRef $TokenizerRef -Alias "main"
    if (-not [string]::IsNullOrWhiteSpace([string]$localTokenizerMain)) {
        $script:ResolvedTokenizerRefMain = [string]$localTokenizerMain
    }
    else {
        $script:ResolvedTokenizerRefMain = $TokenizerRef
    }
    if ($ortBackendSelected) {
        $localTokenizerOrt = Ensure-LocalTokenizerCache -ModelRef $OrtTokenizerRef -Alias "ort"
        if (-not [string]::IsNullOrWhiteSpace([string]$localTokenizerOrt)) {
            $script:ResolvedTokenizerRefOrt = [string]$localTokenizerOrt
        }
        else {
            $script:ResolvedTokenizerRefOrt = $OrtTokenizerRef
        }
    }
    else {
        $script:ResolvedTokenizerRefOrt = $OrtTokenizerRef
        $ortTokenizerStatus = if ($DryRun) { "dry-run" } else { "executed" }
        Add-Step -Name "preflight-tokenizer-cache-ort" -Command "skip because ort backend not selected (skip ORT tokenizer cache preflight)" -ReturnCode 0 -Status $ortTokenizerStatus
    }
    $pipelineMeta.config["tokenizer_ref_effective"] = $script:ResolvedTokenizerRefMain
    $pipelineMeta.config["ort_tokenizer_ref_effective"] = $script:ResolvedTokenizerRefOrt

    Write-Host "[preflight] compare-runs-help start"
    Invoke-Step -Name "preflight-compare-runs-help" -Executable $PythonExe -Arguments @("harness/benchctl.py", "compare-runs", "--help") -TimeoutSeconds $PreflightTimeoutSeconds
    Write-Host "[preflight] compare-runs-help done"

    Write-Host "[preflight] ort attach probe start"
    if ($ortBackendSelected) {
        if ($DryRun) {
            Add-Step -Name "preflight-ort-server" -Command "probe $OrtServerUrl" -ReturnCode 0 -Status "dry-run"
        }
        else {
            if (-not (Test-ServerReachable -BaseUrl $OrtServerUrl)) {
                throw "server_startup_failure: ORT server unreachable at $OrtServerUrl"
            }
            Add-Step -Name "preflight-ort-server" -Command "probe $OrtServerUrl" -ReturnCode 0 -Status "executed"
        }
    }
    else {
        $ortProbeStatus = if ($DryRun) { "dry-run" } else { "executed" }
        Add-Step -Name "preflight-ort-server" -Command "skip because ort backend not selected (skip ORT server probe)" -ReturnCode 0 -Status $ortProbeStatus
    }
    Write-Host "[preflight] ort attach probe done"
    Assert-MinSystemFreeMemory -Context "preflight"

    Write-ProgressCheckpoint -State "running_preflight"

    Write-Host "[preflight] torch attach startup start"
    if ($torchBackendSelected) {
        if ($TorchAttachLifecycle -eq "preflight") {
            $script:TorchServerProc = Start-TorchAttachServer
        }
        else {
            $torchStartupStatus = if ($DryRun) { "dry-run" } else { "executed" }
            Add-Step -Name "preflight-torch-attach-startup" -Command "defer startup until torch_rocm backend stage (on_demand lifecycle)" -ReturnCode 0 -Status $torchStartupStatus
        }
    }
    else {
        $torchStartupStatus = if ($DryRun) { "dry-run" } else { "executed" }
        Add-Step -Name "start-torch-attach-server" -Command "skip because torch_rocm backend not selected (skip torch attach startup)" -ReturnCode 0 -Status $torchStartupStatus
        $pipelineMeta.config["torch_attach_lifecycle"] = "disabled"
    }
    Write-Host "[preflight] torch attach startup done"
    if (-not $DryRun) {
        Write-HourlyCheckpoint -Force -Reason "run_started"
    }

    foreach ($interruptedRunId in @($InterruptedRunIds)) {
        if (-not [string]::IsNullOrWhiteSpace([string]$interruptedRunId)) {
            Add-InterruptedRun -RunId ([string]$interruptedRunId) -Signature "manual_stop_for_telemetry_rebuild" -Reason "carried forward into current run"
        }
    }

    # Stage A: mainline full sweep
    foreach ($backend in $mainlineBackendOrder) {
        foreach ($profile in @($profilesByBackend[$backend])) {
            $currentRunId = "{0}-{1}-stage_a" -f $backend, $profile.id
            Write-ProgressCheckpoint -State "running_stage_a_mainline" -LastRunId $currentRunId

            $row = $null
            if ($backend -eq "mlc") {
                if ($script:MlcCompileGlobalBlocked) {
                    $row = [pscustomobject]@{
                        backend = $backend
                        profile_id = $profile.id
                        stage = "stage_a"
                        status = "failed"
                        attempt = 1
                        run_dir = (Join-Path $runsRoot ("{0}-{1}-stage_a" -f $backend, $profile.id))
                        samples = 0
                        ttft_p50_ms = 0.0
                        itl_p50_ms = 0.0
                        tps_mean = 0.0
                        rps_mean = 0.0
                        gpu_util_avg = 0.0
                        gpu_util_artifact = $null
                        blocker_signature = if ($script:MlcCompileGlobalBlocker) { [string]$script:MlcCompileGlobalBlocker } else { "mlc_compile_preflight_failed" }
                        error = if ($script:MlcCompileGlobalError) { [string]$script:MlcCompileGlobalError } else { "mlc compile preflight globally blocked" }
                        score = 0.0
                    }
                    $pipelineMeta.execution.mlc_compile_preflight.blocked_profiles += [ordered]@{
                        profile_id = $profile.id
                        compile_key = $null
                        blocker_signature = $row.blocker_signature
                        blocked_by_global = $true
                    }
                }
                else {
                    $preflight = Invoke-MlcCompilePreflight -Profile $profile
                    $pipelineMeta.execution.mlc_compile_preflight.by_profile[$profile.id] = [ordered]@{
                        compile_key = $preflight.compile_key
                        ok = [bool]$preflight.ok
                        model_lib = $preflight.model_lib
                        used_validation_bypass = [bool]$preflight.used_validation_bypass
                        degraded_release = [bool]$preflight.degraded_release
                        cache_scope = $preflight.cache_scope
                        cached = [bool]$preflight.cached
                    }
                    $pipelineMeta.execution.mlc_compile_preflight.by_key[$preflight.compile_key] = [ordered]@{
                        ok = [bool]$preflight.ok
                        model_lib = $preflight.model_lib
                        used_validation_bypass = [bool]$preflight.used_validation_bypass
                        degraded_release = [bool]$preflight.degraded_release
                        cache_scope = $preflight.cache_scope
                        cached = [bool]$preflight.cached
                    }
                    if (-not $preflight.ok) {
                        $signature = if ($preflight.signature) { [string]$preflight.signature } else { "mlc_compile_preflight_failed" }
                        $errorText = if ($preflight.error) { [string]$preflight.error } else { "mlc compile preflight failed for profile=$($profile.id)" }
                        $pipelineMeta.execution.mlc_compile_preflight.blocked_profiles += [ordered]@{
                            profile_id = $profile.id
                            compile_key = $preflight.compile_key
                            blocker_signature = $signature
                        }
                        if ($signature -eq "mlc_vulkan_compile_vuid_10866") {
                            $script:MlcCompileGlobalBlocked = $true
                            $script:MlcCompileGlobalBlocker = $signature
                            $script:MlcCompileGlobalError = $errorText
                            $pipelineMeta.execution.mlc_compile_preflight.global_blocker = $signature
                            $pipelineMeta.execution.mlc_compile_preflight.global_error = $errorText
                        }
                        $row = [pscustomobject]@{
                            backend = $backend
                            profile_id = $profile.id
                            stage = "stage_a"
                            status = "failed"
                            attempt = 1
                            run_dir = (Join-Path $runsRoot ("{0}-{1}-stage_a" -f $backend, $profile.id))
                            samples = 0
                            ttft_p50_ms = 0.0
                            itl_p50_ms = 0.0
                            tps_mean = 0.0
                            rps_mean = 0.0
                            gpu_util_avg = 0.0
                            gpu_util_artifact = $null
                            blocker_signature = $signature
                            error = $errorText
                            score = 0.0
                        }
                    }
                }
            }
            if ($null -eq $row) {
                $row = [pscustomobject](Invoke-ProfileWithRetry -Backend $backend -Profile $profile -Stage "stage_a")
            }
            $allRows += $row
            $pipelineMeta.stage_results.stage_a += [ordered]@{
                backend = $row.backend
                profile_id = $row.profile_id
                status = $row.status
                blocker_signature = $row.blocker_signature
            }
            $pipelineMeta.progress.stage_a_completed = [int]$pipelineMeta.progress.stage_a_completed + 1
            $pipelineMeta.execution.mainline_stage_a_completed = [int]$pipelineMeta.execution.mainline_stage_a_completed + 1
            $pipelineMeta.progress.completed_profiles = [int]$pipelineMeta.progress.completed_profiles + 1
            Write-ProgressCheckpoint -State "running_stage_a_mainline" -LastRunId $currentRunId
        }
    }

    $rankAByBackend = @{}
    foreach ($backend in $mainlineBackendOrder) {
        $rankedMain = @(Rank-BackendRows -Rows $allRows -Backend $backend -Stage "stage_a")
        $rankAByBackend[$backend] = $rankedMain
        if ($rankedMain.Count -eq 0) {
            $failedRows = @($allRows | Where-Object { $_.backend -eq $backend -and $_.stage -eq "stage_a" -and $_.status -eq "failed" })
            if ($failedRows.Count -gt 0) {
                $topSig = @(
                    $failedRows |
                        Group-Object -Property blocker_signature |
                        Sort-Object -Property Count -Descending |
                        Select-Object -First 1
                )
                if ($topSig.Count -gt 0 -and -not [string]::IsNullOrWhiteSpace([string]$topSig[0].Name)) {
                    $pipelineMeta.blocker_signature = [string]$topSig[0].Name
                }
            }
            throw "stage_a has no successful runs for backend=$backend"
        }
    }

    if (-not $MainlineClosureFirst) {
        # Stage A: experimental full sweep (legacy ordering)
        foreach ($backend in $experimentalBackendOrder) {
            foreach ($profile in @($experimentalProfilesByBackend[$backend])) {
                $currentRunId = "{0}-{1}-stage_a" -f $backend, $profile.id
                Write-ProgressCheckpoint -State "running_stage_a_experimental" -LastRunId $currentRunId
                $row = [pscustomobject](Invoke-ProfileWithRetry -Backend $backend -Profile $profile -Stage "stage_a")
                $allRows += $row
                $pipelineMeta.stage_results.stage_a += [ordered]@{
                    backend = $row.backend
                    profile_id = $row.profile_id
                    status = $row.status
                    blocker_signature = $row.blocker_signature
                }
                $pipelineMeta.progress.stage_a_completed = [int]$pipelineMeta.progress.stage_a_completed + 1
                $pipelineMeta.progress.completed_profiles = [int]$pipelineMeta.progress.completed_profiles + 1
                Write-ProgressCheckpoint -State "running_stage_a_experimental" -LastRunId $currentRunId
            }
        }
        foreach ($backend in $selectedBackends) {
            $rankAByBackend[$backend] = @(Rank-BackendRows -Rows $allRows -Backend $backend -Stage "stage_a")
        }
    }

    # Stage B: mainline top1 recheck
    foreach ($backend in $mainlineBackendOrder) {
        $topA = $rankAByBackend[$backend][0]
        $profile = @($profilesByBackend[$backend] | Where-Object { $_.id -eq $topA.profile_id } | Select-Object -First 1)[0]
        if (-not $profile) {
            throw "failed to resolve top profile for backend=$backend"
        }
        $currentRunId = "{0}-{1}-stage_b_recheck" -f $backend, $profile.id
        Write-ProgressCheckpoint -State "running_stage_b_mainline" -LastRunId $currentRunId
        $rowB = [pscustomobject](Invoke-ProfileWithRetry -Backend $backend -Profile $profile -Stage "stage_b_recheck")
        $allRows += $rowB
        $pipelineMeta.stage_results.stage_b += [ordered]@{
            backend = $rowB.backend
            profile_id = $rowB.profile_id
            status = $rowB.status
            blocker_signature = $rowB.blocker_signature
        }
        $pipelineMeta.progress.stage_b_completed = [int]$pipelineMeta.progress.stage_b_completed + 1
        $pipelineMeta.execution.mainline_stage_b_completed = [int]$pipelineMeta.execution.mainline_stage_b_completed + 1
        $pipelineMeta.progress.completed_profiles = [int]$pipelineMeta.progress.completed_profiles + 1
        Write-ProgressCheckpoint -State "running_stage_b_mainline" -LastRunId $currentRunId
    }
    $pipelineMeta.execution.mainline_completed = $true

    if ($MainlineClosureFirst) {
        # Stage A: experimental full sweep (mainline closure first ordering)
        foreach ($backend in $experimentalBackendOrder) {
            foreach ($profile in @($experimentalProfilesByBackend[$backend])) {
                $currentRunId = "{0}-{1}-stage_a" -f $backend, $profile.id
                Write-ProgressCheckpoint -State "running_stage_a_experimental" -LastRunId $currentRunId
                $row = [pscustomobject](Invoke-ProfileWithRetry -Backend $backend -Profile $profile -Stage "stage_a")
                $allRows += $row
                $pipelineMeta.stage_results.stage_a += [ordered]@{
                    backend = $row.backend
                    profile_id = $row.profile_id
                    status = $row.status
                    blocker_signature = $row.blocker_signature
                }
                $pipelineMeta.progress.stage_a_completed = [int]$pipelineMeta.progress.stage_a_completed + 1
                $pipelineMeta.progress.completed_profiles = [int]$pipelineMeta.progress.completed_profiles + 1
                Write-ProgressCheckpoint -State "running_stage_a_experimental" -LastRunId $currentRunId
            }
            $rankAByBackend[$backend] = @(Rank-BackendRows -Rows $allRows -Backend $backend -Stage "stage_a")
        }
    }

    $bestByBackend = @{}
    foreach ($backend in $selectedBackends) {
        $rankB = @(Rank-BackendRows -Rows $allRows -Backend $backend -Stage "stage_b_recheck")
        $best = if ($rankB.Count -gt 0) { $rankB[0] } else { $rankAByBackend[$backend][0] }
        if ($null -eq $best) {
            continue
        }
        $bestByBackend[$backend] = $best
        $pipelineMeta.best[$backend] = [ordered]@{
            profile_id = $best.profile_id
            stage = $best.stage
            score = [double]$best.score
            run_dir = $best.run_dir
        }
    }

    Write-Leaderboard -Rows $allRows

    $backendLabelMap = @{
        llama = "llama_cpp"
        mlc = "mlc_llm"
        ort = "ort_dml"
        torch_rocm = "torch_rocm"
    }
    $pairPlan = @()
    for ($i = 0; $i -lt $mainlineBackendOrder.Count; $i++) {
        for ($j = $i + 1; $j -lt $mainlineBackendOrder.Count; $j++) {
            $baselineBackend = [string]$mainlineBackendOrder[$i]
            $candidateBackend = [string]$mainlineBackendOrder[$j]
            if (-not $bestByBackend.ContainsKey($baselineBackend) -or -not $bestByBackend.ContainsKey($candidateBackend)) {
                continue
            }
            $pairPlan += [ordered]@{
                pair_name = "{0}_vs_{1}" -f $baselineBackend, $candidateBackend
                baseline_backend = $baselineBackend
                candidate_backend = $candidateBackend
            }
        }
    }
    $pairResults = @{}
    foreach ($pair in $pairPlan) {
        $pairName = [string]$pair.pair_name
        $baselineBackend = [string]$pair.baseline_backend
        $candidateBackend = [string]$pair.candidate_backend
        $pairResults[$pairName] = Invoke-ComparePair `
            -PairName $pairName `
            -BaselineRun $bestByBackend[$baselineBackend].run_dir `
            -CandidateRun $bestByBackend[$candidateBackend].run_dir `
            -BaselineLabel $backendLabelMap[$baselineBackend] `
            -CandidateLabel $backendLabelMap[$candidateBackend]
    }

    foreach ($pairName in $pairResults.Keys) {
        $pairResult = $pairResults[$pairName]
        $pipelineMeta.compare_winners[$pairName] = [ordered]@{
            winner = Get-FieldValue -Object $pairResult -Name "winner"
            vote_counts = Get-FieldValue -Object $pairResult -Name "vote_counts"
        }
    }

    if (-not $DryRun) {
        foreach ($pairName in @($pairResults.Keys)) {
            $pairDir = Join-Path $compareRoot $pairName
            foreach ($name in @("comparison_summary.csv", "comparison_report.md", "compare_meta.json")) {
                $path = Join-Path $pairDir $name
                if (-not (Test-Path $path)) {
                    throw "compare_artifacts_gate_failure: missing $path"
                }
            }
        }
        Build-MatrixReport -PairResults $pairResults
        if (-not (Test-Path $leaderboardPath)) { throw "compare_artifacts_gate_failure: leaderboard missing" }
        if (-not (Test-Path $matrixReportPath)) { throw "compare_artifacts_gate_failure: matrix report missing" }
        $pipelineMeta.artifacts_ready = $true
        Write-HourlyCheckpoint -Force -Reason "pipeline_completed"
        Write-ProgressCheckpoint -State "completed"
    }
    else {
        Add-Step -Name "postcheck-artifacts" -Command "dry-run verify leaderboard/matrix/compare triplets" -ReturnCode 0 -Status "dry-run"
    }
}
catch {
    $exitCode = 1
    $err = $_.Exception.Message
    if ([string]::IsNullOrWhiteSpace($err)) {
        $err = ($_ | Out-String)
    }
    $pipelineMeta.error = $err
    if (-not $pipelineMeta.blocker_signature) {
        $pipelineMeta.blocker_signature = Resolve-RunFailureSignature -Message $err -Backend ""
    }
    Write-Host "[error] $err"
    Write-HourlyCheckpoint -Force -Reason "pipeline_failed"
    Write-ProgressCheckpoint -State "failed"
    Write-Leaderboard -Rows $allRows
}
finally {
    if ($script:TorchServerManaged -and $null -ne $script:TorchServerProc) {
        Stop-TorchAttachServer -Process $script:TorchServerProc
    }
    Remove-Item Env:PERFLAB_TOOL_TIMEOUT_SEC -ErrorAction SilentlyContinue
    Exit-MlcCompileTimeoutEnv
    Exit-AiperfStabilityEnv

    $vizGeneratedAt = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    if ($SkipVisualizationFinalize) {
        $skipVizStatus = if ($DryRun) { "dry-run" } else { "executed" }
        Add-Step -Name "build-perf-timeline-report" -Command "skip visualization finalize because SkipVisualizationFinalize=true" -ReturnCode 0 -Status $skipVizStatus
        Add-Step -Name "collect-rgp-metrics" -Command "skip RGP collection because SkipVisualizationFinalize=true" -ReturnCode 0 -Status $skipVizStatus
        $pipelineMeta.visualization.generated_at_utc = $vizGeneratedAt
        $pipelineMeta.visualization.ready = $false
        $pipelineMeta.visualization.error = "skipped_by_flag"
        $pipelineMeta.visualization.rgp_report_path = $rgpReportPath
        $pipelineMeta.visualization.rgp_ready = $false
        $pipelineMeta.visualization.rgp_error = "skipped_by_flag"
    }
    else {
        $vizError = $null
        try {
            $vizArgs = @(
                "scripts/build_perf_timeline_report.py",
                "--results-root", "results",
                "--out-root", $perfTimelineRoot
            )
            $vizRc = Invoke-Step -Name "build-perf-timeline-report" -Executable $PythonExe -Arguments $vizArgs -AllowFailure
            $pipelineMeta.visualization.generated_at_utc = $vizGeneratedAt
            if ($DryRun) {
                $pipelineMeta.visualization.ready = $false
                $pipelineMeta.visualization.error = "dry-run"
            }
            elseif ($vizRc -eq 0 -and (Test-Path $perfTimelineReportPath)) {
                $pipelineMeta.visualization.ready = $true
                $pipelineMeta.visualization.error = $null
            }
            else {
                $vizError = "timeline report build failed: returncode=$vizRc report_path=$perfTimelineReportPath"
                $pipelineMeta.visualization.ready = $false
                $pipelineMeta.visualization.error = $vizError
            }
        }
        catch {
            $vizError = $_.Exception.Message
            if ([string]::IsNullOrWhiteSpace($vizError)) {
                $vizError = ($_ | Out-String)
            }
            $pipelineMeta.visualization.generated_at_utc = $vizGeneratedAt
            $pipelineMeta.visualization.ready = $false
            $pipelineMeta.visualization.error = "timeline report build exception: $vizError"
        }

        $rgpError = $null
        try {
            $pipelineMeta.visualization.rgp_report_path = $rgpReportPath
            if ($DryRun) {
                $pipelineMeta.visualization.rgp_ready = $false
                $pipelineMeta.visualization.rgp_error = "dry-run"
            }
            elseif (-not (Test-Path $rgpRawRoot)) {
                $pipelineMeta.visualization.rgp_ready = $false
                $pipelineMeta.visualization.rgp_error = "rgp_raw_missing: $rgpRawRoot"
            }
            else {
                $rgpArgs = @(
                    "scripts/collect_rgp_metrics.py",
                    "--input-root", $rgpRawRoot,
                    "--out-root", $perfTimelineRoot
                )
                $rgpRc = Invoke-Step -Name "collect-rgp-metrics" -Executable $PythonExe -Arguments $rgpArgs -AllowFailure
                if ($rgpRc -eq 0 -and (Test-Path $rgpReportPath)) {
                    $pipelineMeta.visualization.rgp_ready = $true
                    $pipelineMeta.visualization.rgp_error = $null
                }
                else {
                    $rgpError = "rgp metrics collect failed: returncode=$rgpRc report_path=$rgpReportPath"
                    $pipelineMeta.visualization.rgp_ready = $false
                    $pipelineMeta.visualization.rgp_error = $rgpError
                }
            }
        }
        catch {
            $rgpError = $_.Exception.Message
            if ([string]::IsNullOrWhiteSpace($rgpError)) {
                $rgpError = ($_ | Out-String)
            }
            $pipelineMeta.visualization.rgp_ready = $false
            $pipelineMeta.visualization.rgp_error = "rgp collect exception: $rgpError"
        }
    }

    $pipelineMeta.execution.long_run_auto_closed = [bool]$script:LongRunAutoClosed
    $pipelineMeta.execution.no_progress_abort_count = [int]$script:NoProgressAbortCount
    $pipelineMeta.execution.adaptive_profile_downgrade_count = [int]$script:AdaptiveProfileDowngradeCount
    $pipelineMeta.execution.llama_kv_fallback_count = [int]$script:LlamaKvFallbackCount
    Write-PipelineMeta
    if ($DryRun) {
        Write-Host "dry-run complete (no artifacts written)"
    }
    elseif ($exitCode -eq 0) {
        Write-Host "gpu util uplift completed: $outRootResolved"
        Write-Host "leaderboard: $leaderboardPath"
        Write-Host "matrix_report: $matrixReportPath"
        if ($pipelineMeta.visualization.ready) {
            Write-Host "timeline_report: $perfTimelineReportPath"
        }
        elseif (-not [string]::IsNullOrWhiteSpace([string]$pipelineMeta.visualization.error)) {
            Write-Host "timeline_report_warning: $($pipelineMeta.visualization.error)"
        }
        if ($pipelineMeta.visualization.rgp_ready) {
            Write-Host "rgp_report: $rgpReportPath"
        }
        elseif (-not [string]::IsNullOrWhiteSpace([string]$pipelineMeta.visualization.rgp_error)) {
            Write-Host "rgp_report_warning: $($pipelineMeta.visualization.rgp_error)"
        }
        Write-Host "meta: $pipelineMetaPath"
    }
    else {
        Write-Host "gpu util uplift failed: $outRootResolved"
        if (-not [string]::IsNullOrWhiteSpace([string]$pipelineMeta.visualization.error)) {
            Write-Host "timeline_report_warning: $($pipelineMeta.visualization.error)"
        }
        if (-not [string]::IsNullOrWhiteSpace([string]$pipelineMeta.visualization.rgp_error)) {
            Write-Host "rgp_report_warning: $($pipelineMeta.visualization.rgp_error)"
        }
        Write-Host "meta: $pipelineMetaPath"
    }
    exit $exitCode
}

