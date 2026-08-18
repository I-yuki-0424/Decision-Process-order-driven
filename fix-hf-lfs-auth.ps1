<#
Diagnostic and repair script for Hugging Face git-lfs push failures.
Run each numbered step in order. The script stops and prints guidance
at the first branch point where a precondition fails, rather than
proceeding on an unverified assumption.

Fill in these three values before running:
#>
$HFUsername = "I-yuki-0424"                                            # your Hugging Face username
$RepoPath   = "I-yuki-0424/Decision-process-order-driven-prttype"      # <namespace>/<repo>
$RepoHost   = "huggingface.co"

Write-Host "=== Step 1: Confirm HF_TOKEN is actually set in this session ===" -ForegroundColor Cyan
if (-not $env:HF_TOKEN -or $env:HF_TOKEN.Trim() -eq "") {
    Write-Host "FAIL: `$env:HF_TOKEN is empty or unset in this shell." -ForegroundColor Red
    Write-Host "Set it first: `$env:HF_TOKEN = '<your token>'  (this session only)"
    Write-Host "Or persist it: [System.Environment]::SetEnvironmentVariable('HF_TOKEN','<token>','User')"
    exit 1
}
Write-Host "OK: HF_TOKEN is set (length $($env:HF_TOKEN.Length))." -ForegroundColor Green

Write-Host "`n=== Step 2: Verify the token against the HF API directly (bypasses git entirely) ===" -ForegroundColor Cyan
try {
    $whoami = Invoke-RestMethod -Uri "https://huggingface.co/api/whoami-v2" `
        -Headers @{ Authorization = "Bearer $($env:HF_TOKEN)" } -ErrorAction Stop
    Write-Host "OK: token is valid for account '$($whoami.name)'." -ForegroundColor Green
} catch {
    Write-Host "FAIL: token rejected by the HF API itself. This IS a token problem." -ForegroundColor Red
    Write-Host "Regenerate a token with write scope at https://huggingface.co/settings/tokens and re-run this script."
    exit 1
}

Write-Host "`n=== Step 3: Confirm this token can write to the target repo (permission, not just identity) ===" -ForegroundColor Cyan
try {
    $repoInfo = Invoke-RestMethod -Uri "https://huggingface.co/api/models/$RepoPath" `
        -Headers @{ Authorization = "Bearer $($env:HF_TOKEN)" } -ErrorAction Stop
    Write-Host "OK: repo '$RepoPath' is reachable with this token." -ForegroundColor Green
} catch {
    Write-Host "WARN: could not confirm repo access via API (may be a dataset-type repo, not model-type)." -ForegroundColor Yellow
    Write-Host "If this repo is a dataset, the correct path convention is 'datasets/$RepoPath' -- confirm the exact URL in your browser before continuing."
}

Write-Host "`n=== Step 4: Wipe all existing HF-related git config in this repo (local scope only) ===" -ForegroundColor Cyan
git config --local --unset-all lfs.url 2>$null
git config --local --unset-all "http.https://$RepoHost/$RepoPath.git.extraheader" 2>$null
git config --local --remove-section "lfs" 2>$null
git config --local --remove-section "credential.https://co" 2>$null
git config --local --remove-section "http.https://huggingface" 2>$null
Write-Host "Local lfs/http/credential sections related to HF cleared."

Write-Host "`n=== Step 5: Check for an empty or missing credential.helper at any scope ===" -ForegroundColor Cyan
$globalHelper = git config --global --get credential.helper 2>$null
$localHelper  = git config --local --get credential.helper 2>$null
if (-not $globalHelper -or $globalHelper.Trim() -eq "") {
    Write-Host "No global credential.helper found. Setting one so credentials can actually be stored." -ForegroundColor Yellow
    git config --global credential.helper manager
    $globalHelper = "manager"
}
Write-Host "Active credential.helper (global): $globalHelper"
if ($localHelper -eq "") {
    Write-Host "WARN: local credential.helper is explicitly empty, which disables the global one for this repo." -ForegroundColor Red
    git config --local --unset credential.helper
    Write-Host "Removed the empty local override."
}

Write-Host "`n=== Step 6: Clear any stale Windows Credential Manager entries for this host ===" -ForegroundColor Cyan
$cmdkeyList = cmdkey /list 2>$null | Select-String -Pattern $RepoHost
if ($cmdkeyList) {
    Write-Host "Found existing entries referencing $RepoHost -- removing them:" -ForegroundColor Yellow
    foreach ($line in $cmdkeyList) {
        if ($line -match "Target:\s*(\S+)") {
            $target = $Matches[1]
            cmdkey /delete:$target 2>$null
            Write-Host "  removed $target"
        }
    }
} else {
    Write-Host "No cmdkey entries found for $RepoHost."
}

Write-Host "`n=== Step 7: Set the plain (no embedded credentials) LFS URL ===" -ForegroundColor Cyan
$lfsUrl = "https://$RepoHost/$RepoPath.git/info/lfs"
git config --local lfs.url "$lfsUrl"
Write-Host "lfs.url set to: $lfsUrl"

Write-Host "`n=== Step 8: Inject the credential directly via 'git credential approve' ===" -ForegroundColor Cyan
Write-Host "This writes the credential into the active helper's store without needing a popup to appear."
$credInput = @"
protocol=https
host=$RepoHost
path=$RepoPath.git
username=$HFUsername
password=$($env:HF_TOKEN)

"@
$credInput | git credential approve
Write-Host "Credential submitted to helper '$globalHelper'."

Write-Host "`n=== Step 9: Test plain git auth against this host (no LFS involved yet) ===" -ForegroundColor Cyan
$lsRemote = git ls-remote "https://$RepoHost/$RepoPath.git" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAIL: plain git auth against $RepoHost still failing:" -ForegroundColor Red
    Write-Host $lsRemote
    Write-Host "`nAt this point the problem is in how the credential helper is storing/retrieving the entry,"
    Write-Host "not in the token or the LFS-specific config. Consider switching credential.helper to 'store'"
    Write-Host "(plaintext file, local machine only) as a diagnostic: git config --global credential.helper store"
    Write-Host "then repeat step 8 and this step again."
    exit 1
}
Write-Host "OK: git ls-remote succeeded -- Basic auth against $RepoHost is working." -ForegroundColor Green

Write-Host "`n=== Step 10: Confirm the LFS endpoint resolves correctly ===" -ForegroundColor Cyan
git lfs env | Select-String -Pattern "Endpoint"

Write-Host "`n=== Step 11: Attempt the actual push ===" -ForegroundColor Cyan
git lfs push origin main
if ($LASTEXITCODE -eq 0) {
    Write-Host "`nSUCCESS." -ForegroundColor Green
} else {
    Write-Host "`nStill failing after every known cause has been eliminated." -ForegroundColor Red
    Write-Host "At this point the remaining candidates are outside local config: HF-side repo permissions"
    Write-Host "(e.g. token scope not covering write, or repo under an organization requiring separate access),"
    Write-Host "or a mismatch between the repo type assumed here (model) and its actual type (dataset/space)."
    Write-Host "Confirm the exact repo URL and type in a browser before further changes."
}
