[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{40}$')][string]$Commit,
    [ValidateSet('Check','Brazil','Synthetic','Both')][string]$Study = 'Both',
    [string]$OutputRoot
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$tag = 'prereg-v2-2026-11'
$repoUrl = 'https://github.com/stanescu2022-png/credit-risk-fairness.git'
Push-Location -LiteralPath $root
try {
    # Only Git communicates with GitHub, before the local data-processing stages.
    git fetch --no-tags $repoUrl $Commit
    if ($LASTEXITCODE -ne 0) { throw 'Could not fetch the public freeze commit.' }
    $python = Join-Path $root '.venv/Scripts/python.exe'
    $launcher = @('-S','-m','research.model_selection_v2.confirmation_v2_jobs.offline',
                  'research.model_selection_v2.confirmation_v2_jobs.registered')
    $tagRef = "refs/tags/$tag"
    $remote = @(git ls-remote --tags $repoUrl $tagRef)
    if ($LASTEXITCODE -ne 0) { throw 'Could not check the public freeze tag.' }
    if ($remote.Count -eq 0) {
        & $python @launcher check --commit $Commit
        if ($LASTEXITCODE -ne 0) { throw 'Registered inputs or timing evidence failed verification.' }
        git show-ref --verify --quiet $tagRef
        if ($LASTEXITCODE -eq 1) {
            git tag -a $tag $Commit -m 'Public preregistration for Brazil and synthetic studies'
            if ($LASTEXITCODE -ne 0) { throw 'Could not create the local freeze tag.' }
        }
        $localTarget = git rev-parse "$tagRef^{commit}"
        if ($LASTEXITCODE -ne 0 -or $localTarget -ne $Commit) { throw 'Local freeze tag targets a different commit.' }
        git push $repoUrl "${tagRef}:$tagRef"
        if ($LASTEXITCODE -ne 0) { throw 'Could not publish the freeze tag.' }
    }
    git fetch --no-tags $repoUrl "${tagRef}:$tagRef"
    if ($LASTEXITCODE -ne 0) { throw 'Could not fetch the public freeze tag.' }
    $target = git rev-parse "$tagRef^{commit}"
    if ($LASTEXITCODE -ne 0 -or $target -ne $Commit) { throw 'Freeze tag targets a different commit.' }
    & $python @launcher check
    if ($LASTEXITCODE -ne 0) { throw 'Registered inputs or timing evidence failed verification.' }
    if ($Study -eq 'Both' -or $Study -eq 'Brazil') {
        $argsBrazil = @('brazil')
        if ($OutputRoot) { $argsBrazil += @('--output', (Join-Path $OutputRoot 'brazil_registered_01')) }
        & $python @launcher @argsBrazil
        if ($LASTEXITCODE -ne 0) { throw 'Brazil stopped. See its local output and worker logs.' }
    }
    if ($Study -eq 'Both' -or $Study -eq 'Synthetic') {
        $argsSynthetic = @('synthetic')
        if ($OutputRoot) { $argsSynthetic += @('--output', (Join-Path $OutputRoot 'synthetic_registered_01')) }
        & $python @launcher @argsSynthetic
        if ($LASTEXITCODE -ne 0) { throw 'Synthetic production or report replay stopped.' }
    }
} finally {
    Pop-Location
}
