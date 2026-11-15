<#
.SYNOPSIS
Builds the Windows release assets of 3D Viewer.

.DESCRIPTION
Writes to target\dist:
  viewer3d-<version>-setup.exe          installer (Inno Setup 6)
  viewer3d-<version>-windows-x64.zip    portable version
  SHA256SUMS.txt                        checksums of both
and to target\package the staged files and release-notes.md (the CHANGELOG
section of this version plus installation notes).

Requires Inno Setup 6 (ISCC.exe) and, unless -SkipNotices, cargo-about.
Runs locally and in .github\workflows\release.yml; works with Windows
PowerShell 5.1 and PowerShell 7.

.PARAMETER SkipBuild
Package the existing target\release\viewer3d.exe instead of building it.

.PARAMETER SkipNotices
Leave out the third-party licence notices (local test builds only: a release
must ship them).

.PARAMETER Iscc
Path to ISCC.exe. Default: the ISCC environment variable, the PATH, then the
usual Inno Setup install folders.
#>
[CmdletBinding()]
param(
    [switch]$SkipBuild,
    [switch]$SkipNotices,
    [string]$Iscc
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$root = Split-Path -Parent $PSScriptRoot

function Invoke-Native {
    param([Parameter(Mandatory)][string]$Exe, [string[]]$Arguments = @())
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "'$Exe $($Arguments -join ' ')' failed with exit code $LASTEXITCODE" }
}

function Find-Iscc {
    if ($Iscc) {
        if (-not (Test-Path -LiteralPath $Iscc)) { throw "ISCC.exe not found at '$Iscc'" }
        return $Iscc
    }
    if ($env:ISCC -and (Test-Path -LiteralPath $env:ISCC)) { return $env:ISCC }
    $cmd = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    # Install folder recorded by the Inno Setup installer (per-user or per-machine).
    $keys = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1',
            'HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1',
            'HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1'
    foreach ($key in $keys) {
        $item = Get-ItemProperty -LiteralPath $key -ErrorAction SilentlyContinue
        if ($item -and $item.PSObject.Properties['InstallLocation']) {
            $candidate = Join-Path $item.InstallLocation 'ISCC.exe'
            if (Test-Path -LiteralPath $candidate) { return $candidate }
        }
    }
    foreach ($dir in @(${env:ProgramFiles(x86)}, $env:ProgramFiles, (Join-Path $env:LOCALAPPDATA 'Programs'))) {
        if (-not $dir) { continue }
        $candidate = Join-Path $dir 'Inno Setup 6\ISCC.exe'
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    throw 'Inno Setup 6 (ISCC.exe) not found: install it (winget install JRSoftware.InnoSetup) or pass -Iscc'
}

# Returns the CHANGELOG.md section of a version, without its heading.
function Get-ChangelogSection {
    param([string]$Path, [string]$Version)
    $lines = @(Get-Content -LiteralPath $Path -Encoding UTF8)
    $heading = '^## \[' + [regex]::Escape($Version) + '\]'
    $start = -1
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match $heading) { $start = $i + 1; break }
    }
    if ($start -lt 0) { throw "CHANGELOG.md has no section for $Version (expected a '## [$Version] - date' heading)" }
    $section = New-Object System.Collections.Generic.List[string]
    for ($i = $start; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match '^## \[' -or $lines[$i] -match '^\[[^\]]+\]: ') { break }
        $section.Add($lines[$i])
    }
    return ($section -join "`n").Trim()
}

function Write-Utf8NoBom {
    param([string]$Path, [string]$Text)
    [System.IO.File]::WriteAllText($Path, $Text, (New-Object System.Text.UTF8Encoding($false)))
}

Push-Location $root
try {
    $metadata = Invoke-Native cargo @('metadata', '--format-version', '1', '--no-deps', '--locked') | Out-String | ConvertFrom-Json
    $package = $metadata.packages | Where-Object { $_.name -eq 'viewer3d' }
    if (-not $package) { throw 'package viewer3d not found in cargo metadata' }
    $version = $package.version
    # Windows version resources only take numbers (0.2.0-beta.1 -> 0.2.0).
    $numericVersion = ($version -split '[-+]')[0]
    Write-Host "3D Viewer $version"

    if (-not $SkipBuild) {
        Invoke-Native cargo @('build', '--release', '--locked')
    }
    $exe = Join-Path $root 'target\release\viewer3d.exe'
    if (-not (Test-Path -LiteralPath $exe)) { throw "$exe not found: build first or drop -SkipBuild" }

    $name = "viewer3d-$version-windows-x64"
    $packageDir = Join-Path $root 'target\package'
    $stage = Join-Path $packageDir $name
    $dist = Join-Path $root 'target\dist'
    foreach ($dir in @($stage, $dist)) {
        if (Test-Path -LiteralPath $dir) { Remove-Item -LiteralPath $dir -Recurse -Force }
        New-Item -ItemType Directory -Path $dir | Out-Null
    }

    # Staged files: what the zip contains and the installer installs.
    Copy-Item -LiteralPath $exe -Destination $stage
    Copy-Item -LiteralPath (Join-Path $root 'LICENSE') -Destination (Join-Path $stage 'LICENSE.txt')
    Copy-Item -LiteralPath (Join-Path $root 'README.md') -Destination $stage
    # Open CASCADE is linked statically: its LGPL 2.1 text and exception must ship
    # with the binary. The build copies them to target\OCCT.
    $occt = Join-Path $root 'target\OCCT'
    $occtTexts = @{ 'LICENSE_LGPL_21.txt' = 'OpenCASCADE-LGPL-2.1.txt'; 'OCCT_LGPL_EXCEPTION.txt' = 'OpenCASCADE-LGPL-exception.txt' }
    foreach ($source in $occtTexts.Keys) {
        $path = Join-Path $occt $source
        if (-not (Test-Path -LiteralPath $path)) { throw "$path not found (it comes with the Open CASCADE build)" }
        Copy-Item -LiteralPath $path -Destination (Join-Path $stage $occtTexts[$source])
    }
    if ($SkipNotices) {
        Write-Warning 'third-party licence notices skipped: do not publish this build'
    } else {
        $notices = Join-Path $stage 'THIRD-PARTY-LICENSES.html'
        Invoke-Native cargo @('about', 'generate', '--workspace', '--locked', '--fail', '--config', 'installer/about.toml',
                              '--output-file', $notices, 'installer/about.hbs')
    }

    # Installer.
    $isccExe = Find-Iscc
    Write-Host "Inno Setup: $isccExe"
    Invoke-Native $isccExe @('/Qp', "/DAppVersion=$version", "/DAppVersionNumeric=$numericVersion",
                             "/DStageDir=$stage", "/DOutputDir=$dist", (Join-Path $root 'installer\viewer3d.iss'))

    # Portable zip with one top-level folder, so extracting does not scatter
    # files. Written directly: Compress-Archive stores '\' path separators,
    # which the zip format forbids (other tools then create 'a\b' file names).
    Add-Type -AssemblyName System.IO.Compression, System.IO.Compression.FileSystem
    $zip = Join-Path $dist "$name.zip"
    $archive = [System.IO.Compression.ZipFile]::Open($zip, [System.IO.Compression.ZipArchiveMode]::Create)
    try {
        Get-ChildItem -LiteralPath $stage -File -Recurse | Sort-Object FullName | ForEach-Object {
            $relative = $_.FullName.Substring($stage.Length).TrimStart('\') -replace '\\', '/'
            [void][System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $archive, $_.FullName, "$name/$relative", [System.IO.Compression.CompressionLevel]::Optimal)
        }
    } finally {
        $archive.Dispose()
    }

    # Checksums (sha256sum format, LF line endings).
    $sums = Get-ChildItem -LiteralPath $dist -File | Sort-Object Name | ForEach-Object {
        '{0}  {1}' -f (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant(), $_.Name
    }
    Write-Utf8NoBom (Join-Path $dist 'SHA256SUMS.txt') (($sums -join "`n") + "`n")

    # Release notes: CHANGELOG section + how to install.
    $changes = Get-ChangelogSection (Join-Path $root 'CHANGELOG.md') $version
    $install = @"
### Installing (Windows 10/11, 64-bit)

- **viewer3d-$version-setup.exe**: installs for the current user without administrator rights (all users can be chosen on the first page), adds a Start menu shortcut and, optionally, the "Open with" entry for 3D files. Uninstall from Settings > Apps.
- **$name.zip**: portable version, nothing to install: extract it and run ``viewer3d.exe``.

The files are not code-signed yet, so Windows SmartScreen may show a warning on first launch: choose *More info* > *Run anyway*. ``SHA256SUMS.txt`` lists the SHA-256 checksums.
"@
    Write-Utf8NoBom (Join-Path $packageDir 'release-notes.md') ($changes + "`n`n" + ($install -replace "`r`n", "`n").Trim() + "`n")

    Write-Host ''
    Get-ChildItem -LiteralPath $dist -File | ForEach-Object { Write-Host ('{0,-40} {1,8:N1} MB' -f $_.Name, ($_.Length / 1MB)) }
} finally {
    Pop-Location
}
