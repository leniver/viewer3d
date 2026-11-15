; Inno Setup 6 script for 3D Viewer.
; Built by installer/build.ps1, which stages the files and passes the version;
; compiling this file directly fails on purpose.

#ifndef AppVersion
  #error AppVersion is not defined: build with installer/build.ps1
#endif
#ifndef AppVersionNumeric
  #define AppVersionNumeric AppVersion
#endif
#ifndef StageDir
  #error StageDir is not defined: build with installer/build.ps1
#endif
#ifndef OutputDir
  #define OutputDir "..\target\dist"
#endif

#define AppName "3D Viewer"
#define AppExe "viewer3d.exe"
#define AppUrl "https://github.com/leniver/viewer3d"

[Setup]
; Never change AppId: it identifies the installation for upgrades and uninstall.
AppId={{65518FBE-3B72-4F28-BA81-E2777ED34D65}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=leniver
AppPublisherURL={#AppUrl}
AppSupportURL={#AppUrl}/issues
AppUpdatesURL={#AppUrl}/releases
AppCopyright=Copyright (c) 2026 leniver
VersionInfoVersion={#AppVersionNumeric}
VersionInfoProductTextVersion={#AppVersion}
; Per-user install by default (no administrator rights, %LOCALAPPDATA%\Programs);
; the first page offers "all users" (Program Files), as does /ALLUSERS.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog commandline
DefaultDirName={autopf}\{#AppName}
DisableProgramGroupPage=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; wgpu uses Direct3D 12, available from Windows 10.
MinVersion=10.0
WizardStyle=modern
SetupIconFile=..\assets\icon.ico
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName}
; Upgrades close a running viewer (Restart Manager) instead of failing.
CloseApplications=yes
RestartApplications=no
SetupLogging=yes
OutputDir={#OutputDir}
OutputBaseFilename=viewer3d-{#AppVersion}-setup
Compression=lzma2/ultra64
SolidCompression=yes

[Tasks]
Name: "openwith"; Description: "Add 3D Viewer to ""Open with"" for 3D files (glTF, FBX, OBJ, STL, PLY, 3MF, STEP, IGES)"; GroupDescription: "File types:"
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; The staging folder holds exactly what the portable zip contains.
Source: "{#StageDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
; The viewer writes its own "Open with" entries (src/platform.rs), so the list of
; file types has a single source. It registers for the user who started Setup,
; also when Setup was elevated for an all-users install.
Filename: "{app}\{#AppExe}"; Parameters: "--register"; Tasks: openwith; Flags: runhidden waituntilterminated runasoriginaluser; StatusMsg: "Registering the file types..."
; Unticking the task on an upgrade removes an earlier registration.
Filename: "{app}\{#AppExe}"; Parameters: "--unregister"; Tasks: not openwith; Flags: runhidden waituntilterminated runasoriginaluser
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent runasoriginaluser

[UninstallRun]
Filename: "{app}\{#AppExe}"; Parameters: "--unregister"; Flags: runhidden waituntilterminated; RunOnceId: "UnregisterOpenWith"
