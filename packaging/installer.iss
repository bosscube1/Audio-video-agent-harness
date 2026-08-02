; Inno Setup installer for the Gemini Live Agent (onedir bundle).
;
; Prereqs:
;   1. Build the onedir bundle:
;        .venv\Scripts\pyinstaller packaging\gemini-live-agent.spec --clean --noconfirm
;   2. Install Inno Setup 6 (https://jrsoftware.org/isinfo.php)
;   3. Compile:
;        "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\installer.iss
;
; Output: dist\installer\gemini-live-agent-setup-0.2.0.exe

#define AppName "Gemini Live Agent"
#define AppVersion "0.2.0"
#define AppExeName "gemini-live-agent.exe"
#define BundleDir "..\dist\gemini-live-agent"

[Setup]
AppId={{7C4E9B2A-3F1D-4E6A-9B8C-2D5F7A1E3C90}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=Gemini Live Agent
DefaultDirName={autopf}\GeminiLiveAgent
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputDir=..\dist\installer
OutputBaseFilename=gemini-live-agent-setup-{#AppVersion}
SetupIconFile=icon.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
; 64-bit only (PortAudio DLLs are x64).
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Per-machine install under Program Files; state lives in %LOCALAPPDATA%.
PrivilegesRequired=admin

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "{#BundleDir}\{#AppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#BundleDir}\_internal\*"; DestDir: "{app}\_internal"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Parameters: "--no-headless"
Name: "{group}\{#AppName} (terminal)"; Filename: "{app}\{#AppExeName}"; Parameters: "--headless"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Parameters: "--no-headless"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Run]
Filename: "{app}\{#AppExeName}"; Parameters: "--no-headless"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent
