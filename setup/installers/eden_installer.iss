; Eden Trading System - Inno Setup Script
; This script creates a Windows installer for Eden
; Supports both full installation and portable mode

#define MyAppName "Eden Trading System"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Eden Trading"
#define MyAppURL "https://eden.trading"
#define MyAppExeName "Eden.exe"
#define MyAppDescription "Professional AI-Powered Trading System"

[Setup]
; NOTE: The value of AppId uniquely identifies this application.
AppId={{8F2B9B6C-9A4E-4B3F-A1D5-2E7C8D9F1A3B}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
LicenseFile=..\..\docs\LICENSE.txt
OutputDir=..\..\build
OutputBaseFilename=EdenSetup-{#MyAppVersion}
SetupIconFile=..\..\ui\resources\icons\eden-icon.ico
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
WizardImageFile=setup_banner.bmp
WizardSmallImageFile=setup_icon.bmp
ArchitecturesInstallIn64BitMode=x64
MinVersion=10.0
PrivilegesRequired=admin
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName}
AppReadmeFile={app}\README.txt

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Types]
Name: "full"; Description: "Full Installation (Recommended)"
Name: "portable"; Description: "Portable Installation"
Name: "custom"; Description: "Custom Installation"; Flags: iscustom

[Components]
Name: "core"; Description: "Eden Core Application"; Types: full portable custom; Flags: fixed
Name: "python"; Description: "Python Runtime & Workers"; Types: full custom; Flags: checkablealone
Name: "gpu"; Description: "GPU Acceleration Libraries"; Types: full custom; Flags: checkablealone  
Name: "samples"; Description: "Sample Data & Strategies"; Types: full custom; Flags: checkablealone
Name: "docs"; Description: "Documentation"; Types: full custom; Flags: checkablealone
Name: "shortcuts"; Description: "Desktop & Start Menu Shortcuts"; Types: full; Flags: checkablealone

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Components: shortcuts
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; OnlyBelowVersion: 6.1; Components: shortcuts
Name: "firewall"; Description: "Configure Windows Firewall for Eden"; GroupDescription: "Security"; Flags: unchecked
Name: "envpath"; Description: "Add Eden to system PATH"; GroupDescription: "Advanced"; Flags: unchecked

[Files]
; Core Application Files
Source: "..\..\build\Eden.exe"; DestDir: "{app}"; Components: core; Flags: ignoreversion
Source: "..\..\build\*.dll"; DestDir: "{app}"; Components: core; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\..\ui\qml\*"; DestDir: "{app}\qml"; Components: core; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\..\ui\resources\*"; DestDir: "{app}\resources"; Components: core; Flags: ignoreversion recursesubdirs createallsubdirs

; Python Worker & Runtime
Source: "..\..\worker\python\*"; DestDir: "{app}\worker\python"; Components: python; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "python-embed\*"; DestDir: "{app}\python"; Components: python; Flags: ignoreversion recursesubdirs createallsubdirs; Check: not IsTaskSelected('portable')

; GPU Libraries
Source: "gpu-libs\onnxruntime\*"; DestDir: "{app}\gpu\onnxruntime"; Components: gpu; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "gpu-libs\directml\*"; DestDir: "{app}\gpu\directml"; Components: gpu; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "gpu-libs\cuda\*"; DestDir: "{app}\gpu\cuda"; Components: gpu; Flags: ignoreversion recursesubdirs createallsubdirs

; Shared Data
Source: "..\..\shared\*"; DestDir: "{app}\shared"; Components: core; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\..\data\*"; DestDir: "{app}\data"; Components: samples; Flags: ignoreversion recursesubdirs createallsubdirs

; Documentation
Source: "..\..\docs\*"; DestDir: "{app}\docs"; Components: docs; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\..\README.md"; DestDir: "{app}"; DestName: "README.txt"; Components: core; Flags: ignoreversion
Source: "..\..\docs\LICENSE.txt"; DestDir: "{app}"; Components: core; Flags: ignoreversion

; Configuration Files
Source: "..\..\worker\python\config.yml"; DestDir: "{app}"; Components: core; Flags: ignoreversion onlyifdoesntexist
Source: "..\..\worker\python\.env.example"; DestDir: "{app}"; Components: core; Flags: ignoreversion

; Portable Mode Files
Source: "portable\Eden-Portable.bat"; DestDir: "{app}"; Components: core; Check: IsTaskSelected('portable')
Source: "portable\portable.ini"; DestDir: "{app}"; Components: core; Check: IsTaskSelected('portable')

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Components: shortcuts
Name: "{group}\Eden Documentation"; Filename: "{app}\docs\index.html"; Components: shortcuts and docs
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"; Components: shortcuts
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: quicklaunchicon

[Registry]
; File associations for .eden project files
Root: HKCR; Subkey: ".eden"; ValueType: string; ValueName: ""; ValueData: "EdenProject"; Flags: uninsdeletevalue; Components: core
Root: HKCR; Subkey: "EdenProject"; ValueType: string; ValueName: ""; ValueData: "{#MyAppName} Project"; Flags: uninsdeletekey; Components: core  
Root: HKCR; Subkey: "EdenProject\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#MyAppExeName},0"; Components: core
Root: HKCR; Subkey: "EdenProject\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""; Components: core

; Add to PATH if requested
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "PATH"; ValueData: "{olddata};{app}"; Tasks: envpath; Check: not IsValueEmpty('PATH')
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "PATH"; ValueData: "{app}"; Tasks: envpath; Check: IsValueEmpty('PATH')

[Run]
; Run Eden after installation
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent; Check: not IsTaskSelected('portable')

; Configure Windows Firewall
Filename: "netsh"; Parameters: "advfirewall firewall add rule name=""Eden Trading System"" dir=in action=allow program=""{app}\{#MyAppExeName}"""; Tasks: firewall; Flags: runhidden

; Initialize data directories
Filename: "{app}\{#MyAppExeName}"; Parameters: "--init-workspace"; Flags: runhidden; Components: core

[UninstallRun]
; Remove firewall rules
Filename: "netsh"; Parameters: "advfirewall firewall delete rule name=""Eden Trading System"""; Flags: runhidden; RunOnceId: "RemoveFirewallRule"

[Code]
function IsValueEmpty(ValueName: String): Boolean;
var
  Value: String;
begin
  Result := not RegQueryStringValue(HKCU, 'Environment', ValueName, Value) or (Value = '');
end;

function GetPythonPath(): String;
var
  PythonPath: String;
begin
  Result := '';
  if RegQueryStringValue(HKLM, 'SOFTWARE\Python\PythonCore\3.11\InstallPath', '', PythonPath) or
     RegQueryStringValue(HKLM, 'SOFTWARE\Python\PythonCore\3.10\InstallPath', '', PythonPath) or
     RegQueryStringValue(HKLM, 'SOFTWARE\Python\PythonCore\3.9\InstallPath', '', PythonPath) then
  begin
    Result := PythonPath;
  end;
end;

function InitializeSetup(): Boolean;
begin
  Result := True;
  
  // Check minimum Windows version
  if GetWindowsVersion < $0A000000 then
  begin
    MsgBox('Eden Trading System requires Windows 10 or later.', mbError, MB_OK);
    Result := False;
  end;
end;

procedure InitializeWizard();
var
  PythonPath: String;
begin
  // Check for existing Python installation
  PythonPath := GetPythonPath();
  if PythonPath <> '' then
  begin
    Log('Found Python installation at: ' + PythonPath);
  end else
  begin
    Log('No system Python installation found - will use embedded Python');
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
begin
  if CurStep = ssPostInstall then
  begin
    // Install Python dependencies if Python component is selected
    if IsComponentSelected('python') then
    begin
      Exec(ExpandConstant('{app}\python\python.exe'), '-m pip install -r "' + ExpandConstant('{app}\worker\python\requirements.txt') + '"', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
    end;
    
    // Set permissions for data directories
    Exec('icacls', '"' + ExpandConstant('{app}\data') + '" /grant Users:(OI)(CI)F /T', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
    Exec('icacls', '"' + ExpandConstant('{app}\shared') + '" /grant Users:(OI)(CI)F /T', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  end;
end;

function ShouldSkipPage(PageID: Integer): Boolean;
begin
  Result := False;
  
  // Skip license page in portable mode
  if (PageID = wpLicense) and IsTaskSelected('portable') then
    Result := True;
    
  // Skip directory selection in portable mode
  if (PageID = wpSelectDir) and IsTaskSelected('portable') then
    Result := True;
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  // Customize component selection for portable mode
  if (CurPageID = wpSelectComponents) and IsTaskSelected('portable') then
  begin
    WizardForm.ComponentsList.Checked[1] := False; // Disable Python component for portable
    WizardForm.ComponentsList.ItemEnabled[1] := False;
  end;
end;