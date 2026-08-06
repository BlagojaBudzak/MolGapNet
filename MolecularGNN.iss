[Setup]
AppName=MolecularGNN
AppVersion=1.0.0
AppPublisher=Blagoja Budjakoski
DefaultDirName={autopf}\MolecularGNN
DefaultGroupName=MolecularGNN
OutputDir=dist
OutputBaseFilename=MolecularGNN_Setup_1.0.0
Compression=lzma2/ultra64
SolidCompression=yes
SetupIconFile=icon.ico
WizardStyle=modern
; Allow running without admin rights
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
; Bundle the entire PyInstaller output folder
Source: "C:\Users\User\Desktop\GNN\dist\MolecularGNN\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\MolecularGNN";       Filename: "{app}\MolecularGNN.exe"
Name: "{group}\Uninstall";        Filename: "{uninstallexe}"
Name: "{userdesktop}\MolecularGNN"; Filename: "{app}\MolecularGNN.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\MolecularGNN.exe"; Description: "Launch MolecularGNN"; Flags: nowait postinstall skipifsilent