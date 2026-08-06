"""
=============================================================================
BUILD SCRIPT — Molecular GNN Desktop App
=============================================================================

Packages inference_gui.py + molecular_gnn_blueprint.py + best_model.pt +
scaler.pkl into a standalone Windows installer using PyInstaller + Inno Setup.

PREREQUISITES (run once):
    pip install pyinstaller
    Download and install Inno Setup from https://jrsoftware.org/isinfo.php
    Default install path: C:\\Program Files (x86)\\Inno Setup 6\\ISCC.exe

USAGE:
    python build.py
    python build.py --target lumo --version 1.1.0

OUTPUT:
    dist/MolecularGNN_Setup.exe   ← send this to anyone on Windows
"""
import torch
import argparse
import json
import os
import shutil
import subprocess
import sys

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these if your paths differ
# ─────────────────────────────────────────────────────────────────────────────

APP_NAME     = "MolecularGNN"
APP_VERSION  = "1.0.0"
AUTHOR       = "Blagoja Budjakoski"
INNO_SETUP   = r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe"  # adjust if needed

# Files that must exist before building
REQUIRED_FILES = [
    "inference_gui.py",
    "molecular_gnn_blueprint.py",
    "best_model.pt",
    "scaler.pkl",
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def check_required_files():
    missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
    if missing:
        print("ERROR: Missing required files:")
        for f in missing:
            print(f"  ✗ {f}")
        print("\nMake sure you run this script from the GNN/ folder")
        print("and that best_model.pt + scaler.pkl exist (train first).")
        sys.exit(1)
    print("Required files: all present ✓")


def check_pyinstaller():
    try:
        import PyInstaller
        print(f"PyInstaller {PyInstaller.__version__} ✓")
    except ImportError:
        print("PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])


def write_app_config(target: str, version: str):
    """
    Write a small JSON config that the packaged app reads at startup.
    This lets the app know which QM9 target it was built for,
    and where to find the model files relative to the install dir.
    """
    config = {
        "app_name":    APP_NAME,
        "version":     version,
        "target_key":  target,
        "model_file":  "best_model.pt",
        "scaler_file": "scaler.pkl",
        "hidden_dim":  128,
    }
    with open("app_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Written app_config.json  (target={target}, version={version})")


def write_inno_script(version: str):
    """Generate the Inno Setup .iss script that creates the Windows installer."""
    dist_app_dir = os.path.abspath(f"dist/{APP_NAME}")
    script = f"""
[Setup]
AppName={APP_NAME}
AppVersion={version}
AppPublisher={AUTHOR}
DefaultDirName={{autopf}}\\{APP_NAME}
DefaultGroupName={APP_NAME}
OutputDir=dist
OutputBaseFilename={APP_NAME}_Setup_{version}
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
Source: "{dist_app_dir}\\*"; DestDir: "{{app}}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{{group}}\\{APP_NAME}";       Filename: "{{app}}\\{APP_NAME}.exe"
Name: "{{group}}\\Uninstall";        Filename: "{{uninstallexe}}"
Name: "{{userdesktop}}\\{APP_NAME}"; Filename: "{{app}}\\{APP_NAME}.exe"; Tasks: desktopicon

[Run]
Filename: "{{app}}\\{APP_NAME}.exe"; Description: "Launch {APP_NAME}"; Flags: nowait postinstall skipifsilent
"""
    iss_path = f"{APP_NAME}.iss"
    with open(iss_path, "w") as f:
        f.write(script.strip())
    print(f"Written {iss_path}")
    return iss_path


def write_icon_if_missing():
    """
    Create a minimal valid .ico file if none exists.
    Replace icon.ico with your own 256x256 icon for a polished installer.
    """
    if os.path.exists("icon.ico"):
        return
    try:
        from PIL import Image, ImageDraw
        size = 256
        img  = Image.new("RGBA", (size, size), (15, 17, 23, 255))
        draw = ImageDraw.Draw(img)
        # Simple hexagon to match the ⬡ in the GUI title
        cx, cy, r = size//2, size//2, size//2 - 20
        import math
        pts = [(cx + r*math.cos(math.radians(60*i - 30)),
                cy + r*math.sin(math.radians(60*i - 30))) for i in range(6)]
        draw.polygon(pts, outline=(0, 200, 224, 255), width=12)
        img.save("icon.ico", format="ICO", sizes=[(256,256),(128,128),(64,64),(32,32),(16,16)])
        print("Generated icon.ico ✓  (replace with your own for a custom look)")
    except Exception as e:
        print(f"Could not generate icon: {e} — continuing without one")
        # Remove SetupIconFile line from ISS if no icon
        with open(f"{APP_NAME}.iss", "r") as f:
            content = f.read()
        with open(f"{APP_NAME}.iss", "w") as f:
            f.write(content.replace("SetupIconFile=icon.ico\n", ""))


def run_pyinstaller(version: str):
    """Run PyInstaller to produce the dist/MolecularGNN/ folder."""
    print("\n── Running PyInstaller ──────────────────────────────────────────")

    # Hidden imports PyInstaller misses for torch / rdkit / pyg
    hidden = [
        "torch",
        "torch.nn",
        "torch.nn.functional",
        "torch_geometric",
        "torch_geometric.nn",
        "torch_geometric.data",
        "torch_geometric.loader",
        "rdkit",
        "rdkit.Chem",
        "rdkit.Chem.Draw",
        "rdkit.Chem.rdMolDescriptors",
        "sklearn.preprocessing",
        "sklearn.utils._cython_blas",
        "sklearn.neighbors._partition_nodes",
        "PIL",
        "PIL.Image",
        "PIL.ImageTk",
        "pickle",
        "json",
    ]

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name",          APP_NAME,
        "--onedir",        # folder mode — faster startup than --onefile for large models
        "--windowed",      # no console window
        "--noconfirm",     # overwrite previous build without asking
        "--clean",
        "--icon",          "icon.ico",
        # Bundle the model, scaler, blueprint, and config alongside the exe
        "--add-data",      f"best_model.pt{os.pathsep}.",
        "--add-data",      f"scaler.pkl{os.pathsep}.",
        "--add-data",      f"molecular_gnn_blueprint.py{os.pathsep}.",
        "--add-data",      f"app_config.json{os.pathsep}.",
    ]

    for h in hidden:
        cmd += ["--hidden-import", h]

    cmd.append("inference_gui.py")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("\nERROR: PyInstaller failed. Check output above.")
        sys.exit(1)
    print(f"\nPyInstaller done → dist/{APP_NAME}/")


def run_inno_setup(iss_path: str):
    """Run Inno Setup compiler to produce the installer .exe."""
    if not os.path.exists(INNO_SETUP):
        print(f"\nInno Setup not found at:\n  {INNO_SETUP}")
        print("Either install it from https://jrsoftware.org/isinfo.php")
        print(f"or update INNO_SETUP path in build.py")
        print(f"\nThe packaged app folder is still at: dist/{APP_NAME}/")
        print("You can zip and share that folder without the installer.")
        return False

    print("\n── Running Inno Setup ───────────────────────────────────────────")
    result = subprocess.run([INNO_SETUP, iss_path])
    if result.returncode != 0:
        print("ERROR: Inno Setup failed.")
        return False

    installer = None
    for f in os.listdir("dist"):
        if f.endswith(".exe") and "Setup" in f:
            installer = os.path.join("dist", f)
            break

    if installer:
        size_mb = os.path.getsize(installer) / 1024 / 1024
        print(f"\n✓ Installer created: {installer}  ({size_mb:.0f} MB)")
        print("  Send this single file to anyone on Windows.")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build MolecularGNN desktop installer")
    parser.add_argument("--target",  default="gap",   help="QM9 target the model was trained on")
    parser.add_argument("--version", default=APP_VERSION, help="App version string")
    parser.add_argument("--skip-inno", action="store_true",
                        help="Stop after PyInstaller (skip Inno Setup step)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  Building {APP_NAME} v{args.version}  (target: {args.target})")
    print("=" * 60 + "\n")

    check_required_files()
    check_pyinstaller()
    write_app_config(args.target, args.version)
    iss_path = write_inno_script(args.version)
    write_icon_if_missing()
    run_pyinstaller(args.version)

    if not args.skip_inno:
        run_inno_setup(iss_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
