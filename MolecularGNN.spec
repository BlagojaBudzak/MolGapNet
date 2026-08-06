# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['inference_gui.py'],
    pathex=[],
    binaries=[],
    datas=[('best_model.pt', '.'), ('scaler.pkl', '.'), ('molecular_gnn_blueprint.py', '.'), ('app_config.json', '.')],
    hiddenimports=['torch', 'torch.nn', 'torch.nn.functional', 'torch_geometric', 'torch_geometric.nn', 'torch_geometric.data', 'torch_geometric.loader', 'rdkit', 'rdkit.Chem', 'rdkit.Chem.Draw', 'rdkit.Chem.rdMolDescriptors', 'sklearn.preprocessing', 'sklearn.utils._cython_blas', 'sklearn.neighbors._partition_nodes', 'PIL', 'PIL.Image', 'PIL.ImageTk', 'pickle', 'json'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MolecularGNN',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['icon.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MolecularGNN',
)
