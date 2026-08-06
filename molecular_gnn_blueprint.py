"""
=============================================================================
MOLECULAR PROPERTY PREDICTION WITH GNN / MPNN
Blueprint: RDKit + PyTorch Geometric  ·  QM9-ready
=============================================================================

PIPELINE OVERVIEW
-----------------
qm9.csv  (SMILES1, gap, mu, alpha, ...)
  └─► load_qm9()          — picks SMILES col, picks target col, drops bad rows
        └─► run_training() — normalises, splits, featurises, trains, evaluates
              └─► smiles_to_graph()   — RDKit → PyG Data object per molecule
                    └─► MolecularDataset / DataLoader
                          └─► MPNNRegressor (or GCN / GAT)
                                └─► predict()  — new SMILES → float property

HOW TO RUN
----------
  1. Put qm9.csv in the same folder as this file.
  2. Edit the two lines under "YOUR TWO KNOBS" at the bottom.
  3. python molecular_gnn_blueprint.py

SWITCHING TO YOUR OWN DATASET
------------------------------
  Replace the load_qm9() call with:
      df = pd.read_csv("my_data.csv")[["smiles_col", "target_col"]]
      df.columns = ["smiles", "target"]
  Then pass df directly to run_training().
  Nothing else changes.

DEPENDENCIES
------------
    pip install torch torchvision
    pip install torch-geometric
    pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-<VERSION>+cpu.html
    pip install rdkit-pypi pandas scikit-learn tqdm
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import (
    NNConv,            # edge-conditioned conv — core of MPNN (Gilmer et al. 2017)
    GCNConv,           # classic GCN (simpler baseline, ignores edge features)
    GATv2Conv,         # graph attention (good for atom-level interpretability)
    global_mean_pool,
    global_add_pool,
)

from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle, os


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — ATOM & BOND FEATURIZERS
# ─────────────────────────────────────────────────────────────────────────────

ATOM_TYPES    = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Si', 'Other']
HYBRIDISATION = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
CHIRALITY  = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

NUM_ATOM_FEATURES = 28   # atom_features() length  — do not change without updating models
NUM_BOND_FEATURES = 6    # bond_features() length


def one_hot(value, allowable_set):
    if value not in allowable_set:
        value = allowable_set[-1]
    return [int(value == s) for s in allowable_set]


def atom_features(atom):
    """28-dim vector per atom: element · hybridisation · chirality · charge · H · ring · aromatic"""
    return (
        one_hot(atom.GetSymbol(),        ATOM_TYPES)       # 11
        + one_hot(atom.GetHybridization(), HYBRIDISATION)  #  5
        + one_hot(atom.GetChiralTag(),     CHIRALITY)      #  3
        + [atom.GetAtomicNum() / 100.0]                    #  1
        + [atom.GetDegree() / 6.0]                         #  1
        + [atom.GetTotalValence() / 8.0]                   #  1
        + [atom.GetMass() / 200.0]                         #  1
        + [atom.GetTotalNumHs() / 4.0]                     #  1
        + [float(atom.GetFormalCharge())]                  #  1
        + [float(atom.GetNumRadicalElectrons())]           #  1
        + [float(atom.IsInRing())]                         #  1
        + [float(atom.GetIsAromatic())]                    #  1
    )                                                      # = 28


def bond_features(bond):
    """6-dim vector per bond: type · conjugated · ring"""
    return (
        one_hot(bond.GetBondType(), BOND_TYPES)   # 4
        + [float(bond.GetIsConjugated())]          # 1
        + [float(bond.IsInRing())]                 # 1
    )                                              # = 6


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — SMILES → torch_geometric.data.Data
# ─────────────────────────────────────────────────────────────────────────────

def smiles_to_graph(smiles: str, y: float):
    """
    RDKit mol  →  PyG Data object.

    data.x          [N_atoms, 23]      node feature matrix
    data.edge_index [2, N_bonds*2]     bidirectional COO adjacency
    data.edge_attr  [N_bonds*2, 6]     edge feature matrix
    data.y          [1]                regression target

    Returns None for unparseable SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    src, dst, eattr = [], [], []
    for bond in mol.GetBonds():
        i, j  = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feats = bond_features(bond)
        src  += [i, j]
        dst  += [j, i]
        eattr += [feats, feats]      # same features for both directions

    if not src:                      # single-atom molecule edge case
        src, dst, eattr = [0], [0], [[0.0] * NUM_BOND_FEATURES]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr  = torch.tensor(eattr,      dtype=torch.float)
    y_t        = torch.tensor([y],        dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y_t)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — DATASET CLASS
# ─────────────────────────────────────────────────────────────────────────────

class MolecularDataset(Dataset):
    """In-memory PyG dataset.  Expects a DataFrame with 'smiles' and 'target' columns."""

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.graphs = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="  Featurising"):
            g = smiles_to_graph(str(row["smiles"]), float(row["target"]))
            if g is not None:
                self.graphs.append(g)

    def len(self):         return len(self.graphs)
    def get(self, idx):    return self.graphs[idx]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — GNN ARCHITECTURES
# ─────────────────────────────────────────────────────────────────────────────

# ── 5a. GCN baseline (no edge features, fast sanity check) ───────────────────

class GCNRegressor(nn.Module):
    def __init__(self, in_channels=NUM_ATOM_FEATURES, hidden=128, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.head  = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, data):
        x, ei, b = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, ei))
        x = F.relu(self.conv2(x, ei))
        x = F.relu(self.conv3(x, ei))
        return self.head(global_mean_pool(x, b)).squeeze(-1)


# ── 5b. MPNN — RECOMMENDED  (uses bond features via edge-conditioned conv) ───

class MPNNRegressor(nn.Module):
    """
    Implements Gilmer et al. 2017 MPNN exactly.
    For each bond, a small MLP maps the 6-d bond vector into a (hidden×hidden)
    weight matrix, then multiplies the neighbour's hidden state through it.
    Bond type / aromaticity directly shape how information flows → physically meaningful.
    """

    def __init__(self, in_channels=NUM_ATOM_FEATURES, edge_dim=NUM_BOND_FEATURES,
                 hidden=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden)

        def edge_net():
            return nn.Sequential(
                nn.Linear(edge_dim, 64), nn.ReLU(), nn.Linear(64, hidden * hidden)
            )

        self.conv1 = NNConv(hidden, hidden, edge_net(), aggr='add')
        self.conv2 = NNConv(hidden, hidden, edge_net(), aggr='add')
        self.conv3 = NNConv(hidden, hidden, edge_net(), aggr='add')
        self.conv4 = NNConv(hidden, hidden, edge_net(), aggr='add')
        self.bn1   = nn.BatchNorm1d(hidden)
        self.bn2   = nn.BatchNorm1d(hidden)
        self.bn3   = nn.BatchNorm1d(hidden)
        self.bn4   = nn.BatchNorm1d(hidden)
        self.drop  = nn.Dropout(dropout)

        # Readout: cat(mean, sum) captures both average environment and molecule size
        self.head  = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 32),  nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, data):
        x, ei, ea, b = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.input_proj(x))
        h = x
        x = self.drop(F.relu(self.bn1(self.conv1(x, ei, ea))))
        x = x + h
        h = x
        x = self.drop(F.relu(self.bn2(self.conv2(x, ei, ea))))
        x = x + h
        h = x
        x = self.drop(F.relu(self.bn3(self.conv3(x, ei, ea))))
        x = x + h
        x = torch.cat(
            [global_mean_pool(x, b), global_add_pool(x, b)],
            dim=-1
        )
        return self.head(x).squeeze(-1)


# ── 5c. GAT (attention weights = interpretable atom importance) ───────────────

class GATRegressor(nn.Module):
    def __init__(self, in_channels=NUM_ATOM_FEATURES, hidden=64, heads=4, dropout=0.2):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels,    hidden,         heads=heads, dropout=dropout, concat=True)
        self.conv2 = GATv2Conv(hidden * heads, hidden,         heads=heads, dropout=dropout, concat=True)
        self.conv3 = GATv2Conv(hidden * heads, hidden,         heads=1,     dropout=dropout, concat=False)
        self.head  = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, data):
        x, ei, b = data.x, data.edge_index, data.batch
        x = F.elu(self.conv1(x, ei))
        x = F.elu(self.conv2(x, ei))
        x = F.elu(self.conv3(x, ei))
        return self.head(global_mean_pool(x, b)).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — TRAINING & EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        loss  = F.mse_loss(model(batch), batch.y.squeeze())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total += loss.item() * batch.num_graphs
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    for batch in loader:
        batch = batch.to(device)
        preds.append(model(batch).cpu())
        targets.append(batch.y.squeeze().cpu())
    p, t  = torch.cat(preds), torch.cat(targets)
    return torch.sqrt(F.mse_loss(p, t)).item(), F.l1_loss(p, t).item()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — run_training()  (accepts a ready DataFrame directly)
# ─────────────────────────────────────────────────────────────────────────────

def run_training(
    df: pd.DataFrame,
    model_class  = MPNNRegressor,
    epochs: int  = 100,
    batch_size   = 64,
    lr: float    = 1e-3,
    hidden: int  = 128,
    dropout: float = 0.2,
    seed: int    = 42,
    save_dir: str = ".",
):
    """
    Train a GNN on any DataFrame with columns ['smiles', 'target'].

    Saves:
        best_model.pt   — best weights (lowest val RMSE)
        scaler.pkl      — fitted StandardScaler (needed for predict())

    Returns:
        model, scaler
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice  : {device}")
    print(f"Model   : {model_class.__name__}  |  hidden={hidden}  dropout={dropout}")
    print(f"Target  : {df['target'].describe().to_dict()}\n")

    # ── Normalise targets (z-score) ───────────────────────────────────────────
    scaler = StandardScaler()
    df = df.copy()
    df["target"] = scaler.fit_transform(df[["target"]])

    # ── Split  80 / 10 / 10 ──────────────────────────────────────────────────
    train_df, test_df = train_test_split(df, test_size=0.10, random_state=seed)
    train_df, val_df  = train_test_split(train_df, test_size=0.10, random_state=seed)
    print(f"Split   : train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    # ── Featurise ─────────────────────────────────────────────────────────────
    print("\nBuilding graph datasets …")
    train_set = MolecularDataset(train_df)
    val_set   = MolecularDataset(val_df)
    test_set  = MolecularDataset(test_df)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size)
    test_loader  = DataLoader(test_set,  batch_size=batch_size)

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = model_class(hidden=hidden, dropout=dropout).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                  patience=10, min_lr=1e-6)

    best_val_rmse = float("inf")
    model_path    = os.path.join(save_dir, "best_model.pt")
    scaler_path   = os.path.join(save_dir, "scaler.pkl")

    print(f"\n{'Epoch':>6}  {'Train MSE':>10}  {'Val RMSE':>10}  {'Val MAE':>9}  {'LR':>10}")
    print("─" * 60)

    for epoch in range(1, epochs + 1):
        train_loss        = train_epoch(model, train_loader, optimizer, device)
        val_rmse, val_mae = evaluate(model, val_loader, device)
        scheduler.step(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), model_path)
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"{epoch:>6}  {train_loss:>10.4f}  {val_rmse:>10.4f}  {val_mae:>9.4f}  {lr_now:>10.2e}")

    # ── Test set ──────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(model_path))
    test_rmse, test_mae = evaluate(model, test_loader, device)
    scale = scaler.scale_[0]
    print(f"\nTest RMSE : {test_rmse:.4f}  ({test_rmse * scale:.4f} original units)")
    print(f"Test MAE  : {test_mae:.4f}  ({test_mae  * scale:.4f} original units)")
    print(f"\nSaved → {model_path}")
    print(f"Saved → {scaler_path}")

    return model, scaler


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — predict()  (load saved model and run on new SMILES)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(smiles_list: list, model_class=MPNNRegressor,
            model_path="best_model.pt", scaler_path="scaler.pkl",
            hidden=128, device=None):
    """
    Load a saved model and predict the property for new SMILES strings.

    Usage (after training):
        results = predict(["CCO", "c1ccccc1", "CC(=O)O"])
        for smi, val in results.items():
            print(smi, val)

    Returns:
        dict  {smiles_string: predicted_float}   (original units, not normalised)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_class(hidden=hidden).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    graphs = [(s, smiles_to_graph(s, 0.0)) for s in smiles_list]
    valid  = [(s, g) for s, g in graphs if g is not None]
    invalid = [s for s, g in graphs if g is None]

    if invalid:
        print(f"Could not parse {len(invalid)} SMILES: {invalid}")
    if not valid:
        return {}

    loader = DataLoader([g for _, g in valid], batch_size=64)
    preds  = np.concatenate([model(b.to(device)).cpu().numpy() for b in loader])
    preds  = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()

    return {s: float(p) for (s, _), p in zip(valid, preds)}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — QM9 LOADER
# ─────────────────────────────────────────────────────────────────────────────

# All 12 regression targets available in QM9
QM9_TARGETS = {
    "mu":    "Dipole moment (Debye)",
    "alpha": "Polarizability (a0^3)",
    "homo":  "HOMO energy (Ha)",
    "lumo":  "LUMO energy (Ha)",
    "gap":   "HOMO-LUMO gap (Ha)  ← most popular benchmark",
    "r2":    "Electronic spatial extent (a0^2)",
    "zpve":  "Zero-point vibrational energy (Ha)",
    "U0":    "Internal energy at 0K (Ha)",
    "U":     "Internal energy at 298K (Ha)",
    "H":     "Enthalpy at 298K (Ha)",
    "G":     "Free energy at 298K (Ha)",
    "Cv":    "Heat capacity (cal/mol·K)",
}


def load_qm9(csv_path: str, target: str = "gap") -> pd.DataFrame:
    """
    Load a QM9 CSV file and return a clean DataFrame with columns
    ['smiles', 'target'] ready to pass straight into run_training().

    Works with both:
      • Chainer/Figshare variant  → SMILES column is 'SMILES1'
      • PyG / DeepChem variant    → SMILES column is 'smiles'

    Args:
        csv_path : path to your qm9.csv
        target   : which property to predict  (default: 'gap')
                   run  print(list(QM9_TARGETS.keys()))  to see all options
    """
    assert target in QM9_TARGETS, (
        f"Unknown target '{target}'.\n"
        f"Valid options: {list(QM9_TARGETS.keys())}"
    )

    df = pd.read_csv(csv_path)

    # ── Auto-detect the SMILES column ─────────────────────────────────────────
    if "SMILES1" in df.columns:
        smiles_col = "SMILES1"
    elif "smiles" in df.columns:
        smiles_col = "smiles"
    else:
        raise ValueError(
            f"No recognised SMILES column found.\n"
            f"Columns in your file: {df.columns.tolist()}"
        )

    if target not in df.columns:
        raise ValueError(
            f"Column '{target}' not in CSV.\n"
            f"Columns available: {df.columns.tolist()}"
        )

    out = df[[smiles_col, target]].copy()
    out.columns = ["smiles", "target"]
    out = out.dropna()

    # Drop rows RDKit cannot parse
    bad = out["smiles"].apply(lambda s: Chem.MolFromSmiles(str(s)) is None)
    if bad.sum():
        print(f"  Dropped {bad.sum()} unparseable SMILES rows.")
    out = out[~bad].reset_index(drop=True)

    print(f"QM9 loaded  →  target: '{target}'  ({QM9_TARGETS[target]})")
    print(f"  Molecules : {len(out):,}")
    print(f"  Range     : {out['target'].min():.5f}  –  {out['target'].max():.5f}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
#
# ╔══════════════════════════════════╗
# ║  YOUR TWO KNOBS — edit these     ║
# ╚══════════════════════════════════╝

if __name__ == "__main__":

    CSV_PATH  = "qm9.csv"   # ← path to your downloaded qm9.csv
    TARGET    = "gap"        # ← any key from QM9_TARGETS above

    # ── Step 1: load & clean QM9 ─────────────────────────────────────────────
    df = load_qm9(CSV_PATH, target=TARGET)

    # ── Step 2: train ────────────────────────────────────────────────────────
    model, scaler = run_training(
        df           = df,
        model_class  = MPNNRegressor,   # swap to GCNRegressor or GATRegressor
        epochs       = 100,
        batch_size   = 64,
        hidden       = 128,
        dropout      = 0.2,
        lr           = 1e-3,
    )

    # ── Step 3: predict on new molecules ─────────────────────────────────────
    new_molecules = [
        "CC(=O)Oc1ccccc1C(=O)O",    # Aspirin
        "CN1CCC[C@H]1c2cccnc2",     # Nicotine
        "CCO",                       # Ethanol
    ]
    print("\nPredictions on new molecules:")
    results = predict(new_molecules)
    for smi, val in results.items():
        print(f"  {smi:<40}  {TARGET} = {val:.5f}")

    # ── To switch datasets entirely, replace Step 1 with: ────────────────────
    #   df = pd.read_csv("my_data.csv")[["my_smiles_col", "my_target_col"]]
    #   df.columns = ["smiles", "target"]
    #   Then call run_training(df, ...) exactly the same way.
