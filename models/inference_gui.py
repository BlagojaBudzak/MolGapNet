"""
=============================================================================
MOLECULAR GNN — INFERENCE GUI
Imports directly from molecular_gnn_blueprint.py — no duplicated code.
=============================================================================

FOLDER LAYOUT (all three files in the same directory):
    GNN/
    ├── molecular_gnn_blueprint.py   ← model definition & helpers
    ├── inference_gui.py             ← this file
    ├── best_model.pt                ← saved after training
    └── scaler.pkl                   ← saved after training

HOW TO RUN:
    python inference_gui.py
    python inference_gui.py --model best_model.pt --scaler scaler.pkl --target gap

NOTE ON RETRAINING:
    If you change atom_features() or bond_features() in the blueprint
    (e.g. adding new descriptors), you MUST retrain before running this GUI.
    The model architecture is built from NUM_ATOM_FEATURES / NUM_BOND_FEATURES
    imported live from the blueprint, so they will always match your checkpoint
    as long as you retrained after any featuriser change.
"""

import argparse
import os
import sys
import io
import pickle

import numpy as np
import torch
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk

# ── RDKit ─────────────────────────────────────────────────────────────────────
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

# ── PyG DataLoader ────────────────────────────────────────────────────────────
try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader

# ── Import everything from the blueprint (single source of truth) ─────────────
# NUM_ATOM_FEATURES and NUM_BOND_FEATURES are read directly from the blueprint,
# so the model built here always matches whatever checkpoint was saved there.
try:
    from molecular_gnn_blueprint import (
        smiles_to_graph,        # SMILES str → PyG Data object
        MPNNRegressor,          # trained model class
        GCNRegressor,
        GATRegressor,
        NUM_ATOM_FEATURES,      # e.g. 28 after your edit — read live, not hardcoded
        NUM_BOND_FEATURES,
    )
except ImportError:
    print("ERROR: Cannot import from molecular_gnn_blueprint.py")
    print("Make sure molecular_gnn_blueprint.py is in the same folder as this file.")
    sys.exit(1)

print(f"Blueprint loaded — atom features: {NUM_ATOM_FEATURES}, bond features: {NUM_BOND_FEATURES}")


# ─────────────────────────────────────────────────────────────────────────────
# COLOURS  (plain 6-char hex only — Tkinter rejects 8-char alpha hex like #rrggbbaa)
# ─────────────────────────────────────────────────────────────────────────────

DARK   = "#0f1117"
PANEL  = "#1a1d27"
CARD   = "#22263a"
ACCENT = "#00c8e0"
GREEN  = "#00d98a"
WARN   = "#ff6b6b"
TEXT   = "#e8eaf0"
MUTED  = "#6b7094"
HOVER  = "#2c3050"   # button hover — replaces the invalid color+"44" pattern

FONT_MONO  = ("Consolas", 11)
FONT_BIG   = ("Segoe UI", 22, "bold")
FONT_LABEL = ("Segoe UI", 9)


# ─────────────────────────────────────────────────────────────────────────────
# QM9 TARGET METADATA  (display only)
# ─────────────────────────────────────────────────────────────────────────────

QM9_TARGETS = {
    "gap":   ("HOMO-LUMO Gap",       "Ha"),
    "mu":    ("Dipole Moment",        "D"),
    "alpha": ("Polarizability",      "a0^3"),
    "homo":  ("HOMO Energy",         "Ha"),
    "lumo":  ("LUMO Energy",         "Ha"),
    "r2":    ("Spatial Extent",      "a0^2"),
    "zpve":  ("ZPVE",                "Ha"),
    "U0":    ("Internal Energy 0K",  "Ha"),
    "U":     ("Internal Energy",     "Ha"),
    "H":     ("Enthalpy",            "Ha"),
    "G":     ("Free Energy",         "Ha"),
    "Cv":    ("Heat Capacity",       "cal/mol/K"),
}

EXAMPLE_SMILES = [
    ("Aspirin",   "CC(=O)Oc1ccccc1C(=O)O"),
    ("Nicotine",  "CN1CCC[C@H]1c2cccnc2"),
    ("Ethanol",   "CCO"),
    ("Benzene",   "c1ccccc1"),
    ("Caffeine",  "Cn1cnc2c1c(=O)n(c(=O)n2C)C"),
    ("Glucose",   "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"),
    ("Dopamine",  "NCCc1ccc(O)c(O)c1"),
    ("Adenine",   "Nc1ncnc2[nH]cnc12"),
]


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str, scaler_path: str, hidden: int = 128):
    """
    Build MPNNRegressor using NUM_ATOM_FEATURES imported live from the blueprint,
    then load the checkpoint. As long as you retrained after changing the featuriser,
    the shapes will always match.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # NUM_ATOM_FEATURES comes from the blueprint import at the top of this file.
    # If you added features and retrained, this will be 28 (or whatever you set).
    # If you haven't retrained yet, delete best_model.pt and run the blueprint first.
    model = MPNNRegressor(
        in_channels=NUM_ATOM_FEATURES,
        edge_dim=NUM_BOND_FEATURES,
        hidden=hidden,
    ).to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError as e:
        # Give a clear, actionable error instead of a raw PyTorch stack trace
        print("\n" + "="*70)
        print("CHECKPOINT MISMATCH — cannot load best_model.pt")
        print("="*70)
        print(f"\nBlueprint expects  in_channels = {NUM_ATOM_FEATURES}")
        print("Checkpoint was built with a different in_channels.")
        print("\nThis happens when you change atom_features() in the blueprint")
        print("without retraining. Fix: re-run molecular_gnn_blueprint.py")
        print("to produce a new best_model.pt, then launch this GUI again.\n")
        print(f"Original error: {e}")
        print("="*70 + "\n")
        sys.exit(1)

    model.eval()

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler, device


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE  (smiles_to_graph imported from blueprint)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(smiles_list: list, model, scaler, device):
    """
    Predict property for a list of SMILES strings.
    Returns list of (smiles, float | None)  — None = RDKit couldn't parse it.
    """
    results = []
    for smi in smiles_list:
        g = smiles_to_graph(smi.strip(), y=0.0)   # y is unused at inference time
        if g is None:
            results.append((smi, None))
            continue
        g = g.to(device)
        g.batch = torch.zeros(g.num_nodes, dtype=torch.long, device=device)
        pred = model(g).item()
        pred = scaler.inverse_transform([[pred]])[0][0]
        results.append((smi, pred))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# MOLECULE IMAGE RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def smiles_to_image(smiles: str, size=(340, 240)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        AllChem.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DCairo(*size)
        drawer.drawOptions().addStereoAnnotation = True
        drawer.drawOptions().bondLineWidth = 1.8
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        img = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")
        # Replace white background with the dark panel colour
        bg = tuple(int(PANEL.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        img.putdata([
            (bg[0], bg[1], bg[2], 255) if (r > 240 and g > 240 and b > 240)
            else (r, g, b, a)
            for r, g, b, a in img.getdata()
        ])
        img = img.convert("RGB")
        img.thumbnail(size, Image.LANCZOS)
        return img
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────

class MolecularGUI:
    def __init__(self, root: tk.Tk, model, scaler, device, target_key: str = "gap"):
        self.root   = root
        self.model  = model
        self.scaler = scaler
        self.device = device
        self.target = target_key
        self.t_name, self.t_unit = QM9_TARGETS.get(target_key, (target_key, ""))

        root.title("Molecular GNN — Inference")
        root.configure(bg=DARK)
        root.geometry("940x700")
        root.minsize(820, 600)

        self._build_ui()

    # ── Layout ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        hdr = tk.Frame(self.root, bg=DARK, pady=12)
        hdr.pack(fill="x", padx=22)
        tk.Label(hdr, text="⬡ Molecular GNN", font=FONT_BIG,
                 fg=ACCENT, bg=DARK).pack(side="left")
        tk.Label(hdr,
                 text=(f"  ·  {self.t_name} [{self.t_unit}]"
                       f"  ·  {str(self.device).upper()}"
                       f"  ·  {NUM_ATOM_FEATURES}d atom features"),
                 font=("Segoe UI", 10), fg=MUTED, bg=DARK).pack(side="left", pady=6)

        body = tk.Frame(self.root, bg=DARK)
        body.pack(fill="both", expand=True, padx=22, pady=(0, 16))
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left  = tk.Frame(body, bg=DARK)
        right = tk.Frame(body, bg=DARK)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        right.grid(row=0, column=1, sticky="nsew")

        self._build_left(left)
        self._build_right(right)

    def _build_left(self, parent):
        # Input card
        c = self._card(parent)
        self._label(c, "SMILES Input  (one per line for batch)")
        self.smiles_entry = tk.Text(
            c, height=4, font=FONT_MONO,
            bg=CARD, fg=TEXT, insertbackground=ACCENT,
            relief="flat", bd=8, wrap="word",
            selectbackground=HOVER)
        self.smiles_entry.pack(fill="x", padx=12, pady=(0, 8))
        self.smiles_entry.insert("1.0", "CC(=O)Oc1ccccc1C(=O)O")
        self.smiles_entry.bind("<Control-Return>", lambda e: self._predict())

        row = tk.Frame(c, bg=PANEL)
        row.pack(fill="x", padx=12, pady=(0, 12))
        self._btn(row, "Predict  (Ctrl+Enter)", self._predict, ACCENT).pack(side="left")
        self._btn(row, "Clear",                 self._clear,   MUTED ).pack(side="left", padx=8)
        self._btn(row, "Batch CSV…",            self._batch_csv, MUTED).pack(side="right")

        # Structure viewer
        cv = self._card(parent, expand=True)
        self._label(cv, "Structure")
        self.mol_canvas = tk.Label(
            cv, bg=CARD, relief="flat",
            text="Enter a SMILES and press Ctrl+Enter",
            fg=MUTED, font=("Segoe UI", 9))
        self.mol_canvas.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        # Examples
        ex = self._card(parent)
        self._label(ex, "Examples")
        grid = tk.Frame(ex, bg=PANEL)
        grid.pack(fill="x", padx=12, pady=(0, 12))
        for i, (name, smi) in enumerate(EXAMPLE_SMILES):
            tk.Button(
                grid, text=name, font=FONT_LABEL,
                bg=CARD, fg=TEXT, relief="flat", bd=0,
                padx=8, pady=5, cursor="hand2",
                activebackground=HOVER, activeforeground=ACCENT,
                command=lambda s=smi: self._load_example(s)
            ).grid(row=i // 4, column=i % 4, padx=3, pady=3, sticky="ew")
        for col in range(4):
            grid.columnconfigure(col, weight=1)

    def _build_right(self, parent):
        # Prediction display
        rc = self._card(parent)
        self._label(rc, "Prediction")
        self.result_var = tk.StringVar(value="—")
        self.result_lbl = tk.Label(
            rc, textvariable=self.result_var,
            font=("Segoe UI", 36, "bold"), fg=GREEN, bg=PANEL)
        self.result_lbl.pack(pady=(4, 0))
        tk.Label(rc, text=self.t_unit,  font=("Segoe UI", 13), fg=MUTED, bg=PANEL).pack()
        tk.Label(rc, text=self.t_name,  font=("Segoe UI",  9), fg=MUTED, bg=PANEL).pack(pady=(0, 12))

        # Molecule info
        ic = self._card(parent)
        self._label(ic, "Molecule Info")
        self.info_text = tk.Text(
            ic, height=7, font=FONT_MONO,
            bg=CARD, fg=TEXT, relief="flat", bd=8,
            state="disabled", selectbackground=HOVER)
        self.info_text.pack(fill="x", padx=12, pady=(0, 12))

        # History table
        hc = self._card(parent, expand=True)
        self._label(hc, "History")

        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview",
                        background=CARD, foreground=TEXT,
                        fieldbackground=CARD, rowheight=26,
                        font=("Consolas", 9))
        style.configure("Treeview.Heading",
                        background=PANEL, foreground=MUTED,
                        font=("Segoe UI", 9, "bold"), relief="flat")
        style.map("Treeview", background=[("selected", HOVER)])

        cols = ("SMILES", "Prediction", "Unit")
        self.tree = ttk.Treeview(hc, columns=cols, show="headings", height=8)
        self.tree.heading("SMILES",     text="SMILES")
        self.tree.heading("Prediction", text=self.t_name)
        self.tree.heading("Unit",       text="Unit")
        self.tree.column("SMILES",      width=240, anchor="w")
        self.tree.column("Prediction",  width=110, anchor="center")
        self.tree.column("Unit",        width=80,  anchor="center")

        sb = ttk.Scrollbar(hc, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=(12, 0), pady=(0, 8))
        sb.pack(side="right", fill="y", pady=(0, 8), padx=(0, 12))

        foot = tk.Frame(parent, bg=DARK)
        foot.pack(fill="x")
        self._btn(foot, "Export History…", self._export_history, MUTED).pack(side="right")
        self._btn(foot, "Clear History",   self._clear_history,  MUTED).pack(side="right", padx=6)

    # ── Widget helpers ────────────────────────────────────────────────────────

    def _card(self, parent, expand=False):
        f = tk.Frame(parent, bg=PANEL, bd=0)
        f.pack(fill="both", expand=expand, pady=(0, 10))
        return f

    def _label(self, parent, text):
        tk.Label(parent, text=text, font=("Segoe UI", 8, "bold"),
                 fg=MUTED, bg=parent["bg"]).pack(anchor="w", padx=12, pady=(10, 4))

    def _btn(self, parent, text, cmd, color):
        # bg and activebackground use proper 6-char hex — no alpha suffix
        return tk.Button(
            parent, text=text, command=cmd,
            font=("Segoe UI", 9, "bold"),
            bg=CARD, fg=color,
            activebackground=HOVER, activeforeground=color,
            relief="flat", bd=0, padx=14, pady=7, cursor="hand2")

    # ── Actions ──────────────────────────────────────────────────────────────

    def _predict(self):
        raw = self.smiles_entry.get("1.0", "end").strip()
        if not raw:
            return
        smiles_list = [s.strip() for s in raw.splitlines() if s.strip()]
        results = run_inference(smiles_list, self.model, self.scaler, self.device)

        first_smi, first_val = results[0]
        if first_val is None:
            self.result_var.set("Invalid")
            self.result_lbl.config(fg=WARN)
        else:
            self.result_var.set(f"{first_val:.5f}")
            self.result_lbl.config(fg=GREEN)
            self._show_structure(first_smi)
            self._show_info(first_smi, first_val)

        for smi, val in results:
            self.tree.insert("", 0, values=(
                smi,
                f"{val:.5f}" if val is not None else "ERROR",
                self.t_unit))

    def _show_structure(self, smiles):
        img = smiles_to_image(smiles)
        if img is None:
            self.mol_canvas.config(image="", text="Could not render structure")
            return
        self._tk_img = ImageTk.PhotoImage(img)
        self.mol_canvas.config(image=self._tk_img, text="")

    def _show_info(self, smiles, pred):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return
        lines = (
            f"Formula  : {rdMolDescriptors.CalcMolFormula(mol)}\n"
            f"Mol. Wt  : {Descriptors.MolWt(mol):.2f} g/mol\n"
            f"LogP     : {Descriptors.MolLogP(mol):.3f}\n"
            f"HBD / HBA: {rdMolDescriptors.CalcNumHBD(mol)} / {rdMolDescriptors.CalcNumHBA(mol)}\n"
            f"Rings    : {rdMolDescriptors.CalcNumRings(mol)}"
            f"   Atoms: {mol.GetNumAtoms()}"
            f"   Bonds: {mol.GetNumBonds()}\n"
            f"Predicted: {pred:.5f} {self.t_unit}"
        )
        self.info_text.config(state="normal")
        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", lines)
        self.info_text.config(state="disabled")

    def _load_example(self, smiles):
        self.smiles_entry.delete("1.0", "end")
        self.smiles_entry.insert("1.0", smiles)
        self._predict()

    def _clear(self):
        self.smiles_entry.delete("1.0", "end")
        self.result_var.set("—")
        self.result_lbl.config(fg=GREEN)
        self.mol_canvas.config(image="", text="Enter a SMILES and press Ctrl+Enter")
        self.info_text.config(state="normal")
        self.info_text.delete("1.0", "end")
        self.info_text.config(state="disabled")

    def _clear_history(self):
        for row in self.tree.get_children():
            self.tree.delete(row)

    def _batch_csv(self):
        path = filedialog.askopenfilename(
            title="Open CSV with a SMILES column",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        import pandas as pd
        try:
            df = pd.read_csv(path)
        except Exception as e:
            messagebox.showerror("Error", f"Could not read file:\n{e}")
            return
        candidates = [c for c in df.columns if "smiles" in c.lower()]
        if not candidates:
            messagebox.showerror(
                "Error",
                f"No column containing 'smiles' found.\nColumns: {df.columns.tolist()}")
            return
        smiles_list = df[candidates[0]].dropna().tolist()
        results = run_inference(smiles_list, self.model, self.scaler, self.device)
        for smi, val in results:
            self.tree.insert("", 0, values=(
                smi,
                f"{val:.5f}" if val is not None else "ERROR",
                self.t_unit))
        messagebox.showinfo(
            "Batch complete",
            f"Predicted {len(results)} molecules from {os.path.basename(path)}.\n"
            "Results added to history.")

    def _export_history(self):
        rows = [self.tree.item(r)["values"] for r in self.tree.get_children()]
        if not rows:
            messagebox.showinfo("Nothing to export", "History is empty.")
            return
        path = filedialog.asksaveasfilename(
            title="Save history as CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        import pandas as pd
        pd.DataFrame(rows, columns=["smiles", self.target, "unit"]).to_csv(path, index=False)
        messagebox.showinfo("Exported", f"Saved {len(rows)} rows to:\n{path}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Molecular GNN Inference GUI")
    parser.add_argument("--model",  default="best_model.pt")
    parser.add_argument("--scaler", default="scaler.pkl")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--target", default="gap",
                        help=f"One of: {list(QM9_TARGETS.keys())}")
    args = parser.parse_args()

    for p, label in [(args.model, "Model"), (args.scaler, "Scaler")]:
        if not os.path.exists(p):
            print(f"\nERROR: {label} file not found → {p}")
            print("Run molecular_gnn_blueprint.py first to train and save the model.")
            sys.exit(1)

    print(f"Loading model  : {args.model}")
    print(f"Loading scaler : {args.scaler}")
    model, scaler, device = load_model(args.model, args.scaler, args.hidden)
    print(f"Device         : {device}")
    print("Launching GUI …")

    root = tk.Tk()
    MolecularGUI(root, model, scaler, device, target_key=args.target)
    root.mainloop()


if __name__ == "__main__":
    main()