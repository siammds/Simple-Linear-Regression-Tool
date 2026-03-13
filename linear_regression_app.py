#!/usr/bin/env python3
# linear_regression_gui_app.py
"""
Ready-made Simple Linear Regression (GUI)

What you do:
1) Run: python linear_regression_gui_app.py
2) A window opens:
   - Click "Select File" (CSV / Excel)
   - Choose Independent (X) and Dependent (Y) from dropdowns
   - Click "Run Regression"
3) It shows the regression plot and saves: linear_regression_plot.png

Update in this version:
- The plot now displays the predicted Y values (ŷ) from the regression equation ON the graph:
  - Equation shown in a text box
  - ŷ values annotated at the left and right ends of the regression line (xmin and xmax)

Large-data friendly:
- CSV is processed in chunks (so it can handle millions of rows).
- Excel is loaded via pandas (Excel can be memory-heavy; you’ll get a helpful warning if it fails).

Dependencies:
  pip install pandas numpy matplotlib openpyxl
(For .xls you may also need: pip install xlrd)
"""

import os
import sys
import math
import random
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Tkinter UI
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# -------------------------
# Data helpers
# -------------------------
def infer_file_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return "csv"
    if ext in (".xlsx", ".xls"):
        return "excel"
    raise ValueError("Unsupported file type. Please select a .csv, .xlsx, or .xls file.")


def get_columns(path: str, ftype: str) -> List[str]:
    if ftype == "csv":
        df0 = pd.read_csv(path, nrows=0)
        return list(df0.columns)
    else:
        df0 = pd.read_excel(path, nrows=0, engine=None)
        return list(df0.columns)


def iter_two_columns_csv(path: str, x_col: str, y_col: str, chunksize: int = 250_000):
    for chunk in pd.read_csv(path, usecols=[x_col, y_col], chunksize=chunksize):
        yield chunk


def read_two_columns_excel(path: str, x_col: str, y_col: str) -> pd.DataFrame:
    return pd.read_excel(path, usecols=[x_col, y_col], engine=None)


def reservoir_sample_update(sample_x: List[float], sample_y: List[float],
                            x_vals: np.ndarray, y_vals: np.ndarray,
                            seen: int, max_points: int, rng: random.Random) -> int:
    for x, y in zip(x_vals, y_vals):
        seen += 1
        if len(sample_x) < max_points:
            sample_x.append(float(x))
            sample_y.append(float(y))
        else:
            j = rng.randint(1, seen)
            if j <= max_points:
                idx = j - 1
                sample_x[idx] = float(x)
                sample_y[idx] = float(y)
    return seen


def first_pass_fit_streaming(path: str, ftype: str, x_col: str, y_col: str,
                             drop_missing: bool, max_plot_points: int,
                             csv_chunksize: int) -> dict:
    n = 0
    sum_x = 0.0
    sum_y = 0.0
    sum_xx = 0.0
    sum_xy = 0.0
    sum_yy = 0.0

    x_min = None
    x_max = None

    sample_x: List[float] = []
    sample_y: List[float] = []
    rng = random.Random(42)
    seen_for_sampling = 0

    rows_read = 0
    rows_dropped = 0

    def process_df(df: pd.DataFrame):
        nonlocal n, sum_x, sum_y, sum_xx, sum_xy, sum_yy, x_min, x_max
        nonlocal rows_read, rows_dropped, seen_for_sampling, sample_x, sample_y

        rows_read += len(df)

        x = pd.to_numeric(df[x_col], errors="coerce")
        y = pd.to_numeric(df[y_col], errors="coerce")

        mask = x.notna() & y.notna()
        rows_dropped += int((~mask).sum())
        x = x[mask]
        y = y[mask]

        if len(x) == 0:
            return

        xv = x.to_numpy(dtype=np.float64, copy=False)
        yv = y.to_numpy(dtype=np.float64, copy=False)

        cmin = float(np.min(xv))
        cmax = float(np.max(xv))
        x_min = cmin if x_min is None else min(x_min, cmin)
        x_max = cmax if x_max is None else max(x_max, cmax)

        n_chunk = xv.size
        n += n_chunk
        sum_x += float(np.sum(xv))
        sum_y += float(np.sum(yv))
        sum_xx += float(np.sum(xv * xv))
        sum_xy += float(np.sum(xv * yv))
        sum_yy += float(np.sum(yv * yv))

        seen_for_sampling = reservoir_sample_update(
            sample_x, sample_y, xv, yv, seen_for_sampling, max_plot_points, rng
        )

    if ftype == "csv":
        for chunk in iter_two_columns_csv(path, x_col, y_col, chunksize=csv_chunksize):
            process_df(chunk)
    else:
        df = read_two_columns_excel(path, x_col, y_col)
        process_df(df)

    if n < 2:
        raise ValueError("Not enough valid numeric rows after cleaning to fit regression (need at least 2).")

    denom = (n * sum_xx) - (sum_x * sum_x)
    if denom == 0.0:
        raise ValueError("Cannot fit regression: X has zero variance (all X values identical).")

    slope = ((n * sum_xy) - (sum_x * sum_y)) / denom
    intercept = (sum_y - slope * sum_x) / n

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "n_used": int(n),
        "rows_read": int(rows_read),
        "rows_dropped": int(rows_dropped),
        "x_min": float(x_min) if x_min is not None else None,
        "x_max": float(x_max) if x_max is not None else None,
        "sum_y": float(sum_y),
        "sum_yy": float(sum_yy),
        "sample_x": np.array(sample_x, dtype=np.float64),
        "sample_y": np.array(sample_y, dtype=np.float64),
    }


def second_pass_metrics(path: str, ftype: str, x_col: str, y_col: str,
                        slope: float, intercept: float,
                        sum_y: float, sum_yy: float, n_used: int,
                        csv_chunksize: int) -> dict:
    y_mean = sum_y / n_used
    sst = sum_yy - (n_used * y_mean * y_mean)

    sse = 0.0
    sae = 0.0
    used = 0

    def process_df(df: pd.DataFrame):
        nonlocal sse, sae, used
        x = pd.to_numeric(df[x_col], errors="coerce")
        y = pd.to_numeric(df[y_col], errors="coerce")
        mask = x.notna() & y.notna()
        x = x[mask]
        y = y[mask]
        if len(x) == 0:
            return
        xv = x.to_numpy(dtype=np.float64, copy=False)
        yv = y.to_numpy(dtype=np.float64, copy=False)
        y_hat = slope * xv + intercept
        err = yv - y_hat
        sse += float(np.sum(err * err))
        sae += float(np.sum(np.abs(err)))
        used += int(err.size)

    if ftype == "csv":
        for chunk in iter_two_columns_csv(path, x_col, y_col, chunksize=csv_chunksize):
            process_df(chunk)
    else:
        df = read_two_columns_excel(path, x_col, y_col)
        process_df(df)

    if used == 0:
        raise ValueError("No valid rows available for metric computation.")

    mae = sae / used
    rmse = math.sqrt(sse / used)
    r2 = float("nan") if sst <= 0.0 else (1.0 - (sse / sst))

    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2), "used": int(used)}


# -------------------------
# Plotting (UPDATED)
# -------------------------
def plot_results(sample_x: np.ndarray, sample_y: np.ndarray,
                 slope: float, intercept: float,
                 x_min: Optional[float], x_max: Optional[float],
                 x_col: str, y_col: str,
                 out_path: str = "linear_regression_plot.png") -> None:
    if sample_x.size == 0:
        raise ValueError("No points available to plot.")

    xmin = x_min if x_min is not None else float(np.min(sample_x))
    xmax = x_max if x_max is not None else float(np.max(sample_x))

    # Regression line endpoints
    line_x = np.array([xmin, xmax], dtype=np.float64)
    line_y = slope * line_x + intercept

    # Predicted y values at endpoints (these are "the y we found from the regression equation")
    y_left = float(line_y[0])
    y_right = float(line_y[1])

    plt.figure()
    plt.scatter(sample_x, sample_y, s=6)
    plt.plot(line_x, line_y, linewidth=2)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title("Simple Linear Regression")

    # Show equation on the plot
    eq_text = f"ŷ = {slope:.6g}·x + {intercept:.6g}"
    plt.gca().text(
        0.02, 0.98, eq_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.2)
    )

    # Annotate predicted y values at xmin and xmax on the line
    plt.annotate(
        f"ŷ={y_left:.6g}\n@ x={xmin:.6g}",
        xy=(xmin, y_left),
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round", alpha=0.2),
        arrowprops=dict(arrowstyle="->", shrinkA=0, shrinkB=0),
    )
    plt.annotate(
        f"ŷ={y_right:.6g}\n@ x={xmax:.6g}",
        xy=(xmax, y_right),
        xytext=(-10, 10),
        textcoords="offset points",
        ha="right",
        bbox=dict(boxstyle="round", alpha=0.2),
        arrowprops=dict(arrowstyle="->", shrinkA=0, shrinkB=0),
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()


# -------------------------
# GUI App
# -------------------------
class RegressionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Simple Linear Regression (CSV/Excel)")
        self.geometry("650x260")
        self.resizable(False, False)

        self.file_path: Optional[str] = None
        self.file_type: Optional[str] = None
        self.columns: List[str] = []

        self.x_var = tk.StringVar()
        self.y_var = tk.StringVar()
        self.drop_missing_var = tk.BooleanVar(value=True)

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 8}

        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True)

        # File row
        file_row = ttk.Frame(frm)
        file_row.pack(fill="x", **pad)

        self.file_label = ttk.Label(file_row, text="No file selected", width=60)
        self.file_label.pack(side="left", padx=(0, 10))

        ttk.Button(file_row, text="Select File", command=self.on_select_file).pack(side="left")

        # Dropdowns
        dd_row = ttk.Frame(frm)
        dd_row.pack(fill="x", **pad)

        ttk.Label(dd_row, text="Independent (X):").grid(row=0, column=0, sticky="w")
        self.x_combo = ttk.Combobox(dd_row, textvariable=self.x_var, state="disabled", width=35)
        self.x_combo.grid(row=0, column=1, padx=(10, 20), sticky="w")

        ttk.Label(dd_row, text="Dependent (Y):").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.y_combo = ttk.Combobox(dd_row, textvariable=self.y_var, state="disabled", width=35)
        self.y_combo.grid(row=1, column=1, padx=(10, 20), sticky="w", pady=(10, 0))

        # Options
        opt_row = ttk.Frame(frm)
        opt_row.pack(fill="x", **pad)

        ttk.Checkbutton(
            opt_row,
            text="Drop missing / non-numeric rows (recommended)",
            variable=self.drop_missing_var
        ).pack(side="left")

        # Run button + status
        run_row = ttk.Frame(frm)
        run_row.pack(fill="x", **pad)

        self.run_btn = ttk.Button(run_row, text="Run Regression", command=self.on_run, state="disabled")
        self.run_btn.pack(side="left")

        self.status = ttk.Label(run_row, text="", foreground="")
        self.status.pack(side="left", padx=12)

        # Note
        note = ttk.Label(
            frm,
            text="Note: For CSV with millions of rows, the script reads in chunks. Excel can be memory-heavy.",
            wraplength=620
        )
        note.pack(fill="x", padx=10, pady=(0, 10))

    def set_status(self, msg: str):
        self.status.config(text=msg)
        self.update_idletasks()

    def on_select_file(self):
        path = filedialog.askopenfilename(
            title="Select CSV or Excel file",
            filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx *.xls"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            ftype = infer_file_type(path)
            cols = get_columns(path, ftype)
            if not cols:
                raise ValueError("No columns found in this file.")
        except Exception as e:
            messagebox.showerror("File Error", str(e))
            return

        self.file_path = path
        self.file_type = ftype
        self.columns = cols

        self.file_label.config(text=os.path.basename(path))

        # enable dropdowns
        self.x_combo.config(state="readonly", values=self.columns)
        self.y_combo.config(state="readonly", values=self.columns)

        # preselect
        self.x_var.set(self.columns[0])
        self.y_var.set(self.columns[1] if len(self.columns) > 1 else self.columns[0])

        self.run_btn.config(state="normal")
        self.set_status("File loaded. Choose X and Y.")

    def on_run(self):
        if not self.file_path or not self.file_type:
            messagebox.showwarning("Missing file", "Please select a dataset file first.")
            return

        x_col = self.x_var.get().strip()
        y_col = self.y_var.get().strip()

        if not x_col or not y_col:
            messagebox.showwarning("Missing columns", "Please choose both X and Y columns.")
            return

        if x_col == y_col:
            messagebox.showwarning("Invalid selection", "X and Y must be different columns.")
            return

        drop_missing = bool(self.drop_missing_var.get())

        # performance knobs
        csv_chunksize = 250_000
        max_plot_points = 100_000

        try:
            self.run_btn.config(state="disabled")
            self.set_status("Fitting regression...")

            fit = first_pass_fit_streaming(
                path=self.file_path,
                ftype=self.file_type,
                x_col=x_col,
                y_col=y_col,
                drop_missing=drop_missing,
                max_plot_points=max_plot_points,
                csv_chunksize=csv_chunksize,
            )

            self.set_status("Computing metrics...")
            metrics = second_pass_metrics(
                path=self.file_path,
                ftype=self.file_type,
                x_col=x_col,
                y_col=y_col,
                slope=fit["slope"],
                intercept=fit["intercept"],
                sum_y=fit["sum_y"],
                sum_yy=fit["sum_yy"],
                n_used=fit["n_used"],
                csv_chunksize=csv_chunksize,
            )

            # console summary
            print("\n=== Summary ===")
            print(f"File: {self.file_path}")
            print(f"X (independent): {x_col}")
            print(f"Y (dependent):   {y_col}")
            print(f"Rows read:   {fit['rows_read']:,}")
            print(f"Rows dropped (missing/non-numeric): {fit['rows_dropped']:,}")
            print(f"Rows used for fit: {fit['n_used']:,}")
            print("\nModel:")
            print(f"  y = {fit['slope']:.8f} * x + {fit['intercept']:.8f}")
            r2_str = f"{metrics['r2']:.6f}" if not math.isnan(metrics["r2"]) else "N/A"
            print("\nMetrics:")
            print(f"  R²   = {r2_str}")
            print(f"  MAE  = {metrics['mae']:.6f}")
            print(f"  RMSE = {metrics['rmse']:.6f}")

            self.set_status("Plotting...")
            plot_results(
                sample_x=fit["sample_x"],
                sample_y=fit["sample_y"],
                slope=fit["slope"],
                intercept=fit["intercept"],
                x_min=fit["x_min"],
                x_max=fit["x_max"],
                x_col=x_col,
                y_col=y_col,
                out_path="linear_regression_plot.png",
            )

            self.set_status("Done. Saved: linear_regression_plot.png")
        except MemoryError:
            messagebox.showerror(
                "Memory Error",
                "This file seems too large to load in memory (especially Excel). "
                "If it's Excel, please convert to CSV and try again."
            )
            self.set_status("")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.set_status("")
        finally:
            self.run_btn.config(state="normal")


def main():
    try:
        app = RegressionApp()
        app.mainloop()
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()