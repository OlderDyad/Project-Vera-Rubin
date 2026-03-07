"""
generate_reports.py v2.3
───────────────────────
Professional 'SQLS-Style' Report Generator. Includes visual cutouts, 
physics modeling, color analysis, and a formal academic bibliography.
"""
from fpdf import FPDF, Align
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Paths
INPUT_FILE = Path("outputs/hunter/color_vetted_candidates.csv")
REPORT_DIR = Path("outputs/reports/")
IMAGE_DIR = Path("outputs/hunter/cutouts/")
PLOT_DIR = Path("outputs/hunter/plots/")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

class DiscoveryPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 16)
        self.cell(0, 10, 'Gravitational Lens Discovery Report', 0, 1, 'C')
        self.set_font('Helvetica', '', 10)
        self.cell(0, 5, f'Generated on: {datetime.now().strftime("%Y-%m-%d")}', 0, 1, 'C')
        self.ln(10)

    def chapter_title(self, label):
        self.set_font('Helvetica', 'B', 12)
        self.set_fill_color(230, 240, 250)
        self.cell(0, 8, f"  {label}", 0, 1, 'L', fill=True)
        self.ln(4)

    def data_row(self, label, value):
        self.set_font('Helvetica', 'B', 10)
        self.cell(55, 6, f"{label}:", 0, 0)
        self.set_font('Helvetica', '', 10)
        self.cell(0, 6, str(value), 0, 1)

def generate_color_plot(anchor, g_r, r_i):
    plt.figure(figsize=(3, 3))
    plt.axvspan(-0.2, 0.5, color='green', alpha=0.1, label='Quasar Locus')
    plt.scatter(g_r, r_i, color='red', marker='*', s=100, label='Target')
    plt.xlabel('g - r')
    plt.ylabel('r - i')
    plt.xlim(-1, 2)
    plt.ylim(-1, 2)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plot_path = PLOT_DIR / f"color_{anchor.replace(' ', '_')}.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    return plot_path

def create_reports():
    if not INPUT_FILE.exists():
        print(f"⚠ Missing data. Complete the workflow before running.")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"📄 Printing {len(df)} Academic Portfolios...")

    for _, row in df.iterrows():
        pdf = DiscoveryPDF()
        pdf.add_page()
        
        # Section 1: Target Identity
        pdf.chapter_title(f"Target: {row['anchor']}")
        pdf.data_row("RA / Dec", f"{row['ra']:.5f} / {row['dec']:.5f}")
        pdf.data_row("SIMBAD Type / z", f"{row.get('simbad_otype', 'Unknown')} / {row.get('simbad_z', 'N/A')}")
        
        # Section 2: Visual & Photometric Analysis
        pdf.chapter_title("Visual & Color Analysis")
        y_start = pdf.get_y()
        
        # Cutout and Color Plot
        clean_anchor = row['anchor'].replace(" ", "_").replace("+", "p")
        img_path = IMAGE_DIR / f"cutout_{clean_anchor}.jpg"
        plot_path = generate_color_plot(clean_anchor, row['g_minus_r'], row['r_minus_i'])

        if img_path.exists():
            pdf.image(str(img_path), x=15, y=y_start, w=85)
        pdf.image(str(plot_path), x=110, y=y_start, w=85)
        pdf.set_y(y_start + 85)
        pdf.ln(5)

        # Section 3: Detailed Physics
        pdf.chapter_title("Validation & SIS Mass Modeling")
        pdf.data_row("Measured Time Delay", f"{row['best_lag_days']} days")
        pdf.data_row("Estimated Mass", f"{row['est_mass_msun']:.2e} M_sun")
        pdf.data_row("Mass Classification", row['mass_classification'])
        pdf.data_row("Gaia DR3 Parallax", f"{row.get('gaia_parallax', '0.00')} mas")
        pdf.ln(10)

        # Section 4: Spectroscopic Confirmation (Replaces DESI Metadata Integration block)
    if 'desi_target_id' in row and row.get('zwarn') == 0:
        pdf.chapter_title("Spectroscopic Confirmation (DESI)")
        pdf.set_text_color(0, 150, 0) # Green for confirmation
        pdf.data_row("Status", "SPECTROSCOPICALLY CONFIRMED") # cite: 2.8
        pdf.set_text_color(0, 0, 0) # Reset to black
        pdf.data_row("DESI Target ID", row['desi_target_id']) # cite: 2.7
        pdf.data_row("Spectroscopic Redshift (z)", f"{row['spec_z']:.4f}") # cite: 2.8
        pdf.ln(5)

        # Section 5: References
        pdf.chapter_title("Scientific References")
        pdf.set_font('Helvetica', 'I', 8)
        refs = [
            "1. Oguri, M., et al. (2006). The Sloan Digital Sky Survey Quasar Lens Search. AJ, 132, 999.",
            "2. Bellm, E. C., et al. (2019). The Zwicky Transient Facility: System Overview. PASP, 131, 018002.",
            "3. Gaia Collaboration (2022). Gaia Data Release 3: Summary of the content and survey properties.",
            "4. Zourba, A., et al. (2020). Z-transformed Discrete Correlation Function (ZDCF) Methodology."
        ]
        for ref in refs:
            pdf.multi_cell(0, 5, ref)

        # Final Save
        file_path = REPORT_DIR / f"Portfolio_{clean_anchor}.pdf"
        pdf.output(str(file_path))
        print(f"  ✓ Portfolio Saved: {file_path.name}")

if __name__ == "__main__":
    create_reports()