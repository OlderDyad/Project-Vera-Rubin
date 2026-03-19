"""
generate_reports.py v2.4 - The Stellar Six Edition
─────────────────────────────────────────────────
Professional 'SQLS-Style' Report Generator. 
"""
from fpdf import FPDF
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
        self.cell(0, 10, 'Gravitational Lens Discovery Report', align='C', 
                  new_x="LMARGIN", new_y="NEXT") 
        self.set_font('Helvetica', '', 10)
        self.cell(0, 5, f'Generated on: {datetime.now().strftime("%Y-%m-%d")}', 
                  align='C', new_x="LMARGIN", new_y="NEXT")
        self.ln(10)

    def chapter_title(self, label):
        self.set_font('Helvetica', 'B', 12)
        self.set_fill_color(230, 240, 250)
        self.cell(0, 8, f"  {label}", fill=True, align='L', 
                  new_x="LMARGIN", new_y="NEXT")
        self.ln(4)

    def data_row(self, label, value):
        self.set_font('Helvetica', 'B', 10)
        # Use a fixed width (55mm) for the label
        self.cell(55, 6, f"{label}:", new_x="RIGHT", new_y="TOP")
        self.set_font('Helvetica', '', 10)
        # Use 0 to tell FPDF to go to the right margin
        self.cell(0, 6, str(value), new_x="LMARGIN", new_y="NEXT")

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
        print(f"❌ Missing data: {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"📄 Processing {len(df)} Stellar Candidates...")

    for index, row in df.iterrows():
        try:
            pdf = DiscoveryPDF()
            pdf.add_page()
            
            # --- Section 1: Target Identity ---
            pdf.chapter_title(f"Target: {row.get('anchor', 'Unknown')}")
            pdf.data_row("RA / Dec", f"{row.get('ra', 0):.5f} / {row.get('dec', 0):.5f}")
            pdf.data_row("SIMBAD Type / z", f"{row.get('simbad_otype', 'Unknown')} / {row.get('simbad_z', 'N/A')}")
            
            # --- Section 2: Visual & Color Analysis ---
            pdf.chapter_title("Visual & Color Analysis")
            y_start = pdf.get_y()
            
            clean_anchor = str(row['anchor']).replace(" ", "_").replace("+", "p")
            img_path = IMAGE_DIR / f"cutout_{clean_anchor}.jpg"
            plot_path = generate_color_plot(clean_anchor, row.get('g_minus_r', 0), row.get('r_minus_i', 0))

            if img_path.exists():
                pdf.image(str(img_path), x=15, y=y_start, w=85)
            pdf.image(str(plot_path), x=110, y=y_start, w=85)
            pdf.set_y(y_start + 85)
            pdf.ln(5)

            # --- Section 3: Detailed Physics ---
            pdf.chapter_title("Validation & SIS Mass Modeling")
            pdf.data_row("Measured Time Delay", f"{row.get('best_lag_days', 'N/A')} days")
            pdf.data_row("Estimated Mass", f"{row.get('est_mass_msun', 0):.2e} M_sun")
            pdf.data_row("Mass Classification", row.get('mass_classification', 'Unknown'))
            pdf.data_row("Gaia DR3 Parallax", f"{row.get('gaia_parallax', '0.00')} mas")
            pdf.ln(5)

            # --- Section 4: Spectroscopic Confirmation (DESI) ---
            if 'desi_target_id' in row and row.get('zwarn') == 0:
                pdf.chapter_title("Spectroscopic Confirmation (DESI)")
                pdf.set_text_color(0, 150, 0) # Green
                pdf.data_row("Status", "SPECTROSCOPICALLY CONFIRMED")
                pdf.set_text_color(0, 0, 0) # Reset
                pdf.data_row("DESI Target ID", row['desi_target_id'])
                pdf.data_row("Spectroscopic Redshift (z)", f"{row['spec_z']:.4f}")
                pdf.ln(5)

            # --- Section 5: References ---
            pdf.chapter_title("Scientific References")
            pdf.set_font('Helvetica', 'I', 8)
            refs = [
                "1. Oguri, M., et al. (2006). The Sloan Digital Sky Survey Quasar Lens Search. AJ, 132, 999.",
                "2. Bellm, E. C., et al. (2019). The Zwicky Transient Facility: System Overview. PASP, 131, 018002.",
                "3. Gaia Collaboration (2022). Gaia Data Release 3 Summary.",
                "4. Zourba, A., et al. (2020). ZDCF Methodology."
            ]
            for ref in refs:
                # 🩹 FIX: Explicitly set width to 'w=0' to fill the line properly
                pdf.multi_cell(w=0, h=5, text=ref, new_x="LMARGIN", new_y="NEXT")

            # --- Final Save ---
            file_path = REPORT_DIR / f"Portfolio_{clean_anchor}.pdf"
            pdf.output(str(file_path))
            if file_path.exists():
                print(f"  ✅ [{index+1}/{len(df)}] Saved: {file_path.name}")

        except Exception as e:
            print(f"  ❌ Error on {row.get('anchor', 'Unknown')}: {e}")

if __name__ == "__main__":
    create_reports()