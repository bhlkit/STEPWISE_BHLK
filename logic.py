# logic.py
import pandas as pd
import numpy as np
import statsmodels.api as sm
import tkinter.messagebox as messagebox

# --- FUNGSI BACA DATA ---
def read_data_native(filename):
    try:
        df = None
        if filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(filename, sheet_name=0)
        elif filename.endswith('.csv'):
            df = pd.read_csv(filename)
        elif filename.endswith('.txt'):
            df = pd.read_csv(filename, sep=r'\s+', header=None, skiprows=24, engine='python')
            # Beri nama kolom otomatis VAR1, VAR2, dst jika txt
            df.columns = [f'VAR{i}' for i in range(1, len(df.columns) + 1)]
        else:
            raise ValueError("Format file tidak didukung.")

        if df is None or len(df) == 0: raise ValueError("File kosong.")

        df = df.dropna().reset_index(drop=True)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna().reset_index(drop=True)
        return df
    except Exception as e:
        messagebox.showerror("Error Baca File", f"{e}")
        return None

def get_descriptive_stats(df):
    return df.describe().T[['count', 'mean', 'std', 'min', 'max']].to_string()

def get_correlation_matrix(df):
    return df.corr().round(4).to_string()

# --- LOGIKA STEPWISE ---
def stepwise_regression_final(df, target_var, fin_threshold=2.5):
    included = []
    results = []
    
    while True:
        changed = False
        excluded = list(set(df.columns) - set(included) - {target_var})
        
        best_f = -1.0
        best_feature = None
        best_model = None
        
        for feature in excluded:
            try:
                features_to_try = included + [feature]
                X_poly = sm.add_constant(df[features_to_try])
                y = df[target_var]
                
                model = sm.OLS(y, X_poly).fit()
                f_value = model.fvalue
                if np.isnan(f_value): f_value = 0.0

                if f_value > best_f:
                    best_f = f_value
                    best_feature = feature
                    best_model = model
            except:
                continue

        if best_feature is not None and best_f > fin_threshold:
            included.append(best_feature)
            changed = True
            
            results.append({
                'step': len(results) + 1,
                'action': f"Entered {best_feature}",
                'r_squared': best_model.rsquared,
                'adj_r_squared': best_model.rsquared_adj,
                'aic': best_model.aic,
                'bic': best_model.bic,
                'f_value': best_model.fvalue,
                'f_pvalue': best_model.f_pvalue,
                'model': best_model,
                'scale': best_model.scale
            })
            
        if not changed:
            break
            
    return results

# --- HELPER: RUMUS ---
def generate_equation(model, target_var, use_log):
    params = model.params
    if use_log:
        eq_str = f"Ln({target_var}) = "
    else:
        eq_str = f"{target_var} = "
    
    const_val = params.get('const', 0)
    eq_str += f"{const_val:.4f}"
    
    for var, coef in params.items():
        if var == 'const': continue
        sign = " + " if coef >= 0 else " - "
        val = abs(coef)
        var_name = f"Ln({var})" if use_log else var
        eq_str += f"{sign}{val:.4f}*{var_name}"
    return eq_str