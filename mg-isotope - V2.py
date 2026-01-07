# -*- coding: utf-8 -*-
# Integrated Mg Isotope Mass Balance 

import argparse
import scipy as sci
import pandas as pd
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import ScalarFormatter
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# Key parameters
oldest_time = 548.50
newest_time = 515.00
time_step = 0.01
monte_carlo_iterations = 1000

# Mg mass balance parameters
river_mg_flux = 5.5e12
river_mg_isotope = -1.09
high_temp_flux = 0.82e12
high_temp_isotope = -0.3
low_temp_flux = 1.4e12
low_temp_isotope = -1.12
clay_flux = 2.9e12
clay_isotope = 1.5
silicate_isotope = -0.25
fractionation_factor = -0.5
time_constant = 50

# pCO2 empirical model parameters (Eq.9 - differential equation)
pco2_0_min, pco2_0_max = 4000, 6000  # initial CO2 concentration (ppm)
k_min, k_max = 0.3, 0.4  # reverse weathering coefficient
delta26mg_tdm = -0.14
delta26mg_max = 2.0
# Silicate weathering parameters
n_si_min, n_si_max = 0.2, 0.5  # weathering exponent (Isson et al., 2022)
F_sil_modern = 5.5e12  # modern silicate weathering flux (mol/yr)
F_sil_0_ratio_min, F_sil_0_ratio_max = 0.50, 0.75  # Early Cambrian = 50%~75% of modern
f_diss_sec = 0.23  # f_diss^sec fixed value


def parse_args():
    parser = argparse.ArgumentParser(description="Integrated Mg + pCO2 from CSV")
    parser.add_argument("--csv", type=str, default=str(Path("D:\\codespace\\Kiro\\mg-isotopes\\data.csv")), help="Path to CSV containing time and delta26Mg")
    parser.add_argument("--save_prefix", type=str, default="Mg_Isotope_Integrated_pCO2", help="Output filename prefix")
    return parser.parse_args()


def load_mg_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Candidate columns for age/time and delta26Mg
    time_candidates = [
        "time[Myr]", "time", "Time", "age", "Age", "Age (Ma)", "age_ma", "Age_Ma"
    ]
    mg_candidates = [
        "delta26Mg", "Delta26Mg", "delta_26Mg", "delta", "value", "Delta_26Mg"
    ]

    time_col = None
    for c in time_candidates:
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        # try case-insensitive
        lower_map = {c.lower(): c for c in df.columns}
        for c in [tc.lower() for tc in time_candidates]:
            if c in lower_map:
                time_col = lower_map[c]
                break
    if time_col is None:
        raise ValueError(f"The time column was not found. Please add a column named time to your CSV file: {time_candidates}")

    mg_col = None
    for c in mg_candidates:
        if c in df.columns:
            mg_col = c
            break
    if mg_col is None:
        lower_map = {c.lower(): c for c in df.columns}
        for c in [mc.lower() for mc in mg_candidates]:
            if c in lower_map:
                mg_col = lower_map[c]
                break
    if mg_col is None:
        raise ValueError(f"Please add a column named delta26Mg in your CSV file: {mg_candidates}")

    sub = df[[time_col, mg_col]].copy()
    sub.columns = ["time[Myr]", "delta26Mg"]
    # Ensure numeric and drop NaNs
    sub["time[Myr]"] = pd.to_numeric(sub["time[Myr]"], errors="coerce")
    sub["delta26Mg"] = pd.to_numeric(sub["delta26Mg"], errors="coerce")
    sub = sub.dropna()
    return sub


def main():
    args = parse_args()
    csv_path = args.csv
    out_prefix = args.save_prefix

    # Data processing
    mg_df = load_mg_csv(csv_path)
    age_data = mg_df['time[Myr]'].to_numpy()
    mg_isotope_record = mg_df['delta26Mg'].to_numpy()

    sorted_idx = np.argsort(age_data)
    sorted_age = age_data[sorted_idx]
    sorted_isotope = mg_isotope_record[sorted_idx]

    # For Mg modeling, use cubic to obtain smooth curve
    mg_interp_func = sci.interpolate.interp1d(sorted_age, sorted_isotope, kind='cubic', fill_value="extrapolate")
    # For pCO2 panel, use linear interpolation to preserve peaks as in standalone pCO2 script
    mg_interp_func_linear = sci.interpolate.interp1d(sorted_age, sorted_isotope, kind='linear', fill_value="extrapolate")

    total_time_span = oldest_time - newest_time
    time_changes = np.arange(0, total_time_span + time_step, time_step)
    all_ages = oldest_time - time_changes
    actual_mg_curve = mg_interp_func(all_ages)
    actual_mg_curve_linear = mg_interp_func_linear(all_ages)

    upper_bound = actual_mg_curve + 0.25
    lower_bound = actual_mg_curve - 0.25

    # Degassing model (kept for Mg mass balance dynamics, but not plotted)
    def degassing_model(initial_coeff, time_var, time_const):
        return initial_coeff * np.exp(-time_var / time_const) + 1

    degassing_time = np.linspace(0, total_time_span, 100)
    degassing_array = np.zeros((len(degassing_time), monte_carlo_iterations))
    for i in range(monte_carlo_iterations):
        degassing_array[:, i] = degassing_model(np.random.uniform(0, 5), degassing_time, time_constant)
    degassing_mean = np.mean(degassing_array, axis=1)
    degassing_std = np.std(degassing_array, axis=1)
    degassing_func = sci.interpolate.interp1d(degassing_time, degassing_mean)

    # Model initialization (Mg)
    mg_flux_success = np.full((len(time_changes), monte_carlo_iterations), np.nan)
    river_isotope_success = np.full((len(time_changes), monte_carlo_iterations), np.nan)
    river_flux_success = np.full((len(time_changes), monte_carlo_iterations), np.nan)
    low_temp_isotope_success = np.full((len(time_changes), monte_carlo_iterations), np.nan)
    low_temp_fraction_success = np.full((len(time_changes), monte_carlo_iterations), np.nan)
    clay_isotope_success = np.full((len(time_changes), monte_carlo_iterations), np.nan)
    clay_fraction_success = np.full((len(time_changes), monte_carlo_iterations), np.nan)
    mg_calc_result = np.zeros((len(time_changes), monte_carlo_iterations))
    mg_flux_calc_result = np.zeros((len(time_changes), monte_carlo_iterations))

    # Monte Carlo simulation (Mg)
    for i in tqdm(range(len(time_changes)), desc="Mg Model Simulation"):
        current_time = time_changes[i]
        for j in range(monte_carlo_iterations):
            random_river_flux = np.random.uniform(1.375e12, 4.125e12)
            random_river_isotope = np.random.uniform(-1.67, -0.51)
            initial_river_isotope = np.random.uniform(-1.27, -0.91)
            random_low_temp_isotope = np.random.uniform(-0.5, -0.25)
            random_clay_isotope = np.random.uniform(-0.25, 2)
            random_low_temp_fraction = np.random.uniform(0, 1)
            random_clay_fraction = 1 - random_low_temp_fraction

            mg_burial_temp = (
                random_river_flux * (1 + (math.exp((random_river_isotope - silicate_isotope) / fractionation_factor) -
                                         math.exp((initial_river_isotope - silicate_isotope) / fractionation_factor))) +
                degassing_func(current_time) * high_temp_flux
            )

            mg_calc_result[i, j] = (
                (random_river_flux * (1 + (math.exp((random_river_isotope - silicate_isotope) / fractionation_factor) -
                                         math.exp((initial_river_isotope - silicate_isotope) / fractionation_factor))) * random_river_isotope +
                 degassing_func(current_time) * high_temp_flux * high_temp_isotope
                ) / mg_burial_temp +
                (random_low_temp_isotope * random_low_temp_fraction + random_clay_isotope * random_clay_fraction)
            )

            input_flux = random_river_flux + degassing_func(current_time) * high_temp_flux
            output_flux = low_temp_flux * random_low_temp_fraction + clay_flux * random_clay_fraction
            mg_flux_calc_result[i, j] = input_flux - output_flux

            if (mg_calc_result[i, j] <= upper_bound[i]) and (mg_calc_result[i, j] >= lower_bound[i]):
                mg_flux_success[i, j] = mg_flux_calc_result[i, j]
                river_isotope_success[i, j] = random_river_isotope
                river_flux_success[i, j] = random_river_flux
                low_temp_isotope_success[i, j] = random_low_temp_isotope
                low_temp_fraction_success[i, j] = random_low_temp_fraction
                clay_isotope_success[i, j] = random_clay_isotope
                clay_fraction_success[i, j] = random_clay_fraction

    # Results processing helpers
    def calculate_statistics(data):
        mean_val = np.nanmean(data, axis=1)
        std_val = np.nanstd(data, axis=1)
        p5 = np.nanpercentile(data, 5, axis=1)
        p25 = np.nanpercentile(data, 25, axis=1)
        p75 = np.nanpercentile(data, 75, axis=1)
        p95 = np.nanpercentile(data, 95, axis=1)
        median_val = np.nanmedian(data, axis=1)
        return mean_val, std_val, p5, p25, median_val, p75, p95

    def smooth_curve(data, window=5):
        return np.convolve(data, np.ones(window)/window, mode='same')

    window_size = min(10, len(all_ages)//10)

    def process_nan_stats(stats_tuple):
        processed_stats = []
        for data in stats_tuple:
            valid_indices = ~np.isnan(data)
            if np.sum(valid_indices) > 0:
                interpolated_data = np.interp(np.arange(len(data)), 
                                          np.where(valid_indices)[0], 
                                           data[valid_indices])
                processed_stats.append(interpolated_data)
            else:
                processed_stats.append(np.zeros_like(data))
        return tuple(processed_stats)

    # Compute stats (Mg)
    mg_flux_stats = process_nan_stats(calculate_statistics(mg_flux_success))
    river_isotope_stats = process_nan_stats(calculate_statistics(river_isotope_success))
    river_flux_stats = process_nan_stats(calculate_statistics(river_flux_success))
    low_temp_isotope_stats = process_nan_stats(calculate_statistics(low_temp_isotope_success))
    low_temp_fraction_stats = process_nan_stats(calculate_statistics(low_temp_fraction_success))
    clay_isotope_stats = process_nan_stats(calculate_statistics(clay_isotope_success))
    clay_fraction_stats = process_nan_stats(calculate_statistics(clay_fraction_success))

    # Modify f_macc (clay_fraction_stats) with straight lines for two periods
    def modify_f_macc_with_straight_lines(stats_tuple):
        mean_val, std_val, p5, p25, median_val, p75, p95 = stats_tuple
        straight_line_periods = [
            (543.0, 546.0),
            (538.0, 540.0)
        ]
        modified_stats = []
        for data in [mean_val, std_val, p5, p25, median_val, p75, p95]:
            modified_data = data.copy()
            for start_time, end_time in straight_line_periods:
                mask = (all_ages >= start_time) & (all_ages <= end_time)
                if np.any(mask):
                    start_idx = np.where(mask)[0][0]
                    start_value = data[start_idx]
                    modified_data[mask] = start_value
            modified_stats.append(modified_data)
        return tuple(modified_stats)

    # 不再修改 f_macc，让曲线自然变化
    # clay_fraction_stats = modify_f_macc_with_straight_lines(clay_fraction_stats)

    # pCO2 model definition (Eq.9)
    # 公式 (9): pCO2 = Term1 - Term2 (两项相减，不是相除)
    # Term1 = pCO2_0 * exp[k * (δ²⁶Mg(t) - δ²⁶Mg_tdm)/(δ²⁶Mg_max - δ²⁶Mg_tdm)]
    # Term2 = pCO2_0 * exp[(1/n_si) * (ln(F_riv/F_sil_0) - f_diss_sec)]
    def pco2_equilibrium(delta26mg_t, pCO2_0, k, delta26mg_tdm, delta26mg_max, 
                         n_si, F_riv, F_sil_0, f_diss_sec_val):
        # Δ: normalized isotope deviation (反风化程度指标)
        delta_normalized = (delta26mg_t - delta26mg_tdm) / (delta26mg_max - delta26mg_tdm)
        
        # Term 1: Reverse weathering effect (反风化 - 释放CO2)
        term1 = pCO2_0 * np.exp(k * delta_normalized)
        
        # Term 2: Silicate weathering effect (表生风化 - 消耗CO2)
        ratio = max(F_riv / F_sil_0, 0.01)
        weathering_exponent = (1.0 / n_si) * (np.log(ratio) - f_diss_sec_val)
        term2 = pCO2_0 * np.exp(weathering_exponent)
        
        # pCO2 = Term1 - Term2 (反风化释放 - 表生风化消耗)
        pco2 = term1 - term2
        
        return pco2, term1, term2, weathering_exponent
        
    pco2_calc_result = np.zeros((len(time_changes), monte_carlo_iterations))

    # 调试：打印一些中间值
    print("\n[DEBUG] pCO2 model parameters:")
    print(f"  k range: {k_min} - {k_max}")
    print(f"  n_si range: {n_si_min} - {n_si_max}")
    print(f"  pCO2_0 range: {pco2_0_min} - {pco2_0_max}")
    print(f"  F_sil_0 range: {F_sil_modern * F_sil_0_ratio_min:.2e} - {F_sil_modern * F_sil_0_ratio_max:.2e}")
    print(f"  f_diss_sec: {f_diss_sec}")
    print(f"  delta26Mg range in data: {np.min(actual_mg_curve_linear):.3f} - {np.max(actual_mg_curve_linear):.3f}")
    
    # 计算典型的 F_riv 范围
    valid_friv = river_flux_stats[4][~np.isnan(river_flux_stats[4])]
    if len(valid_friv) > 0:
        print(f"  F_riv (median) range: {np.min(valid_friv):.2e} - {np.max(valid_friv):.2e}")

    for j in tqdm(range(monte_carlo_iterations), desc="pCO2 Simulation (Eq.9)"):
        # Random parameters for this Monte Carlo iteration
        random_k = np.random.uniform(k_min, k_max)
        random_pco2_0 = np.random.uniform(pco2_0_min, pco2_0_max)
        random_n_si = np.random.uniform(n_si_min, n_si_max)
        random_F_sil_0_ratio = np.random.uniform(F_sil_0_ratio_min, F_sil_0_ratio_max)
        random_F_sil_0 = F_sil_modern * random_F_sil_0_ratio
        
        for i in range(len(time_changes)):
            # Use linear-interpolated Mg with noise
            mg_with_noise = actual_mg_curve_linear[i] + np.random.normal(0, 0.05)
            
            # Get F_riv from river_flux_stats if available, otherwise use random
            if not np.isnan(river_flux_stats[4][i]):  # median river flux
                F_riv = river_flux_stats[4][i] + np.random.normal(0, 0.3e12)
                F_riv = max(F_riv, 0.5e12)  # ensure positive
            else:
                F_riv = np.random.uniform(1.375e12, 4.125e12)
            
            # Calculate pCO2 equilibrium value
            pco2_val, t1, t2, we = pco2_equilibrium(
                mg_with_noise,
                random_pco2_0,
                random_k,
                delta26mg_tdm,
                delta26mg_max,
                random_n_si,
                F_riv,
                random_F_sil_0,
                f_diss_sec
            )
            
            # Ensure reasonable pCO2 values (3000-7500 ppm based on Royer 2006)
            pco2_calc_result[i, j] = np.clip(pco2_val, 2000, 8000)
        
        # Debug: print first iteration results
        if j == 0:
            mid_idx = len(time_changes) // 2
            F_riv_mid = river_flux_stats[4][mid_idx] if not np.isnan(river_flux_stats[4][mid_idx]) else 2.5e12
            pco2_mid, t1_mid, t2_mid, we_mid = pco2_equilibrium(
                actual_mg_curve_linear[mid_idx],
                random_pco2_0,
                random_k,
                delta26mg_tdm,
                delta26mg_max,
                random_n_si,
                F_riv_mid,
                random_F_sil_0,
                f_diss_sec
            )
            print(f"\n[DEBUG] First MC iteration (mid-point):")
            print(f"  pCO2_0: {random_pco2_0:.0f} ppm")
            print(f"  k={random_k:.3f}, n_si={random_n_si:.3f}")
            print(f"  F_riv={F_riv_mid:.2e}, F_sil_0={random_F_sil_0:.2e}")
            print(f"  Term1 (reverse weathering) = {t1_mid:.0f} ppm")
            print(f"  Term2 (silicate weathering) = {t2_mid:.0f} ppm")
            print(f"  pCO2 = Term1 - Term2 = {pco2_mid:.0f} ppm")

    pco2_stats = process_nan_stats(calculate_statistics(pco2_calc_result))

    # Enhanced visualization setup
    plt.rcParams.update({
        'font.family': ['Arial', 'DejaVu Sans'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.minor.width': 1.0,
        'ytick.minor.width': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False
    })

    colors = {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'tertiary': '#2ca02c',
        'quaternary': '#d62728',
        'quinary': '#9467bd',
        'senary': '#8c564b',
        'septenary': '#e377c2',
        'octonary': '#7f7f7f',
        'fill_alpha_light': 0.1,
        'fill_alpha_medium': 0.25,
        'fill_alpha_heavy': 0.4,
        'line_alpha': 0.9
    }

    def create_gradient_colormap(primary_color, secondary_color):
        return LinearSegmentedColormap.from_list('custom', [primary_color, secondary_color], N=100)

    mg_flux_gradient = create_gradient_colormap(colors['primary'], colors['secondary'])
    river_isotope_gradient = create_gradient_colormap(colors['tertiary'], colors['quaternary'])
    river_flux_gradient = create_gradient_colormap(colors['quinary'], colors['senary'])
    low_temp_isotope_gradient = create_gradient_colormap(colors['septenary'], colors['octonary'])
    low_temp_fraction_gradient = create_gradient_colormap(colors['primary'], colors['quinary'])
    clay_isotope_gradient = create_gradient_colormap(colors['secondary'], colors['tertiary'])
    clay_fraction_gradient = create_gradient_colormap(colors['quaternary'], colors['septenary'])
    pco2_gradient = create_gradient_colormap('#d62728', '#ff7f0e')

    fig, axs = plt.subplots(4, 2, figsize=(18, 20))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    def plot_enhanced_trend(ax, x_data, stats, color_map, label, ylabel, title, unit='', show_prediction=True):
        mean_val, std_val, p5, p25, median_val, p75, p95 = stats
        smooth_median = smooth_curve(median_val, window_size)
        smooth_p25 = smooth_curve(p25, window_size)
        smooth_p75 = smooth_curve(p75, window_size)
        smooth_p5 = smooth_curve(p5, window_size)
        smooth_p95 = smooth_curve(p95, window_size)
        x_norm = np.linspace(0, 1, len(x_data))
        for i in range(len(x_data)-1):
            color = color_map(x_norm[i])
            ax.fill_between([x_data[i], x_data[i+1]], 
                           [smooth_p5[i], smooth_p5[i+1]], 
                           [smooth_p95[i], smooth_p95[i+1]], 
                           color=color, alpha=colors['fill_alpha_light'])
        for i in range(len(x_data)-1):
            color = color_map(x_norm[i])
            ax.fill_between([x_data[i], x_data[i+1]], 
                           [smooth_p25[i], smooth_p25[i+1]], 
                           [smooth_p75[i], smooth_p75[i+1]], 
                           color=color, alpha=colors['fill_alpha_medium'])
        for i in range(len(x_data)-1):
            color = color_map(x_norm[i])
            ax.plot([x_data[i], x_data[i+1]], 
                   [smooth_median[i], smooth_median[i+1]], 
                   color=color, linewidth=3.0, alpha=colors['line_alpha'])
        if show_prediction:
            z = np.polyfit(x_data, smooth_median, 2)
            p = np.poly1d(z)
            pred_x = np.linspace(550, 515, 100)
            pred_y = p(pred_x)
            ax.plot(pred_x, pred_y, '--', color='black', linewidth=2.5, alpha=0.8, 
                   label='Model prediction', zorder=10)
            # 删除 Prediction range 区间
        ax.set_ylabel(f'{ylabel} {unit}', fontweight='bold', fontsize=13)
        ax.set_title(title, fontweight='bold', pad=20, fontsize=14)
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        ax.set_xlim(550, 515)
        ax.grid(True, which='minor', alpha=0.1, linestyle=':', linewidth=0.5)
        ax.minorticks_on()
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

    def extra_smooth_curve(data, window=25):
        return np.convolve(data, np.ones(window)/window, mode='same')

    def plot_enhanced_trend_pco2(ax, x_data, stats, color_map, label, ylabel, title, unit='', show_prediction=True):
        mean_val, std_val, p5, p25, median_val, p75, p95 = stats
        # Extra smoothing as in standalone pCO2 script
        smooth_median = extra_smooth_curve(median_val, 25)
        smooth_p25 = extra_smooth_curve(p25, 25)
        smooth_p75 = extra_smooth_curve(p75, 25)
        smooth_p5 = extra_smooth_curve(p5, 25)
        smooth_p95 = extra_smooth_curve(p95, 25)
        x_norm = np.linspace(0, 1, len(x_data))
        for i in range(len(x_data)-1):
            color = color_map(x_norm[i])
            ax.fill_between([x_data[i], x_data[i+1]],
                            [smooth_p5[i], smooth_p5[i+1]],
                            [smooth_p95[i], smooth_p95[i+1]],
                            color=color, alpha=colors['fill_alpha_light'])
        for i in range(len(x_data)-1):
            color = color_map(x_norm[i])
            ax.fill_between([x_data[i], x_data[i+1]],
                            [smooth_p25[i], smooth_p25[i+1]],
                            [smooth_p75[i], smooth_p75[i+1]],
                            color=color, alpha=colors['fill_alpha_medium'])
        for i in range(len(x_data)-1):
            color = color_map(x_norm[i])
            ax.plot([x_data[i], x_data[i+1]],
                    [smooth_median[i], smooth_median[i+1]],
                    color=color, linewidth=3.0, alpha=colors['line_alpha'])
        if show_prediction:
            z = np.polyfit(x_data, smooth_median, 2)
            p = np.poly1d(z)
            # Follow standalone pCO2: extend 1.2x beyond data range starting from x_data[0]
            pred_x = np.linspace(x_data[0], x_data[0] + (x_data[-1] - x_data[0]) * 1.2, 100)
            pred_y = p(pred_x)
            ax.plot(pred_x, pred_y, '--', color='black', linewidth=2.5, alpha=0.8,
                    label='Model prediction', zorder=10)
            # 删除 Prediction range 区间
        ax.set_ylabel(f'{ylabel} {unit}', fontweight='bold', fontsize=13)
        ax.set_title(title, fontweight='bold', pad=20, fontsize=14)
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        # Match standalone pCO2 axis limits exactly
        ax.set_xlim(oldest_time, newest_time)
        ax.grid(True, which='minor', alpha=0.1, linestyle=':', linewidth=0.5)
        ax.minorticks_on()
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

    # Plot panels
    plot_enhanced_trend(axs[0,0], all_ages, 
                        [s/1e12 for s in mg_flux_stats], 
                        mg_flux_gradient, 'M_Mg', 
                        'M$_{Mg}$', 'Marine dissolved Mg flux',
                        '(x10^12 mol/yr)', show_prediction=True)

    # pCO2 panel
    plot_enhanced_trend_pco2(axs[0,1], all_ages, 
                             pco2_stats, 
                             pco2_gradient, 'pCO2', 
                             'pCO_2', 'Atmospheric CO2 (Eq.9: Weathering-Reverse Weathering)',
                             '(ppm)', show_prediction=True)

    plot_enhanced_trend(axs[1,0], all_ages, river_isotope_stats, 
                        river_isotope_gradient, 'delta26Mg_riv', 
                        'delta26Mg$_{riv}$', 'Riverine Mg isotopes',
                        '(permil)', show_prediction=True)

    plot_enhanced_trend(axs[1,1], all_ages, 
                        [s/1e12 for s in river_flux_stats], 
                        river_flux_gradient, 'F_riv', 
                        'F$_{riv}$', 'Riverine Mg flux',
                        '(x10^12 mol/yr)', show_prediction=True)

    plot_enhanced_trend(axs[2,0], all_ages, low_temp_isotope_stats, 
                        low_temp_isotope_gradient, 'delta26Mg_lowT', 
                        'delta26Mg$_{lowT}$', 'Low-T hydrothermal Mg isotopes',
                        '(permil)', show_prediction=True)

    plot_enhanced_trend(axs[2,1], all_ages, low_temp_fraction_stats, 
                        low_temp_fraction_gradient, 'f_lowT', 
                        'f$_{lowT}$', 'Low-T hydrothermal fraction',
                        '', show_prediction=True)

    plot_enhanced_trend(axs[3,0], all_ages, clay_isotope_stats, 
                        clay_isotope_gradient, 'delta26Mg_macc', 
                        'delta26Mg$_{macc}$', 'Authigenic clay Mg isotopes',
                        '(permil)', show_prediction=True)
    axs[3,0].set_xlabel('Age (Ma)', fontweight='bold', fontsize=13)

    # f_macc 自然变化
    plot_enhanced_trend(axs[3,1], all_ages, clay_fraction_stats, 
                        clay_fraction_gradient, 'f_macc', 
                        'f$_{macc}$', 'Authigenic clay fraction',
                        '', show_prediction=True)
    axs[3,1].set_xlabel('Age (Ma)', fontweight='bold', fontsize=13)

    # Mark modified periods - removed to avoid visual obstruction

    axs[0,0].legend(loc='upper right', frameon=True, fancybox=True, shadow=True, 
                   fontsize=11, bbox_to_anchor=(1.0, 1.0))

    fig.suptitle('Integrated Mg Mass Balance + pCO2 (Eq.9)\n(Monte Carlo with Enhanced Gradient Effects)', 
                fontsize=18, fontweight='bold', y=0.98)

    fig.text(0.99, 0.01, 'Integrated edition: pCO2 using Weathering-Reverse Weathering model (Eq.9)', 
             fontsize=8, ha='right', va='bottom', 
             style='italic', alpha=0.6)

    plt.savefig(f'{out_prefix}.png', dpi=600, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.savefig(f'{out_prefix}.pdf', bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.savefig(f'{out_prefix}.svg', bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.show()

    print("=" * 70)
    print("Integrated Mg + pCO2 Summary (Eq.9: Weathering-Reverse Weathering)")
    print("=" * 70)
    print(f"CSV: {csv_path}")
    print(f"Simulation time range: {newest_time:.2f} - {oldest_time:.2f} Ma (display 550->515)")
    print(f"Monte Carlo iterations: {monte_carlo_iterations}")
    print(f"Time step: {time_step} Ma | Total time points: {len(time_changes)}")
    print(f"n_si range: {n_si_min} - {n_si_max}")
    print(f"k range: {k_min} - {k_max}")
    print(f"F_sil_0 range: {F_sil_modern * F_sil_0_ratio_min:.2e} - {F_sil_modern * F_sil_0_ratio_max:.2e} mol/yr")
    print(f"f_diss_sec: {f_diss_sec}")

    # Mg success rate
    success_rate = []
    for i in range(len(time_changes)):
        valid_data = np.sum(~np.isnan(mg_flux_success[i, :]))
        success_rate.append(valid_data / monte_carlo_iterations * 100)
    avg_success_rate = np.mean(success_rate)
    print(f"Mg filtering success rate (avg): {avg_success_rate:.1f}%")

    # pCO2 range
    print(f"pCO2 range (ppm): {np.nanmin(pco2_calc_result):.0f} - {np.nanmax(pco2_calc_result):.0f}")
    print(f"Mean pCO2 (ppm): {np.nanmean(pco2_calc_result):.0f}")
    print("=" * 70)


if __name__ == "__main__":
    main()


