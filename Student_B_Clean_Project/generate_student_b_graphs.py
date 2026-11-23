"""Student B ENPM667 plotting and metrics from real simulation CSV data.
Run: python generate_student_b_graphs.py --data results.csv --out figures
Optional: --fixed fixed_gain_results.csv --dob dob_results.csv --baseline baseline_results.csv
This script does NOT simulate or fabricate control-system outputs.
"""
import argparse, csv, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'white',
    'savefig.facecolor':'white','text.color':'black','axes.labelcolor':'black',
    'axes.edgecolor':'black','xtick.color':'black','ytick.color':'black',
    'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
    'axes.titleweight':'bold','lines.linewidth':1.5,'pdf.fonttype':42})

N=6
# All plots intentionally use different black line patterns/markers, not coloured curves.
STYLES=['-','--','-.',':',(0,(5,2,1,2)),(0,(2,2))]

def read_data(path):
    df=pd.read_csv(path)
    if 'time' not in df: raise ValueError(f'{path}: required column time is missing')
    if len(df)<3: raise ValueError(f'{path}: need at least three time samples')
    df=df.sort_values('time').reset_index(drop=True)
    t=df.time.to_numpy(dtype=float)
    if not np.all(np.isfinite(t)) or np.any(np.diff(t)<=0):
        raise ValueError(f'{path}: time must be finite and strictly increasing')
    return df

def need(df, names, name):
    missing=[c for c in names if c not in df]
    if missing:
        print(f'SKIP {name}: missing {", ".join(missing)}')
        return False
    for c in names:
        if not np.all(np.isfinite(pd.to_numeric(df[c],errors='coerce'))):
            print(f'SKIP {name}: {c} has NaN, Inf, or non-numeric values')
            return False
    return True

def save(fig, out, stem):
    fig.tight_layout(pad=1.6)
    fig.savefig(out/(stem+'.png'),dpi=300,bbox_inches='tight')
    fig.savefig(out/(stem+'.pdf'),bbox_inches='tight')
    plt.close(fig)
    print('SAVED',stem)

def panel(ax, df, cols, title, ylabel, labels=None):
    for i,c in enumerate(cols):
        ax.plot(df.time,df[c],color='black',ls=STYLES[i%len(STYLES)],
                label=(labels[i] if labels else c),alpha=0.95)
    ax.set(title=title,xlabel='Time (s)',ylabel=ylabel)
    ax.grid(True,color='#dddddd',linewidth=.5)
    ax.legend(fontsize=7,ncol=3,loc='best',frameon=False)

def group(df,out,stem,cols,title,ylabel):
    if not need(df,cols,stem):return
    fig,ax=plt.subplots(figsize=(9,4.7))
    panel(ax,df,cols,title,ylabel)
    save(fig,out,stem)

def paired(df,out,stem,cols_a,cols_b,titles,units):
    if not need(df,cols_a+cols_b,stem):return
    fig,axs=plt.subplots(1,2,figsize=(12,4.4),sharex=True)
    panel(axs[0],df,cols_a,titles[0],units[0])
    panel(axs[1],df,cols_b,titles[1],units[1])
    save(fig,out,stem)

def true_est_grid(df,out,stem,truth_prefix,est_prefix,ids,units):
    cols=[f'{p}{i}' for i in ids for p in (truth_prefix,est_prefix)]
    if not need(df,cols,stem):return
    n=len(ids);fig,axs=plt.subplots((n+1)//2,2,figsize=(12,3.2*((n+1)//2)),squeeze=False)
    for ax,i in zip(axs.flat,ids):
        ax.plot(df.time,df[f'{truth_prefix}{i}'],color='black',lw=1.8,label='True')
        ax.plot(df.time,df[f'{est_prefix}{i}'],color='black',lw=1.3,ls='--',label='FTRE estimate')
        ax.set(title=f'Vehicle {i}',xlabel='Time (s)',ylabel=units)
        ax.grid(True,color='#dddddd',lw=.5);ax.legend(frameon=False,fontsize=8)
    for ax in list(axs.flat)[n:]:ax.axis('off')
    save(fig,out,stem)

def compute_metrics(df,out):
    cols=[f'e{i}' for i in range(1,7)]
    if not need(df,cols,'RMSE metrics'):return
    t=df.time.to_numpy(); duration=t[-1]-t[0]
    rows=[]
    for i,c in enumerate(cols,1):
        e=df[c].to_numpy(dtype=float)
        rmse=np.sqrt(np.trapezoid(e*e,t)/duration)
        peak=np.max(np.abs(e))
        rows.append({'vehicle':i,'rmse_m':float(rmse),'rmse_cm':float(rmse*100),
                     'peak_abs_error_m':float(peak)})
    pd.DataFrame(rows).to_csv(out/'rmse_table.csv',index=False)
    print('SAVED rmse_table.csv')
    # Peak/RMSE comparisons are empirical, not a mathematical proof of string stability.
    return rows

def compare_runs(df,other,out,stem,description):
    cols=[f'e{i}' for i in range(1,7)]
    if not (need(df,cols,stem) and need(other,cols,stem)):return
    if not np.isclose(df.time.iloc[0],other.time.iloc[0]) or not np.isclose(df.time.iloc[-1],other.time.iloc[-1]):
        print('WARNING:',stem,'uses different simulation windows; check comparison validity')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    panel(axs[0],df,cols,'Adaptive STISMC','Spacing error (m)')
    panel(axs[1],other,cols,description,'Spacing error (m)')
    save(fig,out,stem)

def comparison_table(runs,out):
    rows=[]
    for method,df in runs.items():
        cols=[f'e{i}' for i in range(1,7)]
        if not need(df,cols,f'{method} RMSE comparison'):continue
        t=df.time.to_numpy();d=t[-1]-t[0]
        row={'method':method}
        for i in range(1,7):
            e=df[f'e{i}'].to_numpy(dtype=float)
            row[f'vehicle_{i}_rmse_cm']=float(100*np.sqrt(np.trapezoid(e*e,t)/d))
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(out/'table_II_style_comparison.csv',index=False)
        print('SAVED table_II_style_comparison.csv (computed from provided runs)')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',required=True,help='Adaptive STISMC results CSV')
    ap.add_argument('--out',default='figures')
    ap.add_argument('--fixed',help='Fixed gain results CSV')
    ap.add_argument('--dob',help='DOB ISMC results CSV')
    ap.add_argument('--baseline',help='Baseline ISMC results CSV')
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    df=read_data(a.data)
    e=[f'e{i}' for i in range(1,7)]
    eb=[f'ebar{i}' for i in range(1,7)]
    s=[f's{i}' for i in range(1,7)]
    S=[f'S{i}' for i in range(1,7)]
    x=[f'x{i}' for i in range(0,7)]
    v=[f'v{i}' for i in range(0,7)]
    u=[f'u{i}' for i in range(1,7)]
    L=[f'Ltilde{i}' for i in range(1,7)]
    group(df,out,'01_mcth_spacing_error',e,'CTH spacing error with MCTH controller','Error (m)')
    # This is an actual separate no-MCTH RUN, not ebar vs e from same MCTH run.
    group(df,out,'02_modified_error_ebar',eb,'Modified MCTH error (diagnostic)','Modified error (m)')
    group(df,out,'03_integral_surfaces',s,'Integral sliding surfaces','s (m)')
    group(df,out,'04_coupled_surfaces',S,'Coupled integral sliding surfaces','S (m)')
    paired(df,out,'05_positions_velocities',x,v,('Vehicle positions','Vehicle velocities'),('Position (m)','Velocity (m/s)'))
    group(df,out,'06_control_inputs',u,'Control inputs','Control input (plant units)')
    group(df,out,'07_matched_uncertainty_residual',L,'Matched uncertainty residual','L tilde (defined model units)')
    group(df,out,'08_adaptive_Omega',[f'Omega{i}' for i in range(1,7)],'Adaptive gain parameter Omega','Omega')
    group(df,out,'09_gain_k1',[f'k1{i}' for i in range(1,7)],'Adaptive super-twisting gain k1','k1')
    paired(df,out,'10_Omega_vs_xi', ['Omega2'],['xi_abs2'],('Omega2','Absolute xi2'),('Omega','|xi|'))
    true_est_grid(df,out,'11_FTRE_acceleration','a','ahat',range(2,7),'Acceleration (m/s²)')
    true_est_grid(df,out,'12_FTRE_mismatched_disturbance','dv','dvhat',range(1,7),'Disturbance (m/s)')
    compute_metrics(df,out)
    runs={'adaptive_STISMC':df}
    for label,path in [('fixed_STISMC',a.fixed),('DOB_ISMC',a.dob),('baseline_ISMC',a.baseline)]:
        if path:
            other=read_data(path);runs[label]=other
            compare_runs(df,other,out,'compare_'+label,' '.join(label.split('_')))
    comparison_table(runs,out)
    print('DONE. Only plots with all required real data columns were created.')

if __name__=='__main__':main()
