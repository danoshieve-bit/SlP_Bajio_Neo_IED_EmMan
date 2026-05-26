"""
Dashboard Econométrico — Región del Bajío (VERSIÓN FINAL PURIFICADA)
========================================================
UI: Fondo Crema, Insights de Negocio, Títulos limpios.
ETL: IED desde CSV Local, Empleo/Exp desde INEGI.
Fixes: _parse_trim restaurado (cero duplicación), Clave Ags corregida.
"""

import os
import json
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import threading
from datetime import date

import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import plotly.graph_objects as go
import plotly.express as px

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from linearmodels.panel import PanelOLS

# ══════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════
load_dotenv()
TOKEN_INEGI = os.getenv("INEGI_TOKEN", "2c63db48-9a6a-4468-be5b-8ab85da04eb6")

ESTADOS_BAJIO = {
    "San Luis Potosí": "24",
    "Aguascalientes":  "01",
    "Guanajuato":      "11",
    "Jalisco":         "14",
    "Querétaro":       "22",
}

COLORES_ESTADOS = {
    "San Luis Potosí": "#E63946",  
    "Aguascalientes":  "#1B4F72",
    "Guanajuato":      "#2874A6",
    "Jalisco":         "#5DADE2",
    "Querétaro":       "#A6ACAF",
}

YEARS = list(range(2015, 2026))

AZUL_OSCURO = "#1B4F72"
CAFE        = "#8B5E3C"
NARANJA     = "#D35400"
BG          = "#F4EFEA"  
CARD_BG     = "#FFFFFF"
TEXT_PRIM   = "#2C3E50"  
TEXT_SEC    = "#6B6B6B"

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'Helvetica Neue', Arial, sans-serif", color=TEXT_PRIM, size=12),
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(orientation="h", y=-0.15, x=0, font_size=11),
)

INDICADORES_EMPLEO = {
    "Aguascalientes":  "702846",
    "Guanajuato":      "702855",
    "Jalisco":         "702858",
    "Querétaro":       "702866",
    "San Luis Potosí": "702868",
}

# CLAVES CORREGIDAS EN SECUENCIA PERFECTA
INDICADORES_EXPORTACIONES = {
    "Aguascalientes":  "739276",  
    "Guanajuato":      "739277",
    "Jalisco":         "739278",
    "Querétaro":       "739279",
    "San Luis Potosí": "739280",
}

GEOJSON_URLS = [
    "https://raw.githubusercontent.com/PhantomInsights/mexican-geojson/main/src/states/states.json",
    "https://raw.githubusercontent.com/angelnmara/geojson/master/mexicoHigh.json",
]

VARS_DEF   = [("empleo","Empleo Manufacturero"),("ied","Inversión Extranjera Directa (IED)"),("exportaciones","Exportaciones Manufactureras")]
VAR_COL    = {"empleo":"Empleo_Manufacturero","ied":"IED","exportaciones":"Exportaciones"}
VAR_LABEL  = {
    "empleo":        "Empleo Manufacturero (personas)",
    "ied":           "Inversión Extranjera Directa (IED) (M USD)",
    "exportaciones": "Exportaciones Manufactureras (M USD)",
}

# ══════════════════════════════════════════════
# ETL — ESTRICTO (100% DATOS REALES Y CSV)
# ══════════════════════════════════════════════
def fetch_inegi_serie(indicador, fuente="BIE", geo="00"):
    url = (
        f"https://www.inegi.org.mx/app/api/indicadores/desarrolladores/"
        f"jsonxml/INDICATOR/{indicador}/es/{geo}/false/{fuente}/2.0/{TOKEN_INEGI}?type=json"
    )
    try:
        r = requests.get(url, timeout=15); r.raise_for_status()
        data = r.json()
        rows = []
        for serie in data.get("Series",[]):
            for s in serie.get("OBSERVATIONS",[]):
                if s["OBS_VALUE"] not in (None,"","N/A"):
                    try: rows.append({"fecha":s["TIME_PERIOD"],"valor":float(s["OBS_VALUE"])})
                    except: pass
        if not rows: raise ValueError("vacío")
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"❌ Error API INEGI ({indicador}): {e}")
        return pd.DataFrame(columns=["fecha","valor"])

def _mensual_a_trim(df, estado, col):
    if df.empty: return pd.DataFrame(columns=["Estado","Año","Trimestre",col])
    df["fecha"] = pd.to_datetime(df["fecha"], format="%Y/%m", errors="coerce")
    df = df.dropna(subset=["fecha"])
    df["Año"] = df["fecha"].dt.year; df["Mes"] = df["fecha"].dt.month
    df = df[(df["Año"]>=2015)&(df["Año"]<=2025)]
    df["Trimestre"] = df["Mes"].apply(lambda m:(m-1)//3+1)
    return df.groupby([pd.Series([estado]*len(df),name="Estado"),"Año","Trimestre"])["valor"].mean().reset_index().rename(columns={"valor":col})

# RESTAURADA A SU VERSIÓN ORIGINAL SIN DUPLICACIONES
def _parse_trim(df, estado, col):
    if df.empty: return pd.DataFrame(columns=["Estado","Año","Trimestre",col])
    rows=[]
    for _,row in df.iterrows():
        t=str(row["fecha"]).replace("-","/")
        try:
            p=t.split("/")
            yr=int(p[0])
            qn=int(p[1].replace("Q","").replace("T",""))
            if 2015<=yr<=2025 and (1 <= qn <= 4): 
                rows.append({"Estado":estado,"Año":yr,"Trimestre":qn,col:row["valor"]})
        except: pass
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Estado","Año","Trimestre",col])

def procesar_empleo():
    print("📥 Descargando Empleo Manufacturero...")
    frames=[]
    for estado,ind in INDICADORES_EMPLEO.items():
        df = fetch_inegi_serie(ind,fuente="BIE-BISE")
        if not df.empty:
            df_t = _mensual_a_trim(df,estado,"Empleo_Manufacturero")
            if not df_t.empty:
                df_t["Empleo_Manufacturero"] = df_t["Empleo_Manufacturero"].round(0).astype(int)
                frames.append(df_t)
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=["Estado","Año","Trimestre","Empleo_Manufacturero"])

def procesar_ied():
    print("📁 Cargando IED desde archivo local (ied_historica.csv)...")
    try:
        try:
            df = pd.read_csv("ied_historica.csv", encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv("ied_historica.csv", encoding="latin1")
            
        df = df[["entidad", "anio", "trimestre", "millones_de_dolares"]].copy()
        df.rename(columns={
            "entidad": "Estado",
            "anio": "Año",
            "trimestre": "Trimestre",
            "millones_de_dolares": "IED"
        }, inplace=True)
        
        df["Estado"] = df["Estado"].astype(str).str.strip()
        df["Estado"] = df["Estado"].replace({
            "San Luis Potosi": "San Luis Potosí", 
            "Queretaro": "Querétaro"
        })
        
        df = df[df["Estado"].isin(ESTADOS_BAJIO.keys())]
        
        for col in ["Año", "Trimestre", "IED"]: 
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
        df = df.dropna(subset=["Año", "Trimestre", "IED"])
        df = df[(df["Año"] >= 2015) & (df["Año"] <= 2025)]
        
        df = df.groupby(["Estado", "Año", "Trimestre"])["IED"].sum().reset_index()
        return df[["Estado", "Año", "Trimestre", "IED"]]
        
    except FileNotFoundError:
        print("❌ Error: No se encontró 'ied_historica.csv'.")
        return pd.DataFrame(columns=["Estado", "Año", "Trimestre", "IED"])
    except Exception as e:
        print(f"❌ Error archivo local IED: {e}")
        return pd.DataFrame(columns=["Estado", "Año", "Trimestre", "IED"])

def procesar_exportaciones():
    print("📥 Descargando Exportaciones...")
    frames=[]
    for estado,ind in INDICADORES_EXPORTACIONES.items():
        df = fetch_inegi_serie(ind,fuente="BIE-BISE")
        if not df.empty:
            df_t = _parse_trim(df,estado,"Exportaciones")
            if not df_t.empty: frames.append(df_t)
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=["Estado","Año","Trimestre","Exportaciones"])

def construir_panel(df_emp,df_ied,df_exp):
    keys=["Estado","Año","Trimestre"]
    panel=pd.merge(df_emp, df_ied, on=keys, how="left")
    panel=pd.merge(panel, df_exp, on=keys, how="left")
    
    if panel.empty:
        return panel
        
    panel=panel[panel["Estado"].isin(ESTADOS_BAJIO.keys())].copy()
    panel=panel.sort_values(["Estado","Año","Trimestre"]).reset_index(drop=True)
    panel["Var_Empleo_pct"]=(panel.groupby("Estado")["Empleo_Manufacturero"].pct_change()*100).round(2)
    panel["Periodo"]=panel["Año"].astype(str)+" Q"+panel["Trimestre"].astype(str)
    
    if "Exportaciones" in panel.columns:
        panel["Exportaciones"]=panel.groupby("Estado")["Exportaciones"].transform(lambda x:x.interpolate(limit_direction="both"))
    return panel

def cargar_geojson():
    for url in GEOJSON_URLS:
        try:
            r=requests.get(url,timeout=10)
            if r.status_code==200: return r.json()
        except: continue
    return {"type":"FeatureCollection","features":[
        {"type":"Feature","id":"Aguascalientes","properties":{"name":"Aguascalientes"},"geometry":{"type":"Polygon","coordinates":[[[-102.31,21.63],[-102.05,21.63],[-101.92,21.82],[-101.90,22.08],[-102.08,22.22],[-102.31,22.17],[-102.50,22.01],[-102.55,21.78],[-102.31,21.63]]]}},
        {"type":"Feature","id":"Guanajuato","properties":{"name":"Guanajuato"},"geometry":{"type":"Polygon","coordinates":[[[-102.55,19.92],[-101.20,19.85],[-100.80,20.10],[-100.02,20.55],[-99.88,21.15],[-100.15,21.55],[-100.65,21.75],[-101.40,21.85],[-102.10,21.62],[-102.55,21.32],[-102.85,21.00],[-102.55,19.92]]]}},
        {"type":"Feature","id":"Jalisco","properties":{"name":"Jalisco"},"geometry":{"type":"Polygon","coordinates":[[[-105.42,19.05],[-104.70,18.70],[-103.50,18.85],[-103.00,19.00],[-102.55,19.92],[-102.85,21.00],[-102.55,21.32],[-102.10,21.62],[-101.55,21.90],[-101.60,22.42],[-102.30,22.72],[-103.00,22.45],[-103.80,22.00],[-104.70,21.00],[-105.42,20.20],[-105.42,19.05]]]}},
        {"type":"Feature","id":"Querétaro","properties":{"name":"Querétaro"},"geometry":{"type":"Polygon","coordinates":[[[-100.02,20.55],[-99.48,20.30],[-99.05,20.50],[-98.90,20.85],[-99.00,21.15],[-99.30,21.52],[-99.88,21.55],[-100.15,21.55],[-99.88,21.15],[-100.02,20.55]]]}},
        {"type":"Feature","id":"San Luis Potosí","properties":{"name":"San Luis Potosí"},"geometry":{"type":"Polygon","coordinates":[[[-102.55,21.32],[-101.55,21.90],[-101.20,21.85],[-100.65,21.75],[-100.15,21.55],[-99.88,21.55],[-99.30,21.52],[-99.05,21.95],[-98.80,22.55],[-99.05,23.65],[-99.65,24.00],[-100.80,24.00],[-101.60,23.45],[-102.10,22.85],[-102.55,22.17],[-102.55,21.32]]]}},
    ]}

# ══════════════════════════════════════════════
# CACHÉ
# ══════════════════════════════════════════════
_cache={"panel":None,"geojson":None,"fecha":None,"lock":threading.Lock()}

def get_datos():
    hoy=date.today()
    if _cache["panel"] is not None and _cache["fecha"]==hoy:
        return _cache["panel"],_cache["geojson"]
    with _cache["lock"]:
        if _cache["panel"] is not None and _cache["fecha"]==hoy:
            return _cache["panel"],_cache["geojson"]
        df_emp=procesar_empleo(); df_ied=procesar_ied(); df_exp=procesar_exportaciones()
        panel=construir_panel(df_emp,df_ied,df_exp); geo=cargar_geojson()
        _cache["panel"]=panel; _cache["geojson"]=geo; _cache["fecha"]=hoy
    return _cache["panel"],_cache["geojson"]

PANEL,GEOJSON=get_datos()
if PANEL.empty:
    AÑOS_DISPONIBLES = list(range(2018, 2026))
else:
    AÑOS_DISPONIBLES=sorted(PANEL["Año"].unique())

# ══════════════════════════════════════════════
# ECONOMETRÍA
# ══════════════════════════════════════════════
def calcular_econometria(df, vars_x, lags=0):
    if df.empty or not vars_x: return None
    cols=["Empleo_Manufacturero"]+vars_x
    
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols: return None
    
    sub=df[["Estado","Año","Trimestre","Periodo"]+cols].dropna().copy()
    sub=sub.sort_values(["Estado","Año","Trimestre"])
    
    for c in vars_x:
        if lags > 0:
            sub[c] = sub.groupby("Estado")[c].shift(lags)
            
    sub=sub.dropna()
    if len(sub)<20: return None
    
    for c in cols:
        sub[f"Crec_{c}"]=sub.groupby("Estado")[c].pct_change()*100
        
    sub=sub.dropna()
    sub["t"]=(sub["Año"]-sub["Año"].min())*4+sub["Trimestre"]
    sub_index=sub.set_index(["Estado","t"])
    
    Y=sub_index["Crec_Empleo_Manufacturero"]
    X=sub_index[[f"Crec_{c}" for c in vars_x]]
    
    try:
        mod=PanelOLS(Y,X,entity_effects=True,time_effects=True)
        res=mod.fit(cov_type="clustered",cluster_entity=True)
        dw_stat=durbin_watson(res.resids.values)
        
        pred = res.fitted_values
        sub_index["Prediccion"] = pred
        df_pred = sub_index.reset_index()[["Estado", "Periodo", "Crec_Empleo_Manufacturero", "Prediccion"]]
    except: return None
    
    vifs={}
    if len(vars_x)>1:
        X_vif=sub_index[[f"Crec_{c}" for c in vars_x]].reset_index(drop=True)
        try:
            for i,c in enumerate(X_vif.columns):
                vifs[c.replace("Crec_","")]=round(float(variance_inflation_factor(X_vif.values.astype(float),i)),2)
        except: vifs={c:float("nan") for c in vars_x}
    else: vifs={vars_x[0]:1.0}
    
    return {"coefs":res.params,"pvals":res.pvalues,"r2_within":round(res.rsquared,4),
            "n_obs":int(res.nobs),"dw":round(dw_stat,3),"vifs":vifs,
            "corr":sub_index[[f"Crec_{c}" for c in cols]].corr().round(3),
            "df_pred": df_pred}

# ══════════════════════════════════════════════
# FIGURAS
# ══════════════════════════════════════════════
H_CHART = 450

def fig_series(df, variable, estados, tipo="line"):
    col=VAR_COL.get(variable,"Empleo_Manufacturero")
    if col not in df.columns or df.empty: return go.Figure().update_layout(**PLOT_LAYOUT, title="Datos no disponibles")
    
    fig=go.Figure()
    for est in estados:
        sub=df[df["Estado"]==est].sort_values(["Año","Trimestre"])
        if tipo=="line":
            fig.add_trace(go.Scatter(x=sub["Periodo"],y=sub[col],name=est,mode="lines+markers",
                line=dict(color=COLORES_ESTADOS[est],width=2.5),marker=dict(size=5),
                hovertemplate=f"<b>{est}</b><br>%{{x}}<br>{VAR_LABEL[variable]}: %{{y:,.1f}}<extra></extra>"))
        elif tipo=="area":
            fig.add_trace(go.Scatter(x=sub["Periodo"],y=sub[col],name=est,mode="lines",
                stackgroup='one', line=dict(color=COLORES_ESTADOS[est],width=1),
                hovertemplate=f"<b>{est}</b><br>%{{x}}<br>{VAR_LABEL[variable]}: %{{y:,.1f}}<extra></extra>"))
        else:
            fig.add_trace(go.Bar(x=sub["Periodo"],y=sub[col],name=est,
                marker_color=COLORES_ESTADOS[est],
                hovertemplate=f"<b>{est}</b><br>%{{x}}<br>{VAR_LABEL[variable]}: %{{y:,.1f}}<extra></extra>"))
    
    fig.update_layout(**{**PLOT_LAYOUT,
        "yaxis":dict(title=VAR_LABEL[variable],gridcolor="#EEE",tickformat=","),
        "xaxis":dict(tickangle=-45,tickfont_size=9),
        "height":H_CHART,
        "barmode":"group"})
    return fig

def fig_mapa(df, variable, estados, yr_to, geojson):
    col=VAR_COL.get(variable,"Empleo_Manufacturero")
    if col not in df.columns or df.empty: return go.Figure().update_layout(**PLOT_LAYOUT, title="Datos no disponibles")
    
    sub = df[df["Estado"].isin(estados)].copy()
    valid_years = sub.dropna(subset=[col])["Año"].unique()
    actual_yr = yr_to
    if len(valid_years) > 0 and yr_to not in valid_years:
        actual_yr = max(y for y in valid_years if y <= yr_to)
    elif len(valid_years) == 0:
        return go.Figure().update_layout(**PLOT_LAYOUT, title=f"Sin datos para el mapa de {VAR_LABEL[variable]}")
        
    sub = sub[sub["Año"] == actual_yr]
    grp=sub.groupby("Estado")[col].mean().reset_index().rename(columns={col:"valor"})
    grp=pd.DataFrame({"Estado":list(ESTADOS_BAJIO.keys())}).merge(grp,on="Estado",how="left")
    
    fig=go.Figure(go.Choropleth(
        geojson=geojson,locations=grp["Estado"],z=grp["valor"],
        featureidkey="properties.name",
        colorscale=[[0,"#B5D4F4"],[0.5,"#1A5599"],[1,"#0C3460"]],
        marker_line_color="white",marker_line_width=1.5,
        colorbar=dict(title=dict(text=VAR_LABEL[variable],font_size=10),thickness=14,len=0.75),
        hovertemplate="<b>%{location}</b><br>"+VAR_LABEL[variable]+f" ({actual_yr}): %{{z:,.1f}}<extra></extra>",
    ))
    fig.update_geos(fitbounds="locations",visible=False,showland=True,landcolor="#F5F3EF",showframe=False)
    fig.update_layout(**{**PLOT_LAYOUT,"height":H_CHART,"margin":dict(l=0,r=0,t=20,b=0)})
    return fig

def fig_scatter_animado(df, var_x, var_y, estados):
    col_x=VAR_COL.get(var_x,"IED")
    col_y=VAR_COL.get(var_y,"Empleo_Manufacturero")
    if col_x not in df.columns or col_y not in df.columns or df.empty: return go.Figure().update_layout(**PLOT_LAYOUT, title="Datos insuficientes")
    
    sub=df[df["Estado"].isin(estados)].copy()
    sub_anual=sub.groupby(["Estado","Año"]).agg(
        x=(col_x,"mean"), y=(col_y,"mean"),
        size_col=("Empleo_Manufacturero","mean")
    ).reset_index()
    
    all_years = sorted(sub_anual["Año"].unique())
    idx = pd.MultiIndex.from_product([estados, all_years], names=['Estado', 'Año'])
    sub_anual = sub_anual.set_index(['Estado', 'Año']).reindex(idx).reset_index()
    
    sub_anual["x"] = sub_anual.groupby("Estado")["x"].transform(lambda v: v.ffill().bfill()).fillna(0)
    sub_anual["y"] = sub_anual.groupby("Estado")["y"].transform(lambda v: v.ffill().bfill()).fillna(0)
    sub_anual["size_col"] = sub_anual.groupby("Estado")["size_col"].transform(lambda v: v.ffill().bfill()).fillna(10)

    max_size = sub_anual["size_col"].max()
    sub_anual["size_col"] = (sub_anual["size_col"] / max_size * 60 + 10).round(1) if pd.notna(max_size) and max_size > 0 else 10
    sub_anual["Año_str"] = sub_anual["Año"].astype(str)
    sub_anual = sub_anual.sort_values(["Año", "Estado"])

    min_x, max_x = sub_anual["x"].min(), sub_anual["x"].max()
    min_y, max_y = sub_anual["y"].min(), sub_anual["y"].max()
    pad_x = (max_x - min_x) * 0.1 if pd.notna(min_x) and max_x != min_x else 10
    pad_y = (max_y - min_y) * 0.1 if pd.notna(min_y) and max_y != min_y else 10
    rx = [min_x - pad_x, max_x + pad_x] if pd.notna(min_x) else [0, 100]
    ry = [min_y - pad_y, max_y + pad_y] if pd.notna(min_y) else [0, 100]

    fig = px.scatter(
        sub_anual, x="x", y="y",
        color="Estado", size="size_col",
        animation_frame="Año_str",
        color_discrete_map=COLORES_ESTADOS,
        hover_name="Estado",
        range_x=rx, range_y=ry,
        labels={"x": VAR_LABEL[var_x], "y": VAR_LABEL[var_y], "Año_str":"Año"},
        size_max=55,
    )
    fig.update_traces(marker=dict(opacity=0.85, line=dict(width=1, color="white")))
    fig.update_layout(**{**PLOT_LAYOUT,
        "xaxis":dict(title=VAR_LABEL[var_x],gridcolor="#EEE",tickformat=","),
        "yaxis":dict(title=VAR_LABEL[var_y],gridcolor="#EEE"),
        "height":H_CHART,
        "updatemenus":[dict(type="buttons",showactive=False,y=-0.12,x=0.05,
            buttons=[dict(label="▶ Play",method="animate",
                args=[None,{"frame":{"duration":800,"redraw":True},"fromcurrent":True}]),
                dict(label="⏸ Pausa",method="animate",
                args=[[None],{"frame":{"duration":0,"redraw":False},"mode":"immediate"}])]
        )],
    })
    return fig

def fig_scatter_ols(df, var_x, var_y, estados):
    col_x = VAR_COL.get(var_x, "IED")
    col_y = VAR_COL.get(var_y, "Empleo_Manufacturero")
    if col_x not in df.columns or col_y not in df.columns or df.empty: return go.Figure().update_layout(**PLOT_LAYOUT, title="Datos insuficientes")
    
    sub = df[df["Estado"].isin(estados)].dropna(subset=[col_x, col_y]).copy()
    if len(sub) < 2: return go.Figure().update_layout(**PLOT_LAYOUT, title="Datos insuficientes")
    
    fig = px.scatter(sub, x=col_x, y=col_y, color="Estado", color_discrete_map=COLORES_ESTADOS,
        trendline="ols", hover_data=["Periodo"], labels={col_x: VAR_LABEL[var_x], col_y: VAR_LABEL[var_y]})
    fig.update_traces(marker=dict(size=7, opacity=0.7))
    fig.update_layout(**{**PLOT_LAYOUT, "xaxis": dict(title=VAR_LABEL[var_x], gridcolor="#EEE", tickformat=","),
        "yaxis": dict(title=VAR_LABEL[var_y], gridcolor="#EEE"), "height": H_CHART})
    return fig

def fig_prediccion(df_pred, estado):
    sub = df_pred[df_pred["Estado"] == estado]
    if sub.empty: return go.Figure().update_layout(**PLOT_LAYOUT, height=350)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["Periodo"], y=sub["Crec_Empleo_Manufacturero"], name="Crecimiento Real (Δ%)", mode="lines+markers", line=dict(color=AZUL_OSCURO, width=2.5)))
    fig.add_trace(go.Scatter(x=sub["Periodo"], y=sub["Prediccion"], name="Predicción del Modelo", mode="lines", line=dict(color="#E63946", width=2.5, dash="dot")))
    fig.update_layout(**{**PLOT_LAYOUT, "yaxis": dict(title="Tasa de Crecimiento Trimestral (%)", gridcolor="#EEE"), "xaxis": dict(tickangle=-45, tickfont_size=9), "height": 350})
    return fig

def fig_heatmap(df, variable, estados):
    col=VAR_COL.get(variable,"Empleo_Manufacturero")
    if col not in df.columns or df.empty: return go.Figure().update_layout(**PLOT_LAYOUT, title="Datos no disponibles")
    
    sub=df[df["Estado"].isin(estados)]
    piv=sub.pivot_table(index="Estado",columns="Trimestre",values=col,aggfunc="mean").round(1)
    piv.columns=[f"Q{c}" for c in piv.columns]
    fig=go.Figure(go.Heatmap(z=piv.values,x=piv.columns.tolist(),y=piv.index.tolist(),
        colorscale=[[0,"#E6F1FB"],[0.5,"#1A5599"],[1,"#0C3460"]],
        text=np.round(piv.values,0),texttemplate="%{text:,.0f}",textfont_size=11))
    fig.update_layout(**{**PLOT_LAYOUT,"height":H_CHART,"margin":dict(l=140,r=20,t=20,b=20)})
    return fig

def fig_correlacion(corr_df):
    if corr_df is None or corr_df.empty: return go.Figure().update_layout(**PLOT_LAYOUT, height=350)
    labels=[c.replace("Crec_","") for c in corr_df.columns]
    z=corr_df.values
    fig=go.Figure(go.Heatmap(z=z,x=labels,y=labels,
        colorscale=[[0, BG], [0.5, "#A6ACAF"], [1, "#E63946"]],zmin=-1,zmax=1,
        text=np.round(z,2),texttemplate="%{text}",textfont_size=12))
    fig.update_layout(**{**PLOT_LAYOUT,"height":320,"margin":dict(l=80,r=20,t=20,b=80)})
    return fig

# ══════════════════════════════════════════════
# ESTILOS
# ══════════════════════════════════════════════
CARD={"background":CARD_BG,"borderRadius":"12px","border":"1px solid #E5E0D8",
      "padding":"20px 24px","marginBottom":"16px","boxShadow":"0 1px 4px rgba(0,0,0,0.06)"}
METRIC_CARD={"background": CAFE, "borderRadius":"10px","padding":"14px 18px","flex":"1","minWidth":"120px", "border":"1px solid #734c30"}
BTN_BASE={"fontSize":"12px","padding":"5px 14px","borderRadius":"20px","border":"1.5px solid #CCC",
          "background":"white","color":TEXT_PRIM,"cursor":"pointer","marginRight":"6px","marginBottom":"6px","fontFamily":"inherit"}
BTN_VAR_ON={**BTN_BASE,"background":CAFE,"color":"#FFFFFF","border":f"1.5px solid {CAFE}","borderRadius":"6px"}
SEC_HDR={"fontSize":"15px","fontWeight":"600","color":TEXT_PRIM,"margin":"0 0 4px",
         "borderLeft":f"3px solid {NARANJA}","paddingLeft":"10px"}
SUB={"fontSize":"11px","color":TEXT_SEC,"margin":"4px 0 12px 14px"}

TAB_STYLE={"padding":"12px 20px","fontFamily":"'Helvetica Neue',Arial,sans-serif","fontSize":"14px","color":TEXT_SEC,"borderBottom":"2px solid transparent", "background": BG}
TAB_SEL={"padding":"12px 20px","fontFamily":"'Helvetica Neue',Arial,sans-serif","fontSize":"14px","color":TEXT_PRIM,"fontWeight":"bold","borderBottom":f"3px solid {NARANJA}","background": CARD_BG}

# ══════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════
app=dash.Dash(__name__,title="Panel Bajío · Econométrico",
    meta_tags=[{"name":"viewport","content":"width=device-width, initial-scale=1"}])
server=app.server

CONTROLES = html.Div(style={**CARD,"marginBottom":"16px"},children=[
    html.Div(style={"display":"flex","flexWrap":"wrap","gap":"24px","alignItems":"flex-start"},children=[
        html.Div([
            html.P("Estados",style={"fontSize":"11px","color":TEXT_SEC,"textTransform":"uppercase","fontWeight":"bold","margin":"0 0 6px"}),
            html.Div(id="estado-btns",style={"display":"flex","flexWrap":"wrap"}),
        ]),
        html.Div([
            html.P("Período",style={"fontSize":"11px","color":TEXT_SEC,"textTransform":"uppercase","fontWeight":"bold","margin":"0 0 6px"}),
            html.Div(style={"display":"flex","alignItems":"center","gap":"8px"},children=[
                dcc.Dropdown(id="year-from",
                    options=[{"label":str(y),"value":y} for y in AÑOS_DISPONIBLES],
                    value=2018,clearable=False,style={"width":"90px","fontSize":"13px"}),
                html.Span("—",style={"color":TEXT_SEC}),
                dcc.Dropdown(id="year-to",
                    options=[{"label":str(y),"value":y} for y in AÑOS_DISPONIBLES],
                    value=AÑOS_DISPONIBLES[-1],clearable=False,style={"width":"90px","fontSize":"13px"}),
            ]),
        ]),
        html.Div([
            html.P("Variable",style={"fontSize":"11px","color":TEXT_SEC,"textTransform":"uppercase","fontWeight":"bold","margin":"0 0 6px"}),
            html.Div(id="var-btns",style={"display":"flex","gap":"6px","flexWrap":"wrap"}),
        ]),
    ]),
])

app.layout=html.Div(
    style={"fontFamily":"'Helvetica Neue',Arial,sans-serif","background":BG,"minHeight":"100vh","padding":"20px 24px","maxWidth":"1400px","margin":"0 auto"},
    children=[

    html.Div(style={"marginBottom":"20px"},children=[
        html.H1("Panel Econométrico — Región del Bajío",
            style={"fontSize":"28px","fontWeight":"bold","margin":"0 0 4px","color":TEXT_PRIM}),
        html.P("Impacto del Nearshoring: Empleo, IED y Exportaciones Manufactureras (2015–2025)",
            style={"fontSize":"14px","color":TEXT_SEC,"margin":"0"}),
    ]),

    html.Div(style={**CARD, "background": "#EBF5FB", "borderColor": "#D6EAF8", "borderLeft": f"5px solid {AZUL_OSCURO}"}, children=[
        html.P("🔍 Pregunta de Investigación Central", style={"fontSize": "14px", "fontWeight": "bold", "color": AZUL_OSCURO, "margin": "0 0 6px"}),
        html.P("¿Existe una relación positiva entre la Inversión Extranjera Directa (IED), utilizada como proxy del nearshoring, y el empleo manufacturero en San Luis Potosí respecto a los demás estados del Bajío?", 
               style={"fontSize": "14px", "color": TEXT_PRIM, "fontStyle": "italic", "margin": "0 0 10px"}),
        html.Div(children=[
            html.Div([html.Span("H₀ (Nula):", style={"fontWeight": "bold"}), " No existe una relación positiva significativa."]),
            html.Div([html.Span("H₁ (Alternativa):", style={"fontWeight": "bold"}), " Sí existe una relación positiva significativa."])
        ], style={"display": "flex", "gap": "20px", "fontSize": "12px", "color": TEXT_SEC})
    ]),

    html.Div(id="metrics-row",style={"display":"flex","gap":"10px","flexWrap":"wrap","marginBottom":"16px"}),

    dcc.Store(id="active-estados",data=list(ESTADOS_BAJIO.keys())),
    dcc.Store(id="active-var",data="empleo"),

    CONTROLES,

    dcc.Tabs(id="main-tabs",value="tab-visor",
        style={"marginBottom":"16px"},
        children=[

        dcc.Tab(label="📊 Visor de Datos", value="tab-visor",
            style=TAB_STYLE, selected_style=TAB_SEL,
            children=[

            html.Div(style=CARD,children=[
                html.P("Serie de tiempo trimestral",style=SEC_HDR),
                html.Div(style={"display":"flex","alignItems":"center","gap":"16px","margin":"6px 0 12px 14px"},children=[
                    html.P(id="series-sub",style={**SUB,"margin":"0"}),
                    dcc.RadioItems(id="tipo-grafica",
                        options=[{"label":" Líneas","value":"line"},
                                 {"label":" Área","value":"area"},
                                 {"label":" Barras","value":"bar"}],
                        value="line", inline=True,
                        style={"fontSize":"12px","color":TEXT_SEC},
                        inputStyle={"marginRight":"4px"},
                        labelStyle={"marginRight":"14px"}),
                ]),
                dcc.Graph(id="series-chart",config={"displayModeBar":False}),
            ]),

            html.Div(style=CARD,children=[
                html.P("Mapa del Bajío",style=SEC_HDR),
                html.P(id="map-sub",style=SUB),
                dcc.Graph(id="map-chart",config={"displayModeBar":False}),
            ]),

            html.Div(style=CARD,children=[
                html.P("Animación Hans Rosling — Evolución temporal",style=SEC_HDR),
                html.Div(style={"display":"flex","flexWrap":"wrap","gap":"16px","margin":"6px 0 12px 14px","alignItems":"center"},children=[
                    html.Div([
                        html.P("Eje X",style={"fontSize":"10px","color":TEXT_SEC,"margin":"0 0 3px","textTransform":"uppercase"}),
                        dcc.Dropdown(id="scatter-x",
                            options=[{"label":l,"value":v} for v,l in VARS_DEF],
                            value="ied",clearable=False,style={"width":"160px","fontSize":"12px"}),
                    ]),
                    html.Div([
                        html.P("Eje Y",style={"fontSize":"10px","color":TEXT_SEC,"margin":"0 0 3px","textTransform":"uppercase"}),
                        dcc.Dropdown(id="scatter-y",
                            options=[{"label":l,"value":v} for v,l in VARS_DEF],
                            value="empleo",clearable=False,style={"width":"160px","fontSize":"12px"}),
                    ]),
                    html.P("El tamaño del punto = Empleo manufacturero · Presiona ▶ Play",
                        style={"fontSize":"11px","color":TEXT_SEC,"margin":"0"}),
                ]),
                dcc.Graph(id="scatter-chart",config={"displayModeBar":False}),
            ]),

            html.Div(style=CARD,children=[
                html.P("Relación Histórica y Tendencia Lineal",style=SEC_HDR),
                html.P("Muestra la correlación general para todo el periodo. Pasa el cursor sobre la línea para ver el R².",style=SUB),
                dcc.Graph(id="scatter-ols",config={"displayModeBar":False}),
            ]),

            html.Div(style=CARD,children=[
                html.P("Patrón estacional por trimestre",style=SEC_HDR),
                html.P("Promedio del período seleccionado",style=SUB),
                dcc.Graph(id="heatmap-chart",config={"displayModeBar":False}),
            ]),
        ]),

        dcc.Tab(label="🎚 Laboratorio Econométrico", value="tab-eco",
            style=TAB_STYLE, selected_style=TAB_SEL,
            children=[

            html.Div(style={"margin":"20px 0 12px"},children=[
                html.H2("Modelo de Datos Panel",style={"fontSize":"17px","fontWeight":"bold","color":TEXT_PRIM,"margin":"0 0 4px"}),
                html.P("Efectos Fijos de entidad y tiempo · Tasas de crecimiento · Errores clusterizados por entidad",
                    style={"fontSize":"12px","color":TEXT_SEC,"margin":"0"}),
            ]),

            html.Div(style={**CARD, "background": "#FDFCF8", "borderColor": CAFE}, children=[
                html.P("Ecuación del Modelo Estructural:", style={"fontSize": "12px", "color": TEXT_SEC, "margin": "0 0 8px"}),
                html.P("Δ%Empleo_it = β₀ + β₁·Δ%X₁_it + β₂·Δ%X₂_it + μᵢ + λₜ + εᵢₜ", style={"fontSize": "16px", "fontWeight": "bold", "color": CAFE, "fontFamily": "monospace", "margin": "0 0 16px"}),
                
                html.Div(style={"display": "flex", "gap": "40px", "alignItems": "flex-start"}, children=[
                    html.Div([
                        html.P("1. Selecciona las variables independientes (X):", style={"fontSize": "12px", "color": TEXT_SEC, "fontWeight": "bold", "margin": "0 0 8px"}),
                        dcc.Checklist(id="vars-modelo",
                            options=[{"label": f"  {l}", "value": v} for v, l in VARS_DEF if v != "empleo"],
                            value=["ied", "exportaciones"], inline=True, inputStyle={"marginRight": "6px"}, labelStyle={"marginRight": "24px", "fontSize": "13px"},
                        ),
                    ]),
                    html.Div([
                        html.P("2. Rezagos temporales (Trimestres retrasados):", style={"fontSize": "12px", "color": TEXT_SEC, "fontWeight": "bold", "margin": "0 0 8px"}),
                        html.Div(style={"width": "250px"}, children=[
                            dcc.Slider(id="lags-slider", min=0, max=4, step=1, value=0, marks={i: f"{i} trim" for i in range(5)})
                        ])
                    ])
                ])
            ]),

            html.Div(style={"display": "grid", "gridTemplateColumns": "1.5fr 1fr", "gap": "20px", "marginBottom": "20px"}, children=[
                html.Div(style=CARD, children=[
                    html.P("Resultados de la Regresión", style=SEC_HDR),
                    html.P("*** p<0.01  ** p<0.05  * p<0.1", style=SUB),
                    html.Div(id="tabla-regresion"),
                ]),
                html.Div(style=CARD, children=[
                    html.P("Traductor de Insights", style=SEC_HDR),
                    html.P("Lectura automática de impacto al empleo:", style=SUB),
                    html.Div(id="insights-panel"),
                    html.Hr(style={"margin": "15px 0", "borderColor": "#E5E0D8"}),
                    html.P("Pruebas de Robustez:", style={"fontSize": "13px", "fontWeight": "bold", "color": TEXT_PRIM, "marginBottom": "10px"}),
                    html.Div(id="robustez-panel"),
                ]),
            ]),
            
            html.Div(style=CARD, children=[
                html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"}, children=[
                    html.Div([
                        html.P("Realidad vs. Predicción del Modelo", style=SEC_HDR),
                        html.P("¿Qué tan bien se ajusta la fórmula a la realidad? Selecciona un estado para evaluar.", style=SUB),
                    ]),
                    dcc.Dropdown(id="estado-prediccion", options=[{"label": e, "value": e} for e in ESTADOS_BAJIO.keys()], value="San Luis Potosí", clearable=False, style={"width": "200px"})
                ]),
                dcc.Graph(id="prediccion-chart", config={"displayModeBar": False})
            ]),

            html.Div(style=CARD,children=[
                html.P("Matriz de correlación de Pearson",style=SEC_HDR),
                html.P("Sobre tasas de crecimiento trimestral de las variables seleccionadas",style=SUB),
                dcc.Graph(id="corr-chart",config={"displayModeBar":False}),
            ]),
        ]),

        dcc.Tab(label="🗃 Base de Datos", value="tab-datos",
            style=TAB_STYLE, selected_style=TAB_SEL,
            children=[
            html.Div(style={"marginTop":"16px"},children=[
                
                html.Div(style={**CARD, "background": "#FDFCF8", "borderColor": CAFE}, children=[
                    html.P("📖 Diccionario de Datos (Metadata)", style={"fontSize": "15px", "fontWeight": "bold", "color": CAFE, "margin": "0 0 10px"}),
                    html.Table([
                        html.Thead(html.Tr([html.Th("Variable"), html.Th("Unidad de Medida"), html.Th("Fuente Oficial")])),
                        html.Tbody([
                            html.Tr([html.Td("Empleo Manufacturero"), html.Td("Personas ocupadas (Total)"), html.Td("INEGI (EMIM - BIE)")]),
                            html.Tr([html.Td("Inversión Extranjera Directa (IED)"), html.Td("Millones de Dólares (USD)"), html.Td("Secretaría de Economía")]),
                            html.Tr([html.Td("Exportaciones Manufactureras"), html.Td("Millones de Dólares (USD)"), html.Td("INEGI (BISE)")])
                        ])
                    ], style={"width": "100%", "textAlign": "left", "fontSize": "12px", "color": TEXT_PRIM})
                ]),

                html.Div(style=CARD,children=[
                    html.Div(style={"display":"flex","justifyContent":"space-between","alignItems":"center","marginBottom":"12px"},children=[
                        html.Div([
                            html.P("Tabla del panel",style=SEC_HDR),
                            html.P(id="tabla-sub",style=SUB),
                        ]),
                        html.Button("⬇ Descargar CSV",id="btn-csv",
                            style={**BTN_BASE,"background":AZUL_OSCURO,"color":"white","borderColor":AZUL_OSCURO,
                                   "borderRadius":"8px","fontSize":"13px","padding":"8px 16px"}),
                    ]),
                    dcc.Download(id="download-csv"),
                    html.Div(id="tabla-avanzada-container"),
                ]),
            ]),
        ]),
    ]),

    html.P("Fuentes: INEGI (EMIM · BIE-BISE) · Secretaría de Economía · PanelOLS (linearmodels) · Errores clusterizados",
        style={"fontSize":"11px","color":TEXT_SEC,"textAlign":"center","marginTop":"6px"}),
])

# ══════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════
@app.callback(
    Output("estado-btns","children"), Output("active-estados","data"),
    Input({"type":"btn-estado","index":dash.ALL},"n_clicks"),
    State("active-estados","data"), prevent_initial_call=False,
)
def toggle_estado(_,active):
    ctx=callback_context
    if not ctx.triggered or ctx.triggered[0]["prop_id"]==".":
        active=list(ESTADOS_BAJIO.keys())
    else:
        tid=ctx.triggered[0]["prop_id"]
        if "btn-estado" in tid:
            idx=json.loads(tid.split(".")[0])["index"]
            active=[e for e in active if e!=idx] if idx in active and len(active)>1 else (active+[idx] if idx not in active else active)
    btns=[]
    for est in ESTADOS_BAJIO:
        ia=est in active; col=COLORES_ESTADOS[est]
        btns.append(html.Button(est,id={"type":"btn-estado","index":est},n_clicks=0,
            style={**BTN_BASE,"background":col if ia else "white","color": "#FFFFFF" if ia else TEXT_PRIM,"borderColor":col if ia else "#CCC", "fontWeight": "bold" if ia else "normal"}))
    return btns,active

@app.callback(
    Output("var-btns","children"), Output("active-var","data"),
    Input({"type":"btn-var","index":dash.ALL},"n_clicks"),
    State("active-var","data"), prevent_initial_call=False,
)
def toggle_var(_,av):
    ctx=callback_context
    if ctx.triggered and ctx.triggered[0]["prop_id"]!=".":
        tid=ctx.triggered[0]["prop_id"]
        if "btn-var" in tid: av=json.loads(tid.split(".")[0])["index"]
    btns=[]
    for val,label in VARS_DEF:
        ia=val==av
        btns.append(html.Button(label,id={"type":"btn-var","index":val},n_clicks=0,
            style=BTN_VAR_ON if ia else {**BTN_BASE,"borderRadius":"6px"}))
    return btns,av

@app.callback(
    Output("metrics-row","children"),
    Output("series-chart","figure"), Output("map-chart","figure"),
    Output("scatter-chart","figure"), Output("scatter-ols","figure"), Output("heatmap-chart","figure"),
    Output("series-sub","children"), Output("map-sub","children"),
    Input("active-estados","data"), Input("active-var","data"),
    Input("year-from","value"), Input("year-to","value"),
    Input("tipo-grafica","value"),
    Input("scatter-x","value"), Input("scatter-y","value"),
)
def update_visor(estados,variable,yr_from,yr_to,tipo,sx,sy):
    panel,geojson=get_datos()
    if panel.empty:
        df = pd.DataFrame()
        n = 0
    else:
        df=panel[(panel["Estado"].isin(estados))&(panel["Año"]>=yr_from)&(panel["Año"]<=yr_to)].copy()
        n=len(df)

    def m(l,v,s): return html.Div(style=METRIC_CARD,children=[
        html.P(l,style={"fontSize":"10px","color": "#FFFFFF","opacity": 0.8,"textTransform":"uppercase","fontWeight":"bold","margin":"0 0 3px"}),
        html.P(v,style={"fontSize":"22px","fontWeight":"bold","color": "#FFFFFF","margin":"0","lineHeight":"1.1"}),
        html.P(s,style={"fontSize":"11px","color": "#F4EFEA","opacity": 0.9,"margin":"2px 0 0"}),
    ])
    
    col_emp = "Empleo_Manufacturero"
    col_ied = "IED"
    col_exp = "Exportaciones"

    metrics=[
        m("Observaciones",f"{n:,}","trim. × estados"),
        m("Empleo prom.",f"{int(df[col_emp].mean()):,}" if not df.empty and col_emp in df.columns and not df[col_emp].isna().all() else "—","personas/trim."),
        m("IED prom.",f"${df[col_ied].mean():.1f}M" if not df.empty and col_ied in df.columns and not df[col_ied].isna().all() else "—","USD/trim."),
        m("Exportaciones",f"${df[col_exp].mean():.1f}M" if not df.empty and col_exp in df.columns and not df[col_exp].isna().all() else "—","USD/trim."),
    ]

    f_ser  = fig_series(df,variable,estados,tipo)
    f_map  = fig_mapa(df,variable,estados,yr_to,geojson)
    f_scat = fig_scatter_animado(df,sx,sy,estados)
    f_ols  = fig_scatter_ols(df,sx,sy,estados)
    f_heat = fig_heatmap(df,variable,estados)

    return (metrics, f_ser, f_map, f_scat, f_ols, f_heat,
            VAR_LABEL[variable], f"Intensidad promedio {yr_to} · {VAR_LABEL[variable]}")

@app.callback(
    Output("tabla-regresion", "children"), Output("insights-panel", "children"), 
    Output("robustez-panel", "children"), Output("corr-chart", "figure"), Output("prediccion-chart", "figure"),
    Input("active-estados", "data"), Input("active-var", "data"), Input("year-from", "value"), Input("year-to", "value"), 
    Input("vars-modelo", "value"), Input("lags-slider", "value"), Input("estado-prediccion", "value")
)
def update_eco(estados, _, yr_from, yr_to, vars_x, lags, estado_pred):
    panel, _geo = get_datos()
    no_data = html.P("Selecciona variables y amplía el período.", style={"fontSize": "13px", "color": TEXT_SEC, "padding": "12px"})
    
    if panel.empty:
        return no_data, no_data, no_data, go.Figure(), go.Figure()
        
    df = panel[(panel["Estado"].isin(estados)) & (panel["Año"] >= yr_from) & (panel["Año"] <= yr_to)].copy()
    vars_x = vars_x or []
    eco = calcular_econometria(df, [VAR_COL.get(v, v) for v in vars_x], lags)
    
    if eco is None: 
        return no_data, no_data, no_data, go.Figure(), go.Figure()

    nombres = {"Empleo_Manufacturero": "Empleo Mfr.", "IED": "Inversión Extranjera Directa (IED)", "Exportaciones": "Exportaciones Manufactureras"}
    
    hdr = html.Tr([html.Th(c, style={"fontWeight": "bold", "fontSize": "12px", "color": TEXT_SEC, "padding": "8px 12px", "borderBottom": "2px solid #E5E0D8", "textAlign": al}) for c, al in [("Variable (Crec. Δ%)", "left"), ("Coeficiente (Elasticidad)", "right"), ("P-value", "right")]])
    filas = []
    insights = []
    
    for vk, coef in eco["coefs"].items():
        vn = vk.replace("Crec_", "")
        pval = eco["pvals"].get(vk, float("nan"))
        stars = "***" if pd.notna(pval) and pval < 0.01 else ("**" if pd.notna(pval) and pval < 0.05 else ("*" if pd.notna(pval) and pval < 0.1 else ""))
        pcol = "#27AE60" if pd.notna(pval) and pval < 0.05 else TEXT_SEC
        nom_humano = nombres.get(vn, vn)
        
        filas.append(html.Tr([
            html.Td(f"{nom_humano} (Rezago: {lags})", style={"padding": "8px 12px", "fontSize": "13px", "fontWeight": "bold"}),
            html.Td(f"{coef:.4f} {stars}", style={"padding": "8px 12px", "fontSize": "13px", "textAlign": "right"}),
            html.Td(f"{pval:.4f}", style={"padding": "8px 12px", "fontSize": "13px", "textAlign": "right", "color": pcol, "fontWeight": "bold"}),
        ]))
        
        if pd.notna(pval):
            if pval < 0.05:
                accion = "crece" if coef > 0 else "disminuye"
                insights.append(html.Div([
                    html.P(f"✅ {nom_humano} (Significativo)", style={"fontSize": "12px", "fontWeight": "bold", "color": "#27AE60", "margin": "0 0 2px"}),
                    html.P(f"💡 Impacto: Por cada 1% de aumento en {nom_humano}, el Empleo Manufacturero {accion} un {abs(coef):.3f}%.", style={"fontSize": "12px", "color": TEXT_PRIM, "margin": "0 0 8px"})
                ]))
            else:
                insights.append(html.Div([
                    html.P(f"⚠️ {nom_humano} (No Significativo)", style={"fontSize": "12px", "fontWeight": "bold", "color": TEXT_SEC, "margin": "0 0 2px"}),
                    html.P(f"Estadísticamente, no hay evidencia suficiente de que afecte al empleo (P > 0.05).", style={"fontSize": "12px", "color": TEXT_SEC, "margin": "0 0 8px"})
                ]))
                
    summary = [
        html.Tr([html.Td("R² within", style={"padding": "8px 12px", "fontSize": "12px", "color": TEXT_SEC}), html.Td("", ), html.Td(f"{eco['r2_within']:.4f}", style={"padding": "8px 12px", "fontSize": "13px", "textAlign": "right", "fontWeight": "bold"})]),
        html.Tr([html.Td("N obs.", style={"padding": "8px 12px", "fontSize": "12px", "color": TEXT_SEC}), html.Td("", ), html.Td(str(eco["n_obs"]), style={"padding": "8px 12px", "fontSize": "13px", "textAlign": "right"})]),
    ]
    tabla_reg = html.Table([html.Thead(hdr), html.Tbody(filas + summary)], style={"width": "100%", "borderCollapse": "collapse"})

    if eco["r2_within"] < 0.1:
        insights.append(html.P(f"ℹ️ El R² es muy bajo ({eco['r2_within']:.3f}). El modelo actual explica muy poca variación del empleo.", style={"fontSize": "12px", "color": NARANJA, "margin": "0 0 6px"}))
    panel_insights = html.Div(insights, style={"padding": "10px", "background": "#FDFCF8", "borderRadius": "8px", "border": "1px solid #E5E0D8"})

    dw = eco["dw"]
    dw_ok = 1.5 < dw < 2.5
    dw_txt = "Sin autocorrelación temporal ✓" if dw_ok else ("Autocorrelación positiva" if dw < 1.5 else "Autocorrelación negativa")
    dw_col = "#27AE60" if dw_ok else NARANJA
    
    vif_items = [html.Div(style={"display": "flex", "justifyContent": "space-between", "padding": "4px 0", "borderBottom": "1px dashed #E5E0D8"}, children=[
        html.Span(nombres.get(vn, vn), style={"fontSize": "12px", "color": TEXT_SEC}),
        html.Span(f"VIF = {vv:.2f}", style={"fontSize": "12px", "fontWeight": "bold", "color": "#27AE60" if vv <= 5 else "#C0392B" if vv > 10 else NARANJA}),
    ]) for vn, vv in eco["vifs"].items()]
    
    robustez = html.Div([
        html.Div(style={"display": "flex", "alignItems": "center", "gap": "10px", "marginBottom": "10px"}, children=[
            html.Span(f"{dw:.2f}", style={"fontSize": "20px", "fontWeight": "bold", "color": dw_col}),
            html.Span(f"Durbin-Watson ({dw_txt})", style={"fontSize": "12px", "color": TEXT_PRIM}),
        ]),
        html.Div(vif_items),
    ])

    f_pred = fig_prediccion(eco["df_pred"], estado_pred)
    f_corr = fig_correlacion(eco["corr"])

    return tabla_reg, panel_insights, robustez, f_corr, f_pred

@app.callback(
    Output("tabla-avanzada-container", "children"), Output("tabla-sub", "children"),
    Input("active-estados", "data"), Input("year-from", "value"), Input("year-to", "value"),
)
def update_tabla(estados, yr_from, yr_to):
    panel, _ = get_datos()
    if panel.empty:
        return html.P("No hay datos reales disponibles.", style={"color": TEXT_SEC, "padding": "20px"}), "Total: 0 observaciones"
        
    df = panel[(panel["Estado"].isin(estados)) & (panel["Año"] >= yr_from) & (panel["Año"] <= yr_to)].copy()
    df = df.sort_values(["Año", "Trimestre", "Estado"], ascending=[False, False, True])
    
    col_emp = "Empleo_Manufacturero" if "Empleo_Manufacturero" in df.columns else None
    col_ied = "IED" if "IED" in df.columns else None
    col_exp = "Exportaciones" if "Exportaciones" in df.columns else None
    
    if col_emp: df["Empleo"] = df[col_emp].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "N/A")
    else: df["Empleo"] = "N/A"
    
    if col_ied: df["IED_str"] = df[col_ied].apply(lambda x: f"${x:.1f} M" if pd.notna(x) else "N/A")
    else: df["IED_str"] = "N/A"
    
    if col_exp: df["Exp"] = df[col_exp].apply(lambda x: f"${x:.1f} M" if pd.notna(x) else "N/A")
    else: df["Exp"] = "N/A"
    
    if "Var_Empleo_pct" in df.columns:
        df["Crec_Empleo"] = df["Var_Empleo_pct"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
    else:
        df["Crec_Empleo"] = "N/A"
    
    df_show = df[["Estado", "Año", "Trimestre", "Empleo", "IED_str", "Exp", "Crec_Empleo"]]
    df_show.columns = ["Estado", "Año", "Trim.", "Empleo", "IED", "Exportaciones", "Crec. Empleo"]

    tabla_avanzada = dash_table.DataTable(
        data=df_show.to_dict('records'),
        columns=[{"name": i, "id": i} for i in df_show.columns],
        page_size=15, 
        filter_action="native",
        sort_action="native",
        style_header={'backgroundColor': '#F4EFEA', 'fontWeight': 'bold', 'color': '#2C3E50', 'fontSize': '12px', 'textAlign': 'left', 'padding': '10px'},
        style_cell={'fontFamily': 'Arial, sans-serif', 'fontSize': '12px', 'textAlign': 'left', 'padding': '10px', 'color': '#6B6B6B'},
        style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#FAFAFA'}]
    )
    
    fecha_actualizacion = date.today().strftime("%d de %B, %Y")
    return tabla_avanzada, f"Total: {len(df)} observaciones · Última extracción: {fecha_actualizacion}"

@app.callback(
    Output("download-csv", "data"), Input("btn-csv", "n_clicks"),
    State("active-estados", "data"), State("year-from", "value"), State("year-to", "value"), prevent_initial_call=True,
)
def descargar_csv(n, estados, yr_from, yr_to):
    panel, _ = get_datos()
    if panel.empty: return None
    df = panel[(panel["Estado"].isin(estados)) & (panel["Año"] >= yr_from) & (panel["Año"] <= yr_to)]
    return dcc.send_data_frame(df.to_csv, "panel_econometrico_bajio.csv", index=False)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)
