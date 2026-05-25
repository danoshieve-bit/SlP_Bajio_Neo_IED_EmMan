"""
Dashboard Econométrico — Región del Bajío (VERSIÓN DEFINITIVA CORREGIDA)
========================================================
UI: Fondo Crema, SLP Héroe, Mapa Animado, Scatter OLS
ETL: Lógica de Pandas restaurada y segura. Conexión BISE dinámica.
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
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import plotly.express as px

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from linearmodels.panel import PanelOLS

# ══════════════════════════════════════════════
# CONFIGURACIÓN Y PALETA DE COLORES
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

BG          = "#F4EFEA"
CARD_BG     = "#FFFFFF"
TEXT_PRIM   = "#2C3E50"
TEXT_SEC    = "#6B6B6B"
AZUL_OSCURO = "#1B4F72"
CAFE        = "#8B5E3C"
NARANJA     = "#D35400"

COLORES_ESTADOS = {
    "San Luis Potosí": "#E63946",
    "Aguascalientes":  "#1B4F72",
    "Guanajuato":      "#2874A6",
    "Jalisco":         "#5DADE2",
    "Querétaro":       "#A6ACAF",
}

YEARS = list(range(2015, 2026))

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'Helvetica Neue', Arial, sans-serif", color=TEXT_PRIM, size=12),
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(orientation="h", y=-0.15, x=0, font_size=11),
)

# INDICADORES CORREGIDOS (Respetando la API BISE y BIE)
INDICADORES_EMPLEO = {"Aguascalientes":"702846", "Guanajuato":"702855", "Jalisco":"702858", "Querétaro":"702866", "San Luis Potosí":"702868"}

# Indicadores fijos donde solo cambia el código geográfico en la URL
INDICADOR_ACTIND = "738413"
INDICADOR_EXPORTACIONES = "924,739277" 

SE_IED_URL = "https://datos.gob.mx/busca/api/action/datastore_search?resource_id=fc1e3b7b-4027-4c59-9e5a-f02f48e90ca1&limit=5000"
GEOJSON_URLS = [
    "https://raw.githubusercontent.com/PhantomInsights/mexican-geojson/main/src/states/states.json",
    "https://raw.githubusercontent.com/angelnmara/geojson/master/mexicoHigh.json",
]

_EMP_BASE = {"Aguascalientes":95000,"Guanajuato":340000,"Jalisco":280000,"Querétaro":160000,"San Luis Potosí":110000}
_IED_BASE = {"Aguascalientes":220,"Guanajuato":315,"Jalisco":430,"Querétaro":265,"San Luis Potosí":225}
_ACT_BASE = {"Aguascalientes":108,"Guanajuato":115,"Jalisco":112,"Querétaro":120,"San Luis Potosí":106}
_EXP_BASE = {"Aguascalientes":4200,"Guanajuato":9500,"Jalisco":7800,"Querétaro":5600,"San Luis Potosí":3900}

VARS_DEF   = [("empleo","Empleo Manufacturero"),("ied","IED"),("actind","Actividad Manufacturera"),("exportaciones","Exportaciones")]
VAR_COL    = {"empleo":"Empleo_Manufacturero","ied":"IED","actind":"ActInd","exportaciones":"Exportaciones"}
VAR_LABEL  = {
    "empleo":        "Empleo Manufacturero (personas)",
    "ied":           "Inversión Extranjera Directa (IED) (M USD)",
    "actind":        "Actividad Manufacturera (base 2013=100)",
    "exportaciones": "Exportaciones Manufactureras (M USD)",
}

# ══════════════════════════════════════════════
# ETL — PIPELINE DE DATOS SEGURO Y RESTAURADO
# ══════════════════════════════════════════════
def fetch_inegi_serie(indicador, fuente="BIE", geo="00"):
    url = f"https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml/INDICATOR/{indicador}/es/{geo}/false/{fuente}/2.0/{TOKEN_INEGI}?type=json"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        
        series_list = data.get("Series", [])
        if not series_list: 
            return pd.DataFrame(columns=["fecha","valor"])
            
        # Tomamos la última serie (la más específica si hay múltiples, como en exportaciones)
        obs = series_list[-1].get("OBSERVATIONS", [])
        rows = [{"fecha": s["TIME_PERIOD"], "valor": float(s["OBS_VALUE"])} for s in obs if s["OBS_VALUE"] not in (None, "", "N/A")]
        
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"Error INEGI ({indicador} - {geo}): {e}")
        return pd.DataFrame(columns=["fecha","valor"])

def _mensual_a_trim(df, estado, col):
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"], format="%Y/%m", errors="coerce")
    df = df.dropna(subset=["fecha"])
    df["Año"] = df["fecha"].dt.year
    df["Mes"] = df["fecha"].dt.month
    df = df[(df["Año"] >= 2015) & (df["Año"] <= 2025)].copy()
    df["Trimestre"] = df["Mes"].apply(lambda m: (m - 1) // 3 + 1)
    df["Estado"] = estado
    
    return df.groupby(["Estado", "Año", "Trimestre"])["valor"].mean().reset_index().rename(columns={"valor": col})

def _parse_trim(df, estado, col):
    rows = []
    for _, row in df.iterrows():
        t = str(row["fecha"]).replace("-", "/")
        try:
            p = t.split("/")
            yr = int(p[0])
            qn = int(p[1].replace("Q", "").replace("T", ""))
            if 2015 <= yr <= 2025:
                rows.append({"Estado": estado, "Año": yr, "Trimestre": qn, col: row["valor"]})
        except Exception:
            pass
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Estado", "Año", "Trimestre", col])

# ---- Simulaciones Seguras ----
def _sim_empleo(estado):
    rng = np.random.default_rng(abs(hash(estado)) % (2**32))
    base = _EMP_BASE.get(estado, 100000)
    rows = []
    for yr in YEARS:
        for m in range(1, 13):
            if yr == 2025 and m > 3: break
            t = (yr - 2015) * 12 + m
            val = int(base * (1 + 0.018 * t / 12) * rng.uniform(0.97, 1.03) * [.97,.98,1.,1.01,1.02,1.02,1.01,1.01,1.,.99,.98,.97][m-1])
            rows.append({"fecha": pd.Timestamp(yr, m, 1), "valor": val})
    return pd.DataFrame(rows)

def _sim_ied():
    rows = []
    for estado, base in _IED_BASE.items():
        rng = np.random.default_rng(abs(hash(estado+"ied")) % (2**32))
        for yr in YEARS:
            for q in range(1, 5):
                if yr == 2025 and q > 1: break
                t = (yr - 2015) * 4 + q
                val = round(base * (1 + 0.02 * t) * rng.uniform(0.88, 1.20) * [1.,1.1,1.05,1.15][q-1], 1)
                rows.append({"Estado": estado, "Año": yr, "Trimestre": q, "IED": val})
    return pd.DataFrame(rows)

def _sim_actind():
    rows = []
    for estado, base in _ACT_BASE.items():
        rng = np.random.default_rng(abs(hash(estado+"act")) % (2**32))
        for yr in YEARS:
            for q in range(1, 5):
                if yr == 2025 and q > 1: break
                t = (yr - 2015) * 4 + q
                val = round(base * (1 + 0.015 * t) * rng.uniform(0.94, 1.06) * [.98,1.01,1.02,.99][q-1], 1)
                rows.append({"Estado": estado, "Año": yr, "Trimestre": q, "ActInd": val})
    return pd.DataFrame(rows)

def _sim_exportaciones():
    rows = []
    for estado, base in _EXP_BASE.items():
        rng = np.random.default_rng(abs(hash(estado+"exp")) % (2**32))
        for yr in YEARS:
            for q in range(1, 5):
                if yr == 2025 and q > 1: break
                t = (yr - 2015) * 4 + q
                val = round(base * (1 + 0.022 * t) * rng.uniform(0.85, 1.18) * [.95,1.05,1.08,1.12][q-1], 1)
                rows.append({"Estado": estado, "Año": yr, "Trimestre": q, "Exportaciones": val})
    return pd.DataFrame(rows)

# ---- Procesamiento Final ----
def procesar_empleo():
    print("📥 Descargando Empleo...")
    frames = []
    for estado, ind in INDICADORES_EMPLEO.items():
        df = fetch_inegi_serie(ind, fuente="BIE-BISE", geo="00")
        if df.empty:
            sim = _sim_empleo(estado)
            sim["Estado"] = estado
            sim["Año"] = sim["fecha"].dt.year
            sim["Mes"] = sim["fecha"].dt.month
            sim = sim[(sim["Año"] >= 2015) & (sim["Año"] <= 2025)].copy()
            sim["Trimestre"] = sim["Mes"].apply(lambda m: (m - 1) // 3 + 1)
            df_t = sim.groupby(["Estado", "Año", "Trimestre"])["valor"].mean().reset_index().rename(columns={"valor": "Empleo_Manufacturero"})
        else:
            df_t = _mensual_a_trim(df, estado, "Empleo_Manufacturero")
        
        df_t["Empleo_Manufacturero"] = df_t["Empleo_Manufacturero"].round(0).astype(int)
        frames.append(df_t)
    return pd.concat(frames, ignore_index=True)

def procesar_ied():
    print("📥 Descargando IED...")
    try:
        r = requests.get(SE_IED_URL, timeout=15)
        r.raise_for_status()
        recs = r.json().get("result", {}).get("records", [])
        if not recs: raise ValueError("Sin registros de IED")
        df = pd.DataFrame(recs)
        
        ce = next((c for c in df.columns if "entidad" in c.lower() or "estado" in c.lower()), None)
        ca = next((c for c in df.columns if "año" in c.lower() or "anio" in c.lower()), None)
        ct = next((c for c in df.columns if "trim" in c.lower()), None)
        ci = next((c for c in df.columns if "ied" in c.lower() or "inversion" in c.lower()), None)
        
        df = df.rename(columns={ce: "Estado", ca: "Año", ct: "Trimestre", ci: "IED"})
        df = df[df["Estado"].str.strip().isin(ESTADOS_BAJIO.keys())]
        
        for col in ["Año", "Trimestre", "IED"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
        return df.dropna()[(df["Año"] >= 2015) & (df["Año"] <= 2025)][["Estado", "Año", "Trimestre", "IED"]]
    except Exception as e:
        print(f"⚠️ Error IED: {e}. Usando simulación.")
        return _sim_ied()

def procesar_actind():
    print("📥 Descargando Actividad Industrial...")
    frames = []
    for estado, geo_code in ESTADOS_BAJIO.items():
        df = fetch_inegi_serie(INDICADOR_ACTIND, fuente="BIE-BISE", geo=geo_code)
        if df.empty:
            frames.append(_sim_actind()[lambda d: d["Estado"] == estado])
        else:
            frames.append(_mensual_a_trim(df, estado, "ActInd"))
    return pd.concat(frames, ignore_index=True)

def procesar_exportaciones():
    print("📥 Descargando Exportaciones...")
    frames = []
    for estado, geo_code in ESTADOS_BAJIO.items():
        df = fetch_inegi_serie(INDICADOR_EXPORTACIONES, fuente="BIE-BISE", geo=geo_code)
        if not df.empty:
            df_t = _parse_trim(df, estado, "Exportaciones")
            if not df_t.empty:
                frames.append(df_t)
                continue
                
        rng = np.random.default_rng(abs(hash(estado+"exp")) % (2**32))
        base = _EXP_BASE.get(estado, 5000)
        sim = []
        for yr in YEARS:
            for q in range(1, 5):
                if yr == 2025 and q > 1: break
                t = (yr - 2015) * 4 + q
                val = round(base * (1 + 0.022 * t) * rng.uniform(0.85, 1.18) * [.95,1.05,1.08,1.12][q-1], 1)
                sim.append({"Estado": estado, "Año": yr, "Trimestre": q, "Exportaciones": val})
        frames.append(pd.DataFrame(sim))
        
    return pd.concat(frames, ignore_index=True)

def construir_panel(df_emp, df_ied, df_act, df_exp):
    keys = ["Estado", "Año", "Trimestre"]
    # Merge secuencial seguro
    panel = pd.merge(df_emp, df_ied, on=keys, how="inner")
    panel = pd.merge(panel, df_act, on=keys, how="left")
    panel = pd.merge(panel, df_exp, on=keys, how="left")
    
    panel = panel[panel["Estado"].isin(ESTADOS_BAJIO.keys())].copy()
    panel = panel.sort_values(["Estado", "Año", "Trimestre"]).reset_index(drop=True)
    panel["Var_Empleo_pct"] = (panel.groupby("Estado")["Empleo_Manufacturero"].pct_change() * 100).round(2)
    panel["Periodo"] = panel["Año"].astype(str) + " Q" + panel["Trimestre"].astype(str)
    
    for col in ["ActInd", "Exportaciones"]:
        panel[col] = panel.groupby("Estado")[col].transform(lambda x: x.interpolate(limit_direction="both"))
        
    print(f"✅ Panel construido: {len(panel)} observaciones")
    return panel

def cargar_geojson():
    for url in GEOJSON_URLS:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200: return r.json()
        except: continue
    return {"type": "FeatureCollection", "features": [
        {"type":"Feature","id":e,"properties":{"name":e},"geometry":{"type":"Polygon","coordinates":[[[-100,20],[-101,20],[-101,21],[-100,21]]]}} for e in ESTADOS_BAJIO.keys()
    ]}

# ══════════════════════════════════════════════
# CACHÉ DE DATOS
# ══════════════════════════════════════════════
_cache = {"panel": None, "geojson": None, "fecha": None, "lock": threading.Lock()}

def get_datos():
    hoy = date.today()
    if _cache["panel"] is not None and _cache["fecha"] == hoy: 
        return _cache["panel"], _cache["geojson"]
        
    with _cache["lock"]:
        if _cache["panel"] is not None and _cache["fecha"] == hoy: 
            return _cache["panel"], _cache["geojson"]
            
        print("🔄 Procesando datos frescos desde API/Simulación...")
        df_emp = procesar_empleo()
        df_ied = procesar_ied()
        df_act = procesar_actind()
        df_exp = procesar_exportaciones()
        
        _cache["panel"] = construir_panel(df_emp, df_ied, df_act, df_exp)
        _cache["geojson"] = cargar_geojson()
        _cache["fecha"] = hoy
        
    return _cache["panel"], _cache["geojson"]

# Inicializamos para tener variables globales disponibles
PANEL, GEOJSON = get_datos()
AÑOS_DISPONIBLES = sorted(PANEL["Año"].unique())

# ══════════════════════════════════════════════
# ECONOMETRÍA
# ══════════════════════════════════════════════
def calcular_econometria(df, vars_x):
    if not vars_x: return None
    cols = ["Empleo_Manufacturero"] + vars_x
    sub = df[["Estado", "Año", "Trimestre"] + cols].dropna().copy()
    if len(sub) < 20: return None
    
    for c in cols: 
        sub[f"Crec_{c}"] = sub.groupby("Estado")[c].pct_change() * 100
        
    sub = sub.dropna()
    sub["t"] = (sub["Año"] - sub["Año"].min()) * 4 + sub["Trimestre"]
    sub = sub.set_index(["Estado", "t"])
    
    Y = sub["Crec_Empleo_Manufacturero"]
    X = sub[[f"Crec_{c}" for c in vars_x]]
    
    try:
        res = PanelOLS(Y, X, entity_effects=True, time_effects=True).fit(cov_type="clustered", cluster_entity=True)
        dw_stat = durbin_watson(res.resids.values)
    except: return None
    
    vifs = {}
    if len(vars_x) > 1:
        X_vif = sub[[f"Crec_{c}" for c in vars_x]].reset_index(drop=True)
        try:
            for i, c in enumerate(X_vif.columns): 
                vifs[c.replace("Crec_", "")] = round(float(variance_inflation_factor(X_vif.values.astype(float), i)), 2)
        except: 
            vifs = {c: float("nan") for c in vars_x}
    else: 
        vifs = {vars_x[0]: 1.0}
        
    return {
        "coefs": res.params, "pvals": res.pvalues, "r2_within": round(res.rsquared, 4),
        "n_obs": int(res.nobs), "dw": round(dw_stat, 3), "vifs": vifs,
        "corr": sub[[f"Crec_{c}" for c in cols]].corr().round(3)
    }

# ══════════════════════════════════════════════
# FIGURAS GRÁFICAS
# ══════════════════════════════════════════════
H_CHART = 450

def fig_series(df, variable, estados, tipo="line"):
    col = VAR_COL.get(variable, "Empleo_Manufacturero")
    fig = go.Figure()
    for est in estados:
        sub = df[df["Estado"] == est].sort_values(["Año", "Trimestre"])
        if tipo == "line":
            fig.add_trace(go.Scatter(x=sub["Periodo"], y=sub[col], name=est, mode="lines+markers",
                line=dict(color=COLORES_ESTADOS[est], width=2.5), marker=dict(size=5),
                hovertemplate=f"<b>{est}</b><br>%{{x}}<br>{VAR_LABEL[variable]}: %{{y:,.1f}}<extra></extra>"))
        elif tipo == "area":
            fig.add_trace(go.Scatter(x=sub["Periodo"], y=sub[col], name=est, mode="lines",
                stackgroup='one', line=dict(color=COLORES_ESTADOS[est], width=1),
                hovertemplate=f"<b>{est}</b><br>%{{x}}<br>{VAR_LABEL[variable]}: %{{y:,.1f}}<extra></extra>"))
        else:
            fig.add_trace(go.Bar(x=sub["Periodo"], y=sub[col], name=est, marker_color=COLORES_ESTADOS[est],
                hovertemplate=f"<b>{est}</b><br>%{{x}}<br>{VAR_LABEL[variable]}: %{{y:,.1f}}<extra></extra>"))
                
    fig.update_layout(**{**PLOT_LAYOUT, "yaxis": dict(title=VAR_LABEL[variable], gridcolor="#E5E0D8", tickformat=","),
        "xaxis": dict(tickangle=-45, tickfont_size=9), "height": H_CHART, "barmode": "stack" if tipo == "bar" else "group"})
    return fig

def fig_mapa_animado(df, variable, estados, geojson):
    col = VAR_COL.get(variable, "Empleo_Manufacturero")
    sub = df[df["Estado"].isin(estados)].copy()
    
    grp = sub.groupby(["Estado", "Año"])[col].mean().reset_index().rename(columns={col: "valor"})
    all_years = sorted(grp["Año"].unique())
    
    # Rellenar datos faltantes para que la animación no salte
    idx = pd.MultiIndex.from_product([estados, all_years], names=['Estado', 'Año'])
    grp = grp.set_index(['Estado', 'Año']).reindex(idx).reset_index()
    grp["valor"] = grp["valor"].fillna(0)
    grp["Año_str"] = grp["Año"].astype(str)
    grp = grp.sort_values("Año")

    fig = px.choropleth(
        grp, geojson=geojson, locations="Estado", featureidkey="properties.name",
        color="valor", animation_frame="Año_str",
        color_continuous_scale="Blues",
        labels={"valor": "Promedio Anual"}
    )
    fig.update_geos(fitbounds="locations", visible=False, showland=True, landcolor=BG)
    fig.update_layout(**{**PLOT_LAYOUT, "height": H_CHART, "margin": dict(l=0, r=0, t=20, b=0)})
    return fig

def fig_scatter_animado(df, var_x, var_y, estados):
    col_x = VAR_COL.get(var_x, "IED")
    col_y = VAR_COL.get(var_y, "Empleo_Manufacturero")
    sub = df[df["Estado"].isin(estados)].copy()
    
    sub_anual = sub.groupby(["Estado", "Año"]).agg(x=(col_x, "mean"), y=(col_y, "mean"), size_col=("Empleo_Manufacturero", "mean")).reset_index()
    max_size = sub_anual["size_col"].max()
    if max_size > 0:
        sub_anual["size_col"] = (sub_anual["size_col"] / max_size * 60 + 10).round(1)
    else:
        sub_anual["size_col"] = 10
        
    sub_anual["Año_str"] = sub_anual["Año"].astype(str)

    fig = px.scatter(sub_anual, x="x", y="y", color="Estado", size="size_col", animation_frame="Año_str",
        color_discrete_map=COLORES_ESTADOS, hover_name="Estado", labels={"x": VAR_LABEL[var_x], "y": VAR_LABEL[var_y], "Año_str": "Año"}, size_max=55)
    fig.update_traces(marker=dict(opacity=0.85, line=dict(width=1, color="white")))
    fig.update_layout(**{**PLOT_LAYOUT, "xaxis": dict(title=VAR_LABEL[var_x], gridcolor="#E5E0D8", tickformat=","),
        "yaxis": dict(title=VAR_LABEL[var_y], gridcolor="#E5E0D8"), "height": H_CHART})
    return fig

def fig_scatter_ols(df, var_x, var_y, estados):
    col_x = VAR_COL.get(var_x, "IED")
    col_y = VAR_COL.get(var_y, "Empleo_Manufacturero")
    sub = df[df["Estado"].isin(estados)].copy()
    
    fig = px.scatter(sub, x=col_x, y=col_y, color="Estado", color_discrete_map=COLORES_ESTADOS,
        trendline="ols", hover_data=["Periodo"], labels={col_x: VAR_LABEL[var_x], col_y: VAR_LABEL[var_y]})
    fig.update_traces(marker=dict(size=7, opacity=0.7))
    fig.update_layout(**{**PLOT_LAYOUT, "xaxis": dict(title=VAR_LABEL[var_x], gridcolor="#E5E0D8", tickformat=","),
        "yaxis": dict(title=VAR_LABEL[var_y], gridcolor="#E5E0D8"), "height": H_CHART})
    return fig

def fig_heatmap(df, variable, estados):
    col = VAR_COL.get(variable, "Empleo_Manufacturero")
    sub = df[df["Estado"].isin(estados)]
    piv = sub.pivot_table(index="Estado", columns="Trimestre", values=col, aggfunc="mean").round(1)
    piv.columns = [f"Q{c}" for c in piv.columns]
    
    fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns.tolist(), y=piv.index.tolist(),
        colorscale=[[0, BG], [0.5, "#A6ACAF"], [1, "#1B4F72"]],
        text=np.round(piv.values, 0), texttemplate="%{text:,.0f}", textfont_size=11))
    fig.update_layout(**{**PLOT_LAYOUT, "height": H_CHART, "margin": dict(l=140, r=20, t=20, b=20)})
    return fig

def fig_correlacion(corr_df):
    labels = [c.replace("Crec_", "") for c in corr_df.
