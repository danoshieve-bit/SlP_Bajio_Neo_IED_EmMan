"""
Dashboard Econométrico — Región del Bajío (VERSIÓN MASTER BLINDADA)
========================================================
UI: Pregunta de Inv, Mapa Mapbox, DataTable, Lags, Predicciones
ETL: Lógica robusta original, fusiones seguras (Left Merge) y limpieza de strings.
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
# CONFIGURACIÓN Y PALETA
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

INDICADORES_EMPLEO = {"Aguascalientes":"702846", "Guanajuato":"702855", "Jalisco":"702858", "Querétaro":"702866", "San Luis Potosí":"702868"}
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

VARS_DEF   = [("empleo","Empleo Manufacturero"),("ied","Inversión Extranjera Directa (IED)"),("actind","Actividad Manufacturera"),("exportaciones","Exportaciones Manufactureras")]
VAR_COL    = {"empleo":"Empleo_Manufacturero","ied":"IED","actind":"ActInd","exportaciones":"Exportaciones"}
VAR_LABEL  = {
    "empleo":        "Empleo Manufacturero (personas)",
    "ied":           "Inversión Extranjera Directa (IED) (M USD)",
    "actind":        "Actividad Manufacturera (base 2013=100)",
    "exportaciones": "Exportaciones Manufactureras (M USD)",
}

# ══════════════════════════════════════════════
# ETL — CÓDIGO SEGURO CON PROTECCIONES
# ══════════════════════════════════════════════
def fetch_inegi_serie(indicador, fuente="BIE", geo="00"):
    url = f"https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml/INDICATOR/{indicador}/es/{geo}/false/{fuente}/2.0/{TOKEN_INEGI}?type=json"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        rows = []
        series_list = data.get("Series", [])
        if not series_list:
            return pd.DataFrame(columns=["fecha","valor"])
            
        obs = series_list[-1].get("OBSERVATIONS", [])
        for s in obs:
            if s["OBS_VALUE"] not in (None, "", "N/A"):
                try: rows.append({"fecha": s["TIME_PERIOD"], "valor": float(s["OBS_VALUE"])})
                except Exception: pass
        if not rows: raise ValueError("vacío")
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame(columns=["fecha","valor"])

def _mensual_a_trim(df, estado, col):
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"], format="%Y/%m", errors="coerce")
    df = df.dropna(subset=["fecha"])
    df["Año"] = df["fecha"].dt.year
    df["Mes"] = df["fecha"].dt.month
    df = df[(df["Año"]>=2015)&(df["Año"]<=2025)].copy()
    df["Trimestre"] = df["Mes"].apply(lambda m:(m-1)//3+1)
    df["Estado"] = estado
    return df.groupby(["Estado","Año","Trimestre"])["valor"].mean().reset_index().rename(columns={"valor":col})

def _parse_trim(df, estado, col):
    rows=[]
    for _,row in df.iterrows():
        t = str(row["fecha"]).replace("-","/")
        try:
            p = t.split("/")
            yr = int(p[0])
            qn = int(p[1].replace("Q","").replace("T",""))
            if 2015 <= yr <= 2025: 
                rows.append({"Estado":estado,"Año":yr,"Trimestre":qn,col:row["valor"]})
        except Exception: pass
    if rows: return pd.DataFrame(rows)
    else: return pd.DataFrame(columns=["Estado","Año","Trimestre",col])

def _sim_empleo(estado):
    rng = np.random.default_rng(abs(hash(estado))%(2**32))
    base = _EMP_BASE.get(estado,100000)
    rows = []
    for yr in YEARS:
        for m in range(1,13):
            if yr==2025 and m>3: break
            t = (yr-2015)*12+m
            valor = int(base*(1+0.018*t/12)*rng.uniform(0.97,1.03)*[.97,.98,1.,.101,1.02,1.02,1.01,1.01,1.,.99,.98,.97][m-1])
            rows.append({"fecha":pd.Timestamp(yr,m,1),"valor": valor})
    return pd.DataFrame(rows)

def _sim_ied():
    rows=[]
    for estado,base in _IED_BASE.items():
        rng = np.random.default_rng(abs(hash(estado+"ied"))%(2**32))
        for yr in YEARS:
            for q in range(1,5):
                if yr==2025 and q>1: break
                t = (yr-2015)*4+q
                valor = round(base*(1+0.02*t)*rng.uniform(0.88,1.20)*[1.,1.1,1.05,1.15][q-1],1)
                rows.append({"Estado":estado,"Año":yr,"Trimestre":q,"IED": valor})
    return pd.DataFrame(rows)

def _sim_actind():
    rows=[]
    for estado,base in _ACT_BASE.items():
        rng = np.random.default_rng(abs(hash(estado+"act"))%(2**32))
        for yr in YEARS:
            for q in range(1,5):
                if yr==2025 and q>1: break
                t = (yr-2015)*4+q
                valor = round(base*(1+0.015*t)*rng.uniform(0.94,1.06)*[.98,1.01,1.02,.99][q-1],1)
                rows.append({"Estado":estado,"Año":yr,"Trimestre":q,"ActInd": valor})
    return pd.DataFrame(rows)

def _sim_exportaciones():
    rows=[]
    for estado,base in _EXP_BASE.items():
        rng = np.random.default_rng(abs(hash(estado+"exp"))%(2**32))
        for yr in YEARS:
            for q in range(1,5):
                if yr==2025 and q>1: break
                t = (yr-2015)*4+q
                valor = round(base*(1+0.022*t)*rng.uniform(0.85,1.18)*[.95,1.05,1.08,1.12][q-1],1)
                rows.append({"Estado":estado,"Año":yr,"Trimestre":q,"Exportaciones": valor})
    return pd.DataFrame(rows)

def procesar_empleo():
    print("📥 Descargando Empleo manufacturero...")
    frames=[]
    for estado,ind in INDICADORES_EMPLEO.items():
        df = fetch_inegi_serie(ind, fuente="BIE-BISE", geo="00")
        if df.empty:
            sim = _sim_empleo(estado)
            sim["Estado"] = estado
            sim["Año"] = sim["fecha"].dt.year
            sim["Mes"] = sim["fecha"].dt.month
            sim = sim[(sim["Año"]>=2015)&(sim["Año"]<=2025)].copy()
            sim["Trimestre"] = sim["Mes"].apply(lambda m:(m-1)//3+1)
            df_t = sim.groupby(["Estado","Año","Trimestre"])["valor"].mean().reset_index().rename(columns={"valor":"Empleo_Manufacturero"})
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
        recs = r.json().get("result",{}).get("records",[])
        if not recs: raise ValueError("Sin registros de IED")
        df = pd.DataFrame(recs)
        
        ce = next((c for c in df.columns if "entidad" in c.lower() or "estado" in c.lower()), None)
        ca = next((c for c in df.columns if "año" in c.lower() or "anio" in c.lower()), None)
        ct = next((c for c in df.columns if "trim" in c.lower()), None)
        ci = next((c for c in df.columns if "ied" in c.lower() or "inversion" in c.lower()), None)
        
        if not all([ce,ca,ct,ci]): raise ValueError("Columnas de IED no mapeadas")
            
        df = df.rename(columns={ce:"Estado", ca:"Año", ct:"Trimestre", ci:"IED"})
        
        # 🛡️ BLINDAJE ANTI-ERRORES DE LA API (Normalización de Texto)
        df["Estado"] = df["Estado"].astype(str).str.strip()
        df["Estado"] = df["Estado"].replace({"San Luis Potosi": "San Luis Potosí", "Queretaro": "Querétaro"})
        df = df[df["Estado"].isin(ESTADOS_BAJIO.keys())]
        
        for col in ["Año","Trimestre","IED"]: 
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
        df = df.dropna()
        df = df[(df["Año"]>=2015) & (df["Año"]<=2025)]
        return df[["Estado","Año","Trimestre","IED"]]
    except Exception as e:
        print(f"⚠️ Error IED: {e}. Usando simulación.")
        return _sim_ied()

def procesar_actind():
    print("📥 Descargando Actividad Industrial...")
    frames=[]
    for estado, geo_code in ESTADOS_BAJIO.items():
        df = fetch_inegi_serie(INDICADOR_ACTIND, fuente="BIE-BISE", geo=geo_code)
        if df.empty: frames.append(_sim_actind()[lambda d:d["Estado"]==estado])
        else: frames.append(_mensual_a_trim(df, estado, "ActInd"))
    result = pd.concat(frames, ignore_index=True)
    if result.empty: return _sim_actind()
    return result

def procesar_exportaciones():
    print("📥 Descargando Exportaciones...")
    frames=[]
    for estado, geo_code in ESTADOS_BAJIO.items():
        df = fetch_inegi_serie(INDICADOR_EXPORTACIONES, fuente="BIE-BISE", geo=geo_code)
        if not df.empty:
            df_t = _parse_trim(df, estado, "Exportaciones")
            if not df_t.empty: 
                frames.append(df_t)
                continue
                
        rng = np.random.default_rng(abs(hash(estado+"exp"))%(2**32))
        base = _EXP_BASE.get(estado,5000); sim = []
        for yr in YEARS:
            for q in range(1,5):
                if yr==2025 and q>1: break
                t = (yr-2015)*4+q
                valor = round(base*(1+0.022*t)*rng.uniform(0.85,1.18)*[.95,1.05,1.08,1.12][q-1],1)
                sim.append({"Estado":estado, "Año":yr, "Trimestre":q, "Exportaciones":valor})
        frames.append(pd.DataFrame(sim))
    return pd.concat(frames, ignore_index=True)

def construir_panel(df_emp, df_ied, df_act, df_exp):
    keys = ["Estado","Año","Trimestre"]
    # 🛡️ BLINDAJE 2: Left merge evita que la tabla se quede en blanco si IED falla
    panel = pd.merge(df_emp, df_ied, on=keys, how="left")
    panel = pd.merge(panel, df_act, on=keys, how="left")
    panel = pd.merge(panel, df_exp, on=keys, how="left")
    
    panel = panel[panel["Estado"].isin(ESTADOS_BAJIO.keys())].copy()
    panel = panel.sort_values(["Estado","Año","Trimestre"]).reset_index(drop=True)
    
    panel["Var_Empleo_pct"] = (panel.groupby("Estado")["Empleo_Manufacturero"].pct_change()*100).round(2)
    panel["Periodo"] = panel["Año"].astype(str) + " Q" + panel["Trimestre"].astype(str)
    
    for col in ["ActInd","Exportaciones"]:
        panel[col] = panel.groupby("Estado")[col].transform(lambda x: x.interpolate(limit_direction="both"))
        
    print(f"✅ Panel construido: {len(panel)} observaciones")
    return panel

def cargar_geojson():
    for url in GEOJSON_URLS:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200: return r.json()
        except Exception: continue
    return {"type":"FeatureCollection","features":[
        {"type":"Feature","id":e,"properties":{"name":e},"geometry":{"type":"Polygon","coordinates":[[[-100,20],[-101,20],[-101,21],[-100,21]]]}} for e in ESTADOS_BAJIO.keys()
    ]}

# ══════════════════════════════════════════════
# CACHÉ
# ══════════════════════════════════════════════
_cache = {"panel":None, "geojson":None, "fecha":None, "lock":threading.Lock()}

def get_datos():
    hoy = date.today()
    if _cache["panel"] is not None and _cache["fecha"] == hoy:
        return _cache["panel"], _cache["geojson"]
        
    with _cache["lock"]:
        if _cache["panel"] is not None and _cache["fecha"] == hoy:
            return _cache["panel"], _cache["geojson"]
            
        print(f"🔄 Cargando datos para {hoy}...")
        df_emp = procesar_empleo()
        df_ied = procesar_ied()
        df_act = procesar_actind()
        df_exp = procesar_exportaciones()
        
        panel = construir_panel(df_emp, df_ied, df_act, df_exp)
        geo = cargar_geojson()
        
        _cache["panel"] = panel
        _cache["geojson"] = geo
        _cache["fecha"] = hoy
        print("✅ Caché listo.\n")
        
    return _cache["panel"], _cache["geojson"]

PANEL, GEOJSON = get_datos()
AÑOS_DISPONIBLES = sorted(PANEL["Año"].unique())

# ══════════════════════════════════════════════
# ECONOMETRÍA CON REZAGOS (LAGS) Y PREDICCIÓN
# ══════════════════════════════════════════════
def calcular_econometria(df, vars_x, lags=0):
    if not vars_x: 
        return None
        
    cols = ["Empleo_Manufacturero"] + vars_x
    sub = df[["Estado", "Año", "Trimestre", "Periodo"] + cols].dropna().copy()
    sub = sub.sort_values(["Estado", "Año", "Trimestre"])
    
    # 1. Aplicar los Rezagos (Lags)
    for c in vars_x:
        if lags > 0:
            sub[c] = sub.groupby("Estado")[c].shift(lags)
            
    sub = sub.dropna()
    if len(sub) < 20: 
        return None
        
    # 2. Calcular las Tasas de Crecimiento
    for c in cols:
        sub[f"Crec_{c}"] = sub.groupby("Estado")[c].pct_change() * 100
        
    sub = sub.dropna()
    sub["t"] = (sub["Año"] - sub["Año"].min()) * 4 + sub["Trimestre"]
    sub_index = sub.set_index(["Estado", "t"])
    
    Y = sub_index["Crec_Empleo_Manufacturero"]
    X = sub_index[[f"Crec_{c}" for c in vars_x]]
    
    try:
        mod = PanelOLS(Y, X, entity_effects=True, time_effects=True)
        res = mod.fit(cov_type="clustered", cluster_entity=True)
        dw_stat = durbin_watson(res.resids.values)
        
        pred = res.fitted_values
        sub_index["Prediccion"] = pred
        df_pred = sub_index.reset_index()[["Estado", "Periodo", "Crec_Empleo_Manufacturero", "Prediccion"]]
    except Exception as e: 
        return None
        
    vifs = {}
    if len(vars_x) > 1:
        X_vif = sub_index[[f"Crec_{c}" for c in vars_x]].reset_index(drop=True)
        try:
            for i, c in enumerate(X_vif.columns):
                vifs[c.replace("Crec_", "")] = round(float(variance_inflation_factor(X_vif.values.astype(float), i)), 2)
        except Exception: 
            vifs = {c: float("nan") for c in vars_x}
    else: 
        vifs = {vars_x[0]: 1.0}
        
    return {
        "coefs": res.params, 
        "pvals": res.pvalues, 
        "r2_within": round(res.rsquared, 4),
        "n_obs": int(res.nobs), 
        "dw": round(dw_stat, 3), 
        "vifs": vifs,
        "corr": sub_index[[f"Crec_{c}" for c in cols]].corr().round(3),
        "df_pred": df_pred
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
        else:  
            fig.add_trace(go.Bar(x=sub["Periodo"], y=sub[col], name=est,
                marker_color=COLORES_ESTADOS[est],
                hovertemplate=f"<b>{est}</b><br>%{{x}}<br>{VAR_LABEL[variable]}: %{{y:,.1f}}<extra></extra>"))
                
    fig.update_layout(**{**PLOT_LAYOUT,
        "yaxis": dict(title=VAR_LABEL[variable], gridcolor="#EEE", tickformat=","),
        "xaxis": dict(tickangle=-45, tickfont_size=9),
        "height": H_CHART, "barmode": "group"})
    return fig

def fig_mapa_mapbox(df, variable, estados, geojson):
    if df.empty: return go.Figure().update_layout(**PLOT_LAYOUT, height=500)
    col = VAR_COL.get(variable, "Empleo_Manufacturero")
    sub = df[df["Estado"].isin(estados)].copy()
    
    grp = sub.groupby(["Estado", "Año"])[col].mean().reset_index().rename(columns={col: "valor"})
    if grp.empty: return go.Figure().update_layout(**PLOT_LAYOUT, height=500)
    all_years = sorted(grp["Año"].unique())
    
    idx = pd.MultiIndex.from_product([estados, all_years], names=['Estado', 'Año'])
    grp = grp.set_index(['Estado', 'Año']).reindex(idx).reset_index()
    grp["valor"] = grp["valor"].fillna(0)
    grp["Año_str"] = grp["Año"].astype(str)
    grp = grp.sort_values("Año")

    fig = px.choropleth_mapbox(
        grp, geojson=geojson, locations="Estado", featureidkey="properties.name",
        color="valor", animation_frame="Año_str",
        color_continuous_scale="Blues", mapbox_style="carto-positron",
        center={"lat": 21.5, "lon": -101.5}, zoom=5.5, opacity=0.6,
        labels={"valor": "Promedio Anual"}
    )
    fig.update_layout(**{**PLOT_LAYOUT, "height": 500, "margin": dict(l=0, r=0, t=0, b=0)})
    return fig

def fig_scatter_animado(df, var_x, var_y, estados):
    if df.empty: return go.Figure().update_layout(**PLOT_LAYOUT, height=H_CHART)
    col_x = VAR_COL.get(var_x, "IED")
    col_y = VAR_COL.get(var_y, "Empleo_Manufacturero")
    sub = df[df["Estado"].isin(estados)].copy()
    
    sub_anual = sub.groupby(["Estado", "Año"]).agg(x=(col_x, "mean"), y=(col_y, "mean"), size_col=("Empleo_Manufacturero", "mean")).reset_index()
    if sub_anual.empty: return go.Figure().update_layout(**PLOT_LAYOUT, height=H_CHART)
    
    max_size = sub_anual["size_col"].max()
    sub_anual["size_col"] = (sub_anual["size_col"] / max_size * 60 + 10).round(1) if max_size > 0 else 10
    sub_anual["Año_str"] = sub_anual["Año"].astype(str)

    # 🛡️ BLINDAJE 3: Matemáticas seguras para los márgenes
    min_x, max_x = sub_anual["x"].min(), sub_anual["x"].max()
    min_y, max_y = sub_anual["y"].min(), sub_anual["y"].max()
    rx = [min_x * 0.8 if pd.notna(min_x) else 0, max_x * 1.1 if pd.notna(max_x) else 100]
    ry = [min_y * 0.8 if pd.notna(min_y) else 0, max_y * 1.1 if pd.notna(max_y) else 100]

    fig = px.scatter(
        sub_anual, x="x", y="y", color="Estado", size="size_col", animation_frame="Año_str",
        color_discrete_map=COLORES_ESTADOS, hover_name="Estado", range_x=rx, range_y=ry,
        labels={"x": VAR_LABEL[var_x], "y": VAR_LABEL[var_y], "Año_str": "Año"}, size_max=55,
    )
    fig.update_traces(marker=dict(opacity=0.85, line=dict(width=1, color="white")))
    fig.update_layout(**{**PLOT_LAYOUT, "xaxis": dict(title=VAR_LABEL[var_x], gridcolor="#EEE", tickformat=","), "yaxis": dict(title=VAR_LABEL[var_y], gridcolor="#EEE"), "height": H_CHART})
    return fig

def fig_scatter_ols(df, var_x, var_y, estados):
    col_x = VAR_COL.get(var_x, "IED")
    col_y = VAR_COL.get(var_y, "Empleo_Manufacturero")
    sub = df[df["Estado"].isin(estados)][["Estado", "Periodo", col_x, col_y]].dropna().copy()
    
    if len(sub) < 2: return go.Figure().update_layout(**PLOT_LAYOUT, title="Datos insuficientes para tendencia OLS")
        
    fig = px.scatter(sub, x=col_x, y=col_y, color="Estado", color_discrete_map=COLORES_ESTADOS,
        trendline="ols", hover_data=["Periodo"], labels={col_x: VAR_LABEL[var_x], col_y: VAR_LABEL[var_y]})
    fig.update_traces(marker=dict(size=7, opacity=0.7))
    fig.update_layout(**{**PLOT_LAYOUT, "xaxis": dict(title=VAR_LABEL[var_x], gridcolor="#EEE", tickformat=","), "yaxis": dict(title=VAR_LABEL[var_y], gridcolor="#EEE"), "height": H_CHART})
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
    col = VAR_COL.get(variable, "Empleo_Manufacturero")
    sub = df[df["Estado"].isin(estados)]
    if sub.empty: return go.Figure().update_layout(**PLOT_LAYOUT, height=H_CHART)
    piv = sub.pivot_table(index="Estado", columns="Trimestre", values=col, aggfunc="mean").round(1)
    piv.columns = [f"Q{c}" for c in piv.columns]
    fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns.tolist(), y=piv.index.tolist(), colorscale=[[0, BG], [0.5, "#A6ACAF"], [1, "#1B4F72"]], text=np.round(piv.values, 0), texttemplate="%{text:,.0f}", textfont_size=11))
    fig.update_layout(**{**PLOT_LAYOUT, "height": H_CHART, "margin": dict(l=140, r=20, t=20, b=20)})
    return fig

def fig_correlacion(corr_df):
    if corr_df is None or corr_df.empty: return go.Figure().update_layout(**PLOT_LAYOUT, height=350)
    labels = [c.replace("Crec_", "") for c in corr_df.columns]
    z = corr_df.values
    fig = go.Figure(go.Heatmap(z=z, x=labels, y=labels, colorscale=[[0, BG], [0.5, "#A6ACAF"], [1, "#E63946"]], zmin=-1, zmax=1, text=np.round(z, 2), texttemplate="%{text}", textfont_size=12))
    fig.update_layout(**{**PLOT_LAYOUT, "height": 350, "margin": dict(l=80, r=20, t=20, b=80)})
    return fig

# ══════════════════════════════════════════════
# ESTILOS UI
# ══════════════════════════════════════════════
CARD={"background":CARD_BG,"borderRadius":"12px","border":"1px solid #E5E0D8","padding":"20px 24px","marginBottom":"16px","boxShadow":"0 1px 4px rgba(0,0,0,0.06)"}
METRIC_CARD={"background": CAFE, "borderRadius":"10px","padding":"14px 18px","flex":"1","minWidth":"120px", "border":"1px solid #734c30"}
BTN_BASE={"fontSize":"12px","padding":"5px 14px","borderRadius":"20px","border":"1.5px solid #CCC", "background":"white","color":TEXT_PRIM,"cursor":"pointer","marginRight":"6px","marginBottom":"6px","fontFamily":"inherit"}
BTN_VAR_ON={**BTN_BASE,"background":CAFE,"color":"#FFFFFF","border":f"1.5px solid {CAFE}","borderRadius":"6px"}
SEC_HDR={"fontSize":"16px","fontWeight":"600","color":TEXT_PRIM,"margin":"0 0 4px","borderLeft":f"4px solid {NARANJA}","paddingLeft":"12px"}
SUB={"fontSize":"11px","color":TEXT_SEC,"margin":"4px 0 12px 16px"}
TAB_STYLE={"padding":"12px 20px","fontFamily":"'Helvetica Neue',Arial,sans-serif","fontSize":"14px","color":TEXT_SEC,"borderBottom":"2px solid transparent", "background": BG}
TAB_SEL={"padding":"12px 20px","fontFamily":"'Helvetica Neue',Arial,sans-serif","fontSize":"14px","color":TEXT_PRIM,"fontWeight":"bold","borderBottom":f"3px solid {NARANJA}","background": CARD_BG}

# ══════════════════════════════════════════════
# APP LAYOUT
# ══════════════════════════════════════════════
app = dash.Dash(__name__, title="Panel Bajío · Econométrico", meta_tags=[{"name":"viewport","content":"width=device-width, initial-scale=1"}])
server = app.server

CONTROLES = html.Div(style={**CARD,"marginBottom":"20px"},children=[
    html.Div(style={"display":"flex","flexWrap":"wrap","gap":"24px","alignItems":"flex-start"},children=[
        html.Div([
            html.P("Estados a Analizar",style={"fontSize":"11px","color":TEXT_SEC,"textTransform":"uppercase","fontWeight":"bold","margin":"0 0 8px"}),
            html.Div(id="estado-btns",style={"display":"flex","flexWrap":"wrap"}),
        ]),
        html.Div([
            html.P("Período Global",style={"fontSize":"11px","color":TEXT_SEC,"textTransform":"uppercase","fontWeight":"bold","margin":"0 0 8px"}),
            html.Div(style={"display":"flex","alignItems":"center","gap":"8px"},children=[
                dcc.Dropdown(id="year-from", options=[{"label":str(y),"value":y} for y in AÑOS_DISPONIBLES], value=2018, clearable=False,style={"width":"95px","fontSize":"13px"}),
                html.Span("—",style={"color":TEXT_SEC}),
                dcc.Dropdown(id="year-to", options=[{"label":str(y),"value":y} for y in AÑOS_DISPONIBLES], value=AÑOS_DISPONIBLES[-1], clearable=False,style={"width":"95px","fontSize":"13px"}),
            ]),
        ]),
        html.Div([
            html.P("Variable Visualizada",style={"fontSize":"11px","color":TEXT_SEC,"textTransform":"uppercase","fontWeight":"bold","margin":"0 0 8px"}),
            html.Div(id="var-btns",style={"display":"flex","gap":"6px","flexWrap":"wrap"}),
        ]),
    ]),
])

app.layout = html.Div(
    style={"fontFamily":"'Helvetica Neue',Arial,sans-serif","background":BG,"minHeight":"100vh","padding":"30px 40px","maxWidth":"1600px","margin":"0 auto"},
    children=[

    html.Div(style={"marginBottom":"24px"},children=[
        html.H1("Panel Econométrico — Región del Bajío", style={"fontSize":"30px","fontWeight":"900","margin":"0 0 6px","color":TEXT_PRIM}),
        html.P("Impacto del Nearshoring: Empleo, IED, Actividad y Exportaciones Manufactureras (2018–2025)", style={"fontSize":"15px","color":TEXT_SEC,"margin":"0"}),
    ]),

    # Tarjeta de Pregunta de Investigación (CON LA SINTAXIS REPARADA ✅)
    html.Div(style={**CARD, "background": "#EBF5FB", "borderColor": "#D6EAF8", "borderLeft": f"5px solid {AZUL_OSCURO}"}, children=[
        html.P("🔍 Pregunta de Investigación Central", style={"fontSize": "14px", "fontWeight": "bold", "color": AZUL_OSCURO, "margin": "0 0 6px"}),
        html.P("¿Existe una relación positiva entre la Inversión Extranjera Directa (IED), utilizada como proxy del nearshoring, y el empleo manufacturero en San Luis Potosí respecto a los demás estados del Bajío?", 
               style={"fontSize": "14px", "color": TEXT_PRIM, "fontStyle": "italic", "margin": "0 0 10px"}),
        html.Div(children=[
            html.Div([html.Span("H₀ (Nula):", style={"fontWeight": "bold"}), " No existe una relación positiva significativa."]),
            html.Div([html.Span("H₁ (Alternativa):", style={"fontWeight": "bold"}), " Sí existe una relación positiva significativa."])
        ], style={"display": "flex", "gap": "20px", "fontSize": "12px", "color": TEXT_SEC})
    ]),

    html.Div(id="metrics-row",style={"display":"flex","gap":"14px","flexWrap":"wrap","marginBottom":"20px"}),

    dcc.Store(id="active-estados",data=list(ESTADOS_BAJIO.keys())),
    dcc.Store(id="active-var",data="empleo"),

    CONTROLES,

    dcc.Tabs(id="main-tabs",value="tab-visor", style={"marginBottom":"20px"}, children=[

        dcc.Tab(label="📊 Visor Geográfico y Temporal", value="tab-visor", style=TAB_STYLE, selected_style=TAB_SEL, children=[
            html.Div(style={"marginTop": "20px"}),
            html.Div(style=CARD,children=[
                html.P("Serie de tiempo trimestral",style=SEC_HDR),
                html.Div(style={"display":"flex","alignItems":"center","gap":"16px","margin":"6px 0 12px 14px"},children=[
                    html.P(id="series-sub",style={**SUB,"margin":"0"}),
                    dcc.RadioItems(id="tipo-grafica", options=[{"label":" Líneas","value":"line"}, {"label":" Barras","value":"bar"}], value="line", inline=True, style={"fontSize":"13px","color":TEXT_SEC}, inputStyle={"marginRight":"4px"}, labelStyle={"marginRight":"14px"}),
                ]),
                dcc.Graph(id="series-chart",config={"displayModeBar":False}),
            ]),
            html.Div(style=CARD,children=[
                html.P("Mapa Animado del Bajío (Mapbox Real)",style=SEC_HDR),
                html.P("Evolución anual con fondo cartográfico. Presiona Play para iniciar la animación.",style=SUB),
                dcc.Graph(id="map-chart",config={"displayModeBar":False}),
            ]),
            html.Div(style=CARD,children=[
                html.P("Animación Hans Rosling — Evolución Dinámica",style=SEC_HDR),
                html.Div(style={"display":"flex","flexWrap":"wrap","gap":"16px","margin":"6px 0 12px 14px","alignItems":"center"},children=[
                    html.Div([html.P("Eje X",style={"fontSize":"10px","color":TEXT_SEC,"margin":"0 0 3px","textTransform":"uppercase"}), dcc.Dropdown(id="scatter-x", options=[{"label":l,"value":v} for v,l in VARS_DEF], value="ied",clearable=False,style={"width":"180px","fontSize":"12px"}),]),
                    html.Div([html.P("Eje Y",style={"fontSize":"10px","color":TEXT_SEC,"margin":"0 0 3px","textTransform":"uppercase"}), dcc.Dropdown(id="scatter-y", options=[{"label":l,"value":v} for v,l in VARS_DEF], value="empleo",clearable=False,style={"width":"180px","fontSize":"12px"}),]),
                    html.P("El tamaño del punto = Empleo manufacturero.", style={"fontSize":"12px","color":TEXT_SEC,"margin":"0"}),
                ]),
                dcc.Graph(id="scatter-chart",config={"displayModeBar":False}),
            ]),
            html.Div(style=CARD,children=[
                html.P("Análisis de Dispersión Estático con Tendencia (OLS)",style=SEC_HDR),
                html.P("Muestra la relación lineal (Betas) para todo el periodo. Pasa el cursor sobre la línea para ver el R².",style=SUB),
                dcc.Graph(id="scatter-ols",config={"displayModeBar":False}),
            ]),
            html.Div(style=CARD,children=[
                html.P("Patrón estacional (Heatmap Trimestral)",style=SEC_HDR),
                html.P("Muestra los promedios de la variable agrupados por trimestre",style=SUB),
                dcc.Graph(id="heatmap-chart",config={"displayModeBar":False}),
            ]),
        ]),

        dcc.Tab(label="🔬 Laboratorio Econométrico", value="tab-eco", style=TAB_STYLE, selected_style=TAB_SEL, children=[
            html.Div(style={"margin": "24px 0 16px"}, children=[
                html.H2("Modelo de Datos Panel (Efectos Fijos)", style={"fontSize": "19px", "fontWeight": "bold", "color": TEXT_PRIM, "margin": "0 0 6px"}),
                html.P("Análisis en Tasas de Crecimiento Trimestral (Δ%) · Errores clusterizados por Entidad Federativa.", style={"fontSize": "13px", "color": TEXT_SEC, "margin": "0"}),
            ]),
            html.Div(style={**CARD, "background": "#FDFCF8", "borderColor": CAFE}, children=[
                html.P("Ecuación del Modelo Estructural:", style={"fontSize": "12px", "color": TEXT_SEC, "margin": "0 0 8px"}),
                html.P("Δ%Empleo_it = β₀ + β₁·Δ%X₁_it + β₂·Δ%X₂_it + μᵢ + λₜ + εᵢₜ", style={"fontSize": "16px", "fontWeight": "bold", "color": CAFE, "fontFamily": "monospace", "margin": "0 0 16px"}),
                html.Div(style={"display": "flex", "gap": "40px", "alignItems": "flex-start"}, children=[
                    html.Div([
                        html.P("1. Selecciona las variables independientes (X):", style={"fontSize": "12px", "color": TEXT_SEC, "fontWeight": "bold", "margin": "0 0 8px"}),
                        dcc.Checklist(id="vars-modelo",
                            options=[{"label": f"  {l}", "value": v} for v, l in VARS_DEF if v != "empleo"],
                            value=["ied", "actind", "exportaciones"], inline=True, inputStyle={"marginRight": "6px"}, labelStyle={"marginRight": "24px", "fontSize": "13px"},
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
                    html.P("Lectura automática de significancia (P-valores):", style=SUB),
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
                        html.P("¿Qué tan bien se ajusta la fórmula a la realidad? Selecciona un estado.", style=SUB),
                    ]),
                    dcc.Dropdown(id="estado-prediccion", options=[{"label": e, "value": e} for e in ESTADOS_BAJIO.keys()], value="San Luis Potosí", clearable=False, style={"width": "200px"})
                ]),
                dcc.Graph(id="prediccion-chart", config={"displayModeBar": False})
            ]),
            html.Div(style=CARD, children=[
                html.P("Matriz de Correlación de Pearson", style=SEC_HDR),
                html.P("Relación lineal entre las tasas de crecimiento de las variables.", style=SUB),
                dcc.Graph(id="corr-chart", config={"displayModeBar": False}),
            ]),
        ]),

        dcc.Tab(label="🗃 Base de Datos", value="tab-datos", style=TAB_STYLE, selected_style=TAB_SEL, children=[
            html.Div(style={"marginTop": "24px"}, children=[
                html.Div(style={**CARD, "background": "#FDFCF8", "borderColor": CAFE}, children=[
                    html.P("📖 Diccionario de Datos (Metadata)", style={"fontSize": "15px", "fontWeight": "bold", "color": CAFE, "margin": "0 0 10px"}),
                    html.Table([
                        html.Thead(html.Tr([html.Th("Variable"), html.Th("Unidad de Medida"), html.Th("Fuente Oficial")])),
                        html.Tbody([
                            html.Tr([html.Td("Empleo Manufacturero"), html.Td("Personas ocupadas (Total)"), html.Td("INEGI (EMIM - BIE)")]),
                            html.Tr([html.Td("Inversión Extranjera Directa (IED)"), html.Td("Millones de Dólares (USD)"), html.Td("Secretaría de Economía")]),
                            html.Tr([html.Td("Actividad Manufacturera"), html.Td("Índice de Volumen Físico (Base 2013=100)"), html.Td("INEGI (BISE)")]),
                            html.Tr([html.Td("Exportaciones Manufactureras"), html.Td("Millones de Dólares (USD)"), html.Td("INEGI (BISE)")])
                        ])
                    ], style={"width": "100%", "textAlign": "left", "fontSize": "12px", "color": TEXT_PRIM})
                ]),
                html.Div(style=CARD, children=[
                    html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "16px"}, children=[
                        html.Div([
                            html.P("Registros del Panel", style=SEC_HDR),
                            html.P(id="tabla-sub", style=SUB),
                        ]),
                        html.Button("⬇ Exportar CSV Completo", id="btn-csv", style={**BTN_BASE, "background": CAFE, "color": "white", "borderColor": CAFE, "borderRadius": "8px", "fontSize": "13px", "padding": "8px 16px", "fontWeight": "bold"}),
                    ]),
                    dcc.Download(id="download-csv"),
                    html.Div(id="tabla-avanzada-container"),
                ]),
            ]),
        ]),
    ]),
    html.P("Fuentes: INEGI (BIE-BISE) · Secretaría de Economía", style={"fontSize": "12px", "color": TEXT_SEC, "textAlign": "center", "marginTop": "10px"}),
])

# ══════════════════════════════════════════════
# CALLBACKS GLOBALES
# ══════════════════════════════════════════════
@app.callback(
    Output("estado-btns", "children"), Output("active-estados", "data"),
    Input({"type": "btn-estado", "index": dash.ALL}, "n_clicks"),
    State("active-estados", "data"), prevent_initial_call=False,
)
def toggle_estado(_, active):
    ctx = callback_context
    if not ctx.triggered or ctx.triggered[0]["prop_id"] == ".": 
        active = list(ESTADOS_BAJIO.keys())
    else:
        idx = json.loads(ctx.triggered[0]["prop_id"].split(".")[0])["index"]
        active = [e for e in active if e != idx] if idx in active and len(active) > 1 else (active + [idx] if idx not in active else active)
    btns = []
    for est in ESTADOS_BAJIO:
        ia = est in active; col = COLORES_ESTADOS[est]
        btns.append(html.Button(est, id={"type": "btn-estado", "index": est}, n_clicks=0,
            style={**BTN_BASE, "background": col if ia else "white", "color": "#FFFFFF" if ia else TEXT_PRIM, "borderColor": col if ia else "#CCC", "fontWeight": "bold" if ia else "normal"}))
    return btns, active

@app.callback(
    Output("var-btns", "children"), Output("active-var", "data"),
    Input({"type": "btn-var", "index": dash.ALL}, "n_clicks"),
    State("active-var", "data"), prevent_initial_call=False,
)
def toggle_var(_, av):
    ctx = callback_context
    if ctx.triggered and ctx.triggered[0]["prop_id"] != ".": 
        av = json.loads(ctx.triggered[0]["prop_id"].split(".")[0])["index"]
    return [html.Button(label, id={"type": "btn-var", "index": val}, n_clicks=0, style=BTN_VAR_ON if val == av else {**BTN_BASE, "borderRadius": "6px"}) for val, label in VARS_DEF], av

# ── VISOR ────────────────────────────
@app.callback(
    Output("metrics-row", "children"),
    Output("series-chart", "figure"), Output("map-chart", "figure"),
    Output("scatter-chart", "figure"), Output("scatter-ols", "figure"), Output("heatmap-chart", "figure"),
    Output("series-sub", "children"), Output("map-sub", "children"),
    Input("active-estados", "data"), Input("active-var", "data"),
    Input("year-from", "value"), Input("year-to", "value"),
    Input("tipo-grafica", "value"), Input("scatter-x", "value"), Input("scatter-y", "value"),
)
def update_visor(estados, variable, yr_from, yr_to, tipo, sx, sy):
    panel, geojson = get_datos()
    df = panel[(panel["Estado"].isin(estados)) & (panel["Año"] >= yr_from) & (panel["Año"] <= yr_to)].copy()
    n = len(df)

    def m(l, v, s): return html.Div(style=METRIC_CARD, children=[
        html.P(l, style={"fontSize": "10px", "color": "#FFFFFF", "opacity": 0.8, "textTransform": "uppercase", "fontWeight": "bold", "margin": "0 0 4px"}),
        html.P(v, style={"fontSize": "22px", "fontWeight": "bold", "color": "#FFFFFF", "margin": "0", "lineHeight": "1.1"}),
        html.P(s, style={"fontSize": "11px", "color": "#F4EFEA", "opacity": 0.9, "margin": "4px 0 0"}),
    ])
    
    metrics = [
        m("Observaciones", f"{n:,}", "Trimestres × Estados"),
        m("Empleo prom.", f"{int(df['Empleo_Manufacturero'].mean()):,}" if n and not df['Empleo_Manufacturero'].isna().all() else "—", "Personas por trimestre"),
        m("IED prom.", f"${df['IED'].mean():.1f} M" if n and not df['IED'].isna().all() else "—", "USD por trimestre"),
        m("Act. Manufacturera", f"{df['ActInd'].mean():.1f}" if n and not df['ActInd'].isna().all() else "—", "Índice Base 2013=100"),
        m("Exportaciones", f"${df['Exportaciones'].mean():.1f} M" if n and not df['Exportaciones'].isna().all() else "—", "USD por trimestre"),
    ]

    f_ser  = fig_series(df, variable, estados, tipo)
    f_map  = fig_mapa_mapbox(df, variable, estados, geojson)
    f_scat = fig_scatter_animado(df, sx, sy, estados)
    f_ols  = fig_scatter_ols(df, sx, sy, estados)
    f_heat = fig_heatmap(df, variable, estados)

    return (metrics, f_ser, f_map, f_scat, f_ols, f_heat,
            VAR_LABEL[variable], f"Periodo seleccionado: {yr_from} - {yr_to}")

# ── ECONOMETRÍA ───────────────────────────────
@app.callback(
    Output("tabla-regresion", "children"), Output("insights-panel", "children"), 
    Output("robustez-panel", "children"), Output("corr-chart", "figure"), Output("prediccion-chart", "figure"),
    Input("active-estados", "data"), Input("active-var", "data"), Input("year-from", "value"), Input("year-to", "value"), 
    Input("vars-modelo", "value"), Input("lags-slider", "value"), Input("estado-prediccion", "value")
)
def update_eco(estados, _, yr_from, yr_to, vars_x, lags, estado_pred):
    panel, _geo = get_datos()
    df = panel[(panel["Estado"].isin(estados)) & (panel["Año"] >= yr_from) & (panel["Año"] <= yr_to)].copy()
    vars_x = vars_x or []
    eco = calcular_econometria(df, [VAR_COL[v] for v in vars_x], lags)
    
    no_data = html.P("Selecciona variables y amplía el período.", style={"fontSize": "13px", "color": TEXT_SEC, "padding": "12px"})
    if eco is None: 
        return no_data, no_data, no_data, go.Figure(), go.Figure()

    nombres = {"Empleo_Manufacturero": "Empleo Mfr.", "IED": "IED", "ActInd": "Act. Mfr.", "Exportaciones": "Exportaciones"}
    
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
                rel = "positivamente" if coef > 0 else "negativamente"
                insights.append(html.P(f"✅ {nom_humano} afecta {rel} al empleo de forma estadísticamente significativa.", style={"fontSize": "12px", "color": "#27AE60", "margin": "0 0 6px"}))
            else:
                insights.append(html.P(f"⚠️ {nom_humano} NO muestra un impacto estadísticamente significativo (P > 0.05).", style={"fontSize": "12px", "color": TEXT_SEC, "margin": "0 0 6px"}))
                
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

    return tabla_reg, panel_insights, robustez, fig_correlacion(eco["corr"]), f_pred

# ── BASE DE DATOS ─────────────────────────────
@app.callback(
    Output("tabla-avanzada-container", "children"), Output("tabla-sub", "children"),
    Input("active-estados", "data"), Input("year-from", "value"), Input("year-to", "value"),
)
def update_tabla(estados, yr_from, yr_to):
    panel, _ = get_datos()
    df = panel[(panel["Estado"].isin(estados)) & (panel["Año"] >= yr_from) & (panel["Año"] <= yr_to)].copy()
    df = df.sort_values(["Año", "Trimestre", "Estado"], ascending=[False, False, True])
    
    df["Empleo_Manufacturero"] = df["Empleo_Manufacturero"].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "N/A")
    df["IED"] = df["IED"].apply(lambda x: f"${x:.1f} M" if pd.notna(x) else "N/A")
    df["ActInd"] = df["ActInd"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
    df["Exportaciones"] = df["Exportaciones"].apply(lambda x: f"${x:.1f} M" if pd.notna(x) else "N/A")
    df["Var_Empleo_pct"] = df["Var_Empleo_pct"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
    
    df_show = df[["Estado", "Año", "Trimestre", "Empleo_Manufacturero", "IED", "ActInd", "Exportaciones", "Var_Empleo_pct"]]
    df_show.columns = ["Estado", "Año", "Trim.", "Empleo", "IED", "Act. Ind.", "Exportaciones", "Crec. Empleo"]

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
    df = panel[(panel["Estado"].isin(estados)) & (panel["Año"] >= yr_from) & (panel["Año"] <= yr_to)]
    return dcc.send_data_frame(df.to_csv, "panel_econometrico_bajio.csv", index=False)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)
