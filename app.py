"""
Dashboard Econométrico — Región del Bajío
==========================================
Variables: Empleo Manufacturero, IED, Actividad Industrial, Exportaciones
Módulo econométrico: PanelOLS (FE), Correlación, VIF, Durbin-Watson
Mapa coroplético interactivo por estado
"""

import os
import json
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import plotly.express as px

# Econometría
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
    "Aguascalientes":  "01",
    "Guanajuato":      "11",
    "Jalisco":         "14",
    "Querétaro":       "22",
    "San Luis Potosí": "24",
}

COLORES_ESTADOS = {
    "Aguascalientes":  "#0C3460",
    "Guanajuato":      "#1A5599",
    "Jalisco":         "#3A82C4",
    "Querétaro":       "#8B5E3C",
    "San Luis Potosí": "#C4955A",
}

YEARS = list(range(2015, 2026))

AZUL_OSCURO = "#0C3460"
AZUL_MEDIO  = "#1A5599"
AZUL_CLARO  = "#3A82C4"
CAFE        = "#8B5E3C"
NARANJA     = "#D97B2A"
BG          = "#F8F7F4"
CARD_BG     = "#FFFFFF"
TEXT_PRIM   = "#1A1A1A"
TEXT_SEC    = "#6B6B6B"

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'Helvetica Neue', Arial, sans-serif", color=TEXT_PRIM, size=11),
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", y=-0.18, x=0, font_size=10),
)

# ══════════════════════════════════════════════
# INDICADORES INEGI
# NOTA: Sustituye los IDs placeholder por los
# exactos del BIE para cada estado.
# Búscalos en: https://www.inegi.org.mx/app/indicadores/
# ══════════════════════════════════════════════
INDICADORES_EMPLEO = {
    "Aguascalientes":  "6207064389",
    "Guanajuato":      "6207064407",
    "Jalisco":         "6207064416",
    "Querétaro":       "6207064443",
    "San Luis Potosí": "6207064447",
}

# Actividad Industrial Manufacturera — REEMPLAZAR con IDs reales del BIE por estado
INDICADORES_ACTIND = {
    "Aguascalientes":  "738413",
    "Guanajuato":      "738414",
    "Jalisco":         "738415",
    "Querétaro":       "738416",
    "San Luis Potosí": "738417",
}

# Exportaciones manufactureras — REEMPLAZAR con IDs reales del BIE por estado
INDICADORES_EXPORTACIONES = {
    "Aguascalientes":  "127595",
    "Guanajuato":      "739277",
    "Jalisco":         "739278",
    "Querétaro":       "739279",
    "San Luis Potosí": "739280",
}

# ── BISE: indicador único, código geográfico varía por estado ──────────
# ESTADOS_BAJIO ya contiene los códigos: "01","11","14","22","24"
# Se inyectan en la URL al iterar cada estado (ver procesar_actind / procesar_exportaciones)

IND_ACTIND        = "738413"        # Actividad Industrial Manufacturera (mensual)
IND_EXPORTACIONES = "924,739277"    # Exportaciones Manufactureras (trimestral, 2 series)

SE_IED_URL = (
    "https://datos.gob.mx/busca/api/action/datastore_search"
    "?resource_id=fc1e3b7b-4027-4c59-9e5a-f02f48e90ca1&limit=5000"
)

GEOJSON_URLS = [
    "https://raw.githubusercontent.com/PhantomInsights/mexican-geojson/main/src/states/states.json",
    "https://raw.githubusercontent.com/angelnmara/geojson/master/mexicoHigh.json",
]

# ══════════════════════════════════════════════
# VALORES BASE PARA SIMULACIÓN
# ══════════════════════════════════════════════
_EMP_BASE = {"Aguascalientes":95000,"Guanajuato":340000,"Jalisco":280000,"Querétaro":160000,"San Luis Potosí":110000}
_IED_BASE = {"Aguascalientes":220,"Guanajuato":315,"Jalisco":430,"Querétaro":265,"San Luis Potosí":225}
_ACT_BASE = {"Aguascalientes":108,"Guanajuato":115,"Jalisco":112,"Querétaro":120,"San Luis Potosí":106}
_EXP_BASE = {"Aguascalientes":4200,"Guanajuato":9500,"Jalisco":7800,"Querétaro":5600,"San Luis Potosí":3900}

# ══════════════════════════════════════════════
# UTILIDADES DE EXTRACCIÓN
# ══════════════════════════════════════════════
def fetch_inegi_serie(
    indicador: str,
    fuente: str = "BIE",
    geo: str = "00",            # "00" = nacional; "01"–"32" = estado
) -> pd.DataFrame:
    """
    Descarga una o varias series del BIE/BISE del INEGI.

    Parámetros:
        indicador : ID(s) del indicador, ej. "738413" o "924,739277"
        fuente    : "BIE" para empleo; "BIE-BISE" para Act.Industrial y Exportaciones
        geo       : Código geográfico de 2 dígitos del estado (de ESTADOS_BAJIO)
                    "00" = nivel nacional (default para empleo, que usa IDs por estado)

    La URL resultante sigue la estructura oficial del BISE:
    .../INDICATOR/{indicador}/es/{geo}/false/{fuente}/2.0/{token}?type=json
    """
    url = (
        f"https://www.inegi.org.mx/app/api/indicadores/desarrolladores/"
        f"jsonxml/INDICATOR/{indicador}/es/{geo}/false/{fuente}/2.0/{TOKEN_INEGI}?type=json"
    )
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        # La respuesta puede tener múltiples Series (cuando el indicador es "924,739277")
        # Tomamos la primera que tenga datos
        all_rows = []
        for serie in data.get("Series", []):
            obs = serie.get("OBSERVATIONS", [])
            for s in obs:
                if s["OBS_VALUE"] not in (None, "", "N/A"):
                    try:
                        all_rows.append({"fecha": s["TIME_PERIOD"], "valor": float(s["OBS_VALUE"])})
                    except (ValueError, TypeError):
                        pass
        if not all_rows:
            raise ValueError("Serie vacía o sin observaciones válidas")
        return pd.DataFrame(all_rows)
    except Exception as e:
        print(f"    [INEGI] indicador={indicador} geo={geo}: {e}")
        return pd.DataFrame(columns=["fecha", "valor"])


def _mensual_a_trimestral(df: pd.DataFrame, estado: str, col: str) -> pd.DataFrame:
    df["fecha"] = pd.to_datetime(df["fecha"], format="%Y/%m", errors="coerce")
    df = df.dropna(subset=["fecha"])
    df["Año"] = df["fecha"].dt.year
    df["Mes"] = df["fecha"].dt.month
    df = df[(df["Año"]>=2015)&(df["Año"]<=2025)]
    df["Trimestre"] = df["Mes"].apply(lambda m: (m-1)//3+1)
    return df.groupby([pd.Series([estado]*len(df), name="Estado"), "Año","Trimestre"])["valor"].mean().reset_index().rename(columns={"valor":col})


def _parse_trimestral(df: pd.DataFrame, estado: str, col: str) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        t = str(row["fecha"]).replace("-","/")
        try:
            partes = t.split("/")
            yr = int(partes[0])
            qn = int(partes[1].replace("Q","").replace("T",""))
            if 2015<=yr<=2025:
                rows.append({"Estado":estado,"Año":yr,"Trimestre":qn,col:row["valor"]})
        except Exception:
            pass
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Estado","Año","Trimestre",col])

# ══════════════════════════════════════════════
# SIMULACIONES (fallback)
# ══════════════════════════════════════════════
def _sim_empleo_mensual(estado: str) -> pd.DataFrame:
    rng = np.random.default_rng(abs(hash(estado))%(2**32))
    base = _EMP_BASE.get(estado,100000)
    rows = []
    for yr in YEARS:
        for m in range(1,13):
            if yr==2025 and m>3: break
            t = (yr-2015)*12+m
            rows.append({"fecha":pd.Timestamp(yr,m,1),"valor":int(base*(1+0.018*t/12)*rng.uniform(0.97,1.03)*[.97,.98,1.,.101,1.02,1.02,1.01,1.01,1.,.99,.98,.97][m-1])})
    return pd.DataFrame(rows)


def _sim_ied() -> pd.DataFrame:
    rows = []
    for estado,base in _IED_BASE.items():
        rng = np.random.default_rng(abs(hash(estado+"ied"))%(2**32))
        for yr in YEARS:
            for q in range(1,5):
                if yr==2025 and q>1: break
                t=(yr-2015)*4+q
                rows.append({"Estado":estado,"Año":yr,"Trimestre":q,"IED":round(base*(1+0.02*t)*rng.uniform(0.88,1.20)*[1.,1.1,1.05,1.15][q-1],1)})
    return pd.DataFrame(rows)


def _sim_actind() -> pd.DataFrame:
    rows = []
    for estado,base in _ACT_BASE.items():
        rng = np.random.default_rng(abs(hash(estado+"act"))%(2**32))
        for yr in YEARS:
            for q in range(1,5):
                if yr==2025 and q>1: break
                t=(yr-2015)*4+q
                rows.append({"Estado":estado,"Año":yr,"Trimestre":q,"ActInd":round(base*(1+0.015*t)*rng.uniform(0.94,1.06)*[.98,1.01,1.02,.99][q-1],1)})
    return pd.DataFrame(rows)


def _sim_exportaciones() -> pd.DataFrame:
    rows = []
    for estado,base in _EXP_BASE.items():
        rng = np.random.default_rng(abs(hash(estado+"exp"))%(2**32))
        for yr in YEARS:
            for q in range(1,5):
                if yr==2025 and q>1: break
                t=(yr-2015)*4+q
                rows.append({"Estado":estado,"Año":yr,"Trimestre":q,"Exportaciones":round(base*(1+0.022*t)*rng.uniform(0.85,1.18)*[.95,1.05,1.08,1.12][q-1],1)})
    return pd.DataFrame(rows)

# ══════════════════════════════════════════════
# ETL REAL CON FALLBACK
# ══════════════════════════════════════════════
def procesar_empleo() -> pd.DataFrame:
    print("📥 Empleo manufacturero (INEGI EMIM)...")
    frames = []
    for estado,ind in INDICADORES_EMPLEO.items():
        print(f"   → {estado}")
        df = fetch_inegi_serie(ind)
        if df.empty:
            sim = _sim_empleo_mensual(estado)
            sim["Estado"]=estado; sim["Año"]=sim["fecha"].dt.year; sim["Mes"]=sim["fecha"].dt.month
            sim=sim[(sim["Año"]>=2015)&(sim["Año"]<=2025)]
            sim["Trimestre"]=sim["Mes"].apply(lambda m:(m-1)//3+1)
            df_t=sim.groupby(["Estado","Año","Trimestre"])["valor"].mean().reset_index().rename(columns={"valor":"Empleo_Manufacturero"})
        else:
            df["Estado"]=estado
            df_t=_mensual_a_trimestral(df.drop(columns=["Estado"],errors="ignore"),estado,"Empleo_Manufacturero")
        df_t["Empleo_Manufacturero"]=df_t["Empleo_Manufacturero"].round(0).astype(int)
        frames.append(df_t)
    return pd.concat(frames,ignore_index=True)


def procesar_ied() -> pd.DataFrame:
    print("📥 IED (Secretaría de Economía)...")
    try:
        r=requests.get(SE_IED_URL,timeout=15); r.raise_for_status()
        recs=r.json().get("result",{}).get("records",[])
        if not recs: raise ValueError("Sin registros")
        df=pd.DataFrame(recs)
        ce=next((c for c in df.columns if "entidad" in c.lower() or "estado" in c.lower()),None)
        ca=next((c for c in df.columns if "año" in c.lower() or "anio" in c.lower()),None)
        ct=next((c for c in df.columns if "trim" in c.lower()),None)
        ci=next((c for c in df.columns if "ied" in c.lower() or "inversion" in c.lower()),None)
        if not all([ce,ca,ct,ci]): raise ValueError("Columnas no mapeadas")
        df=df.rename(columns={ce:"Estado",ca:"Año",ct:"Trimestre",ci:"IED"})
        df=df[df["Estado"].str.strip().isin(ESTADOS_BAJIO.keys())]
        for col in ["Año","Trimestre","IED"]: df[col]=pd.to_numeric(df[col],errors="coerce")
        df=df.dropna()[(df["Año"]>=2015)&(df["Año"]<=2025)]
        print(f"   ✓ {len(df)} registros")
        return df[["Estado","Año","Trimestre","IED"]]
    except Exception as e:
        print(f"   ⚠ SE no disponible ({e}), simulando.")
        return _sim_ied()


def procesar_actind() -> pd.DataFrame:
    print("📥 Actividad Industrial Manufacturera (INEGI)...")
    frames = []
    for estado,ind in INDICADORES_ACTIND.items():
        print(f"   → {estado}")
        
        # --- SOLUCIÓN APLICADA ---
        codigo_geo = ESTADOS_BAJIO[estado]
        df = fetch_inegi_serie(ind, fuente="BIE-BISE", geo=codigo_geo)
        # -------------------------
        
        if df.empty:
            frames.append(_sim_actind()[lambda d: d["Estado"]==estado])
        else:
            frames.append(_mensual_a_trimestral(df,estado,"ActInd"))
    result = pd.concat(frames,ignore_index=True)
    if result.empty: return _sim_actind()
    return result


def procesar_exportaciones() -> pd.DataFrame:
    print("📥 Exportaciones manufactureras (INEGI)...")
    frames = []
    for estado,ind in INDICADORES_EXPORTACIONES.items():
        print(f"   → {estado}")
        
        # --- SOLUCIÓN APLICADA ---
        # 1. Obtenemos la clave de 2 dígitos (ej. "01" para Aguascalientes)
        codigo_geo = ESTADOS_BAJIO[estado]
        
        # 2. Obligamos a la función a usar el estado real en lugar de "00"
        df = fetch_inegi_serie(ind, fuente="BIE-BISE", geo=codigo_geo)
        # -------------------------
        
        if not df.empty:
            df_t=_parse_trimestral(df,estado,"Exportaciones")
            if not df_t.empty:
                frames.append(df_t); continue
        rng=np.random.default_rng(abs(hash(estado+"exp"))%(2**32))
        base=_EXP_BASE.get(estado,5000)
        sim=[]
        for yr in YEARS:
            for q in range(1,5):
                if yr==2025 and q>1: break
                t=(yr-2015)*4+q
                sim.append({"Estado":estado,"Año":yr,"Trimestre":q,"Exportaciones":round(base*(1+0.022*t)*rng.uniform(0.85,1.18)*[.95,1.05,1.08,1.12][q-1],1)})
        frames.append(pd.DataFrame(sim))
    return pd.concat(frames,ignore_index=True)


def construir_panel(df_emp,df_ied,df_act,df_exp) -> pd.DataFrame:
    print("🔧 Construyendo panel maestro...")
    keys=["Estado","Año","Trimestre"]
    panel=df_emp.merge(df_ied,on=keys,how="inner")
    panel=panel.merge(df_act,on=keys,how="left")
    panel=panel.merge(df_exp,on=keys,how="left")
    panel=panel[panel["Estado"].isin(ESTADOS_BAJIO.keys())].copy()
    panel=panel.sort_values(["Estado","Año","Trimestre"]).reset_index(drop=True)
    panel["Var_Empleo_pct"]=(panel.groupby("Estado")["Empleo_Manufacturero"].pct_change()*100).round(2)
    panel["Periodo"]=panel["Año"].astype(str)+" Q"+panel["Trimestre"].astype(str)
    for col in ["ActInd","Exportaciones"]:
        panel[col]=panel.groupby("Estado")[col].transform(lambda x:x.interpolate(limit_direction="both"))
    print(f"   ✓ {len(panel)} observaciones")
    return panel


def cargar_geojson():
    for url in GEOJSON_URLS:
        try:
            r=requests.get(url,timeout=10)
            if r.status_code==200:
                print(f"   ✓ GeoJSON: {url[:55]}")
                return r.json()
        except Exception:
            continue
    print("   ⚠ GeoJSON externo no disponible, usando polígonos embebidos.")
    return {
        "type":"FeatureCollection",
        "features":[
            {"type":"Feature","id":"Aguascalientes","properties":{"name":"Aguascalientes"},
             "geometry":{"type":"Polygon","coordinates":[[[-102.31,21.63],[-102.05,21.63],[-101.92,21.82],[-101.90,22.08],[-102.08,22.22],[-102.31,22.17],[-102.50,22.01],[-102.55,21.78],[-102.31,21.63]]]}},
            {"type":"Feature","id":"Guanajuato","properties":{"name":"Guanajuato"},
             "geometry":{"type":"Polygon","coordinates":[[[-102.55,19.92],[-101.20,19.85],[-100.80,20.10],[-100.02,20.55],[-99.88,21.15],[-100.15,21.55],[-100.65,21.75],[-101.40,21.85],[-102.10,21.62],[-102.55,21.32],[-102.85,21.00],[-102.55,19.92]]]}},
            {"type":"Feature","id":"Jalisco","properties":{"name":"Jalisco"},
             "geometry":{"type":"Polygon","coordinates":[[[-105.42,19.05],[-104.70,18.70],[-103.50,18.85],[-103.00,19.00],[-102.55,19.92],[-102.85,21.00],[-102.55,21.32],[-102.10,21.62],[-101.55,21.90],[-101.60,22.42],[-102.30,22.72],[-103.00,22.45],[-103.80,22.00],[-104.70,21.00],[-105.42,20.20],[-105.42,19.05]]]}},
            {"type":"Feature","id":"Querétaro","properties":{"name":"Querétaro"},
             "geometry":{"type":"Polygon","coordinates":[[[-100.02,20.55],[-99.48,20.30],[-99.05,20.50],[-98.90,20.85],[-99.00,21.15],[-99.30,21.52],[-99.88,21.55],[-100.15,21.55],[-99.88,21.15],[-100.02,20.55]]]}},
            {"type":"Feature","id":"San Luis Potosí","properties":{"name":"San Luis Potosí"},
             "geometry":{"type":"Polygon","coordinates":[[[-102.55,21.32],[-101.55,21.90],[-101.20,21.85],[-100.65,21.75],[-100.15,21.55],[-99.88,21.55],[-99.30,21.52],[-99.05,21.95],[-98.80,22.55],[-99.05,23.65],[-99.65,24.00],[-100.80,24.00],[-101.60,23.45],[-102.10,22.85],[-102.55,22.17],[-102.55,21.32]]]}},
        ]
    }

# ══════════════════════════════════════════════
# CACHÉ EN MEMORIA — UN SOLO FETCH POR DÍA
# ══════════════════════════════════════════════
# La primera llamada del día descarga los datos.
# Todos los demás visitantes reciben el mismo
# DataFrame ya guardado en RAM (sin tocar la API).
# Al día siguiente Render reinicia y refresca.
import threading
from datetime import date

_cache = {
    "panel":   None,
    "geojson": None,
    "fecha":   None,
    "lock":    threading.Lock(),
}

def _cargar_datos_frescos():
    print("\n" + "═"*40)
    print("  CARGANDO DATOS — PANEL BAJÍO")
    print("═"*40)
    df_emp  = procesar_empleo()
    df_ied_ = procesar_ied()
    df_act  = procesar_actind()
    df_exp  = procesar_exportaciones()
    panel   = construir_panel(df_emp, df_ied_, df_act, df_exp)
    geojson = cargar_geojson()
    print("═"*40+"\n")
    return panel, geojson

def get_datos():
    """
    Devuelve (PANEL, GEOJSON) desde caché.
    Solo llama a las APIs si el caché está vacío
    o es de un día anterior. El Lock garantiza que
    aunque 100 personas abran el dashboard al mismo
    tiempo, la descarga ocurre UNA SOLA VEZ.
    """
    hoy = date.today()
    if _cache["panel"] is not None and _cache["fecha"] == hoy:
        return _cache["panel"], _cache["geojson"]
    with _cache["lock"]:
        if _cache["panel"] is not None and _cache["fecha"] == hoy:
            return _cache["panel"], _cache["geojson"]
        print(f"🔄 Descargando datos frescos para {hoy}...")
        panel, geojson = _cargar_datos_frescos()
        _cache["panel"]   = panel
        _cache["geojson"] = geojson
        _cache["fecha"]   = hoy
        print(f"✅ Caché listo — {len(panel)} observaciones en memoria.\n")
    return _cache["panel"], _cache["geojson"]

# Precarga al arrancar: no espera al primer visitante
PANEL, GEOJSON = get_datos()

AÑOS_DISPONIBLES = sorted(PANEL["Año"].unique())

VARS_DEF = [
    ("empleo","Empleo mfr."),
    ("ied","IED"),
    ("actind","Act. Industrial"),
    ("exportaciones","Exportaciones"),
]
VAR_COL = {
    "empleo":"Empleo_Manufacturero",
    "ied":"IED",
    "actind":"ActInd",
    "exportaciones":"Exportaciones",
}
VAR_LABEL = {
    "empleo":"Empleo manufacturero (personas)",
    "ied":"IED (millones USD)",
    "actind":"Índice de Act. Industrial (base 2013=100)",
    "exportaciones":"Exportaciones manufactureras (M USD)",
}

# ══════════════════════════════════════════════
# MÓDULO ECONOMÉTRICO
# ══════════════════════════════════════════════
def calcular_econometria(df: pd.DataFrame):
    cols=["Empleo_Manufacturero","IED","ActInd","Exportaciones"]
    sub=df[["Estado","Año","Trimestre"]+cols].dropna()
    if len(sub)<20: return None
    sub=sub.copy()
    sub["t"]=(sub["Año"]-sub["Año"].min())*4+sub["Trimestre"]
    sub=sub.set_index(["Estado","t"])
    Y=sub["Empleo_Manufacturero"]
    X=sub[["IED","ActInd","Exportaciones"]]
    try:
        mod=PanelOLS(Y,X,entity_effects=True,time_effects=True)
        res=mod.fit(cov_type="clustered",cluster_entity=True)
        dw_stat=durbin_watson(res.resids.values)
    except Exception as e:
        print(f"   [Eco] Error FE: {e}"); return None
    X_vif=sub[["IED","ActInd","Exportaciones"]].reset_index(drop=True).dropna()
    vifs={}
    try:
        for i,c in enumerate(X_vif.columns):
            vifs[c]=round(float(variance_inflation_factor(X_vif.values.astype(float),i)),2)
    except Exception:
        vifs={c:float("nan") for c in X_vif.columns}
    return {
        "coefs":res.params,"pvals":res.pvalues,
        "r2_within":round(res.rsquared,4),"n_obs":int(res.nobs),
        "dw":round(dw_stat,3),"vifs":vifs,
        "corr":sub[cols].corr().round(3),
    }


# ══════════════════════════════════════════════
# FIGURAS
# ══════════════════════════════════════════════
def fig_series(df,variable,estados):
    col=VAR_COL.get(variable,"Empleo_Manufacturero")
    fig=go.Figure()
    for est in estados:
        sub=df[df["Estado"]==est].sort_values(["Año","Trimestre"])
        fig.add_trace(go.Scatter(x=sub["Periodo"],y=sub[col],name=est,mode="lines+markers",
            line=dict(color=COLORES_ESTADOS[est],width=2),marker=dict(size=4),
            hovertemplate=f"<b>{est}</b><br>%{{x}}<br>{VAR_LABEL[variable]}: %{{y:,.1f}}<extra></extra>"))
    fig.update_layout(**{**PLOT_LAYOUT,"yaxis":dict(title=VAR_LABEL[variable],gridcolor="#EEE",tickformat=","),
        "xaxis":dict(tickangle=-45,tickfont_size=9),"height":300})
    return fig


def fig_mapa(df,variable,estados,yr_to,geojson):
    col=VAR_COL.get(variable,"Empleo_Manufacturero")
    sub=df[(df["Estado"].isin(estados))&(df["Año"]==yr_to)]
    grp=sub.groupby("Estado")[col].mean().reset_index().rename(columns={col:"valor"})
    all_e=pd.DataFrame({"Estado":list(ESTADOS_BAJIO.keys())})
    grp=all_e.merge(grp,on="Estado",how="left")
    fig=go.Figure(go.Choropleth(
        geojson=geojson,locations=grp["Estado"],z=grp["valor"],
        featureidkey="properties.name",
        colorscale=[[0,"#B5D4F4"],[0.5,"#1A5599"],[1,"#0C3460"]],
        marker_line_color="white",marker_line_width=1.5,
        colorbar=dict(title=dict(text=VAR_LABEL[variable],font_size=10),thickness=12,len=0.7),
        hovertemplate="<b>%{location}</b><br>"+VAR_LABEL[variable]+": %{z:,.1f}<extra></extra>",
    ))
    fig.update_geos(fitbounds="locations",visible=False,showland=True,landcolor="#F5F3EF",showframe=False)
    fig.update_layout(**{**PLOT_LAYOUT,"height":300,"margin":dict(l=0,r=0,t=10,b=0)})
    return fig


def fig_scatter(df,variable,estados):
    col=VAR_COL.get(variable,"IED")
    fig=go.Figure()
    for est in estados:
        sub=df[df["Estado"]==est]
        fig.add_trace(go.Scatter(x=sub["Empleo_Manufacturero"],y=sub[col],mode="markers",name=est,
            marker=dict(color=COLORES_ESTADOS[est],size=6,opacity=0.8),text=sub["Periodo"],
            hovertemplate=f"<b>{est}</b><br>%{{text}}<br>Empleo: %{{x:,}}<br>{VAR_LABEL[variable]}: %{{y:,.1f}}<extra></extra>"))
    fig.update_layout(**{**PLOT_LAYOUT,"xaxis":dict(title="Empleo manufacturero",gridcolor="#EEE",tickformat=","),
        "yaxis":dict(title=VAR_LABEL[variable],gridcolor="#EEE"),"height":300})
    return fig


def fig_heatmap(df,variable,estados):
    col=VAR_COL.get(variable,"Empleo_Manufacturero")
    sub=df[df["Estado"].isin(estados)]
    piv=sub.pivot_table(index="Estado",columns="Trimestre",values=col,aggfunc="mean").round(1)
    piv.columns=[f"Q{c}" for c in piv.columns]
    fig=go.Figure(go.Heatmap(z=piv.values,x=piv.columns.tolist(),y=piv.index.tolist(),
        colorscale=[[0,"#E6F1FB"],[0.5,"#1A5599"],[1,"#0C3460"]],
        text=np.round(piv.values,0),texttemplate="%{text:,.0f}",textfont_size=10))
    fig.update_layout(**{**PLOT_LAYOUT,"height":230,"margin":dict(l=130,r=10,t=10,b=10)})
    return fig


def fig_correlacion(corr_df):
    labels=["Empleo","IED","Act.Ind","Exportac."]
    z=corr_df.values
    fig=go.Figure(go.Heatmap(z=z,x=labels,y=labels,
        colorscale=[[0,"#FFF0E0"],[0.5,"#8B5E3C"],[1,"#0C3460"]],zmin=-1,zmax=1,
        text=np.round(z,2),texttemplate="%{text}",textfont_size=12))
    fig.update_layout(**{**PLOT_LAYOUT,"height":280,"margin":dict(l=70,r=20,t=20,b=70),
        "xaxis":dict(tickfont_size=11),"yaxis":dict(tickfont_size=11)})
    return fig

# ══════════════════════════════════════════════
# ESTILOS
# ══════════════════════════════════════════════
STYLE_CARD={"background":CARD_BG,"borderRadius":"10px","border":"1px solid #E5E0D8","padding":"16px 20px","marginBottom":"14px"}
STYLE_METRIC={"background":"#F0EDE8","borderRadius":"8px","padding":"12px 16px","flex":"1","minWidth":"110px"}
STYLE_BTN_BASE={"fontSize":"12px","padding":"5px 13px","borderRadius":"20px","border":"1.5px solid #CCC","background":"white","color":"#555","cursor":"pointer","marginRight":"6px","marginBottom":"6px","fontFamily":"inherit"}
STYLE_BTN_VAR_ACTIVE={**STYLE_BTN_BASE,"background":CAFE,"color":"#FAC775","border":f"1.5px solid {CAFE}","borderRadius":"6px"}
STYLE_SEC_HDR={"fontSize":"14px","fontWeight":"500","color":AZUL_OSCURO,"margin":"0 0 4px","borderLeft":f"3px solid {NARANJA}","paddingLeft":"10px"}
STYLE_SUB={"fontSize":"11px","color":TEXT_SEC,"margin":"4px 0 8px 14px"}

# ══════════════════════════════════════════════
# APP LAYOUT
# ══════════════════════════════════════════════
app=dash.Dash(__name__,title="Panel Bajío · Econométrico",
    meta_tags=[{"name":"viewport","content":"width=device-width, initial-scale=1"}])
server=app.server

app.layout=html.Div(
    style={"fontFamily":"'Helvetica Neue',Arial,sans-serif","background":BG,"minHeight":"100vh","padding":"24px 20px","maxWidth":"1400px","margin":"0 auto"},
    children=[

    html.Div(style={"marginBottom":"20px"},children=[
        html.H1("Panel Econométrico — Región del Bajío",style={"fontSize":"22px","fontWeight":"500","margin":"0 0 4px","color":TEXT_PRIM}),
        html.P("Empleo · IED · Actividad Industrial · Exportaciones  ·  2015–2025  ·  Datos: INEGI + SE",style={"fontSize":"13px","color":TEXT_SEC,"margin":"0"}),
    ]),

    html.Div(id="metrics-row",style={"display":"flex","gap":"10px","flexWrap":"wrap","marginBottom":"18px"}),

    html.Div(style={**STYLE_CARD,"marginBottom":"16px"},children=[
        html.Div(style={"display":"flex","flexWrap":"wrap","gap":"20px","alignItems":"flex-start"},children=[
            html.Div([
                html.P("Estados",style={"fontSize":"11px","color":TEXT_SEC,"textTransform":"uppercase","letterSpacing":"0.05em","margin":"0 0 6px"}),
                html.Div(id="estado-btns",style={"display":"flex","flexWrap":"wrap"}),
            ]),
            html.Div([
                html.P("Período",style={"fontSize":"11px","color":TEXT_SEC,"textTransform":"uppercase","letterSpacing":"0.05em","margin":"0 0 6px"}),
                html.Div(style={"display":"flex","alignItems":"center","gap":"8px"},children=[
                    dcc.Dropdown(id="year-from",options=[{"label":str(y),"value":y} for y in AÑOS_DISPONIBLES],value=2015,clearable=False,style={"width":"90px","fontSize":"13px"}),
                    html.Span("—",style={"color":TEXT_SEC}),
                    dcc.Dropdown(id="year-to",options=[{"label":str(y),"value":y} for y in AÑOS_DISPONIBLES],value=AÑOS_DISPONIBLES[-1],clearable=False,style={"width":"90px","fontSize":"13px"}),
                ]),
            ]),
            html.Div([
                html.P("Variable",style={"fontSize":"11px","color":TEXT_SEC,"textTransform":"uppercase","letterSpacing":"0.05em","margin":"0 0 6px"}),
                html.Div(id="var-btns",style={"display":"flex","gap":"6px","flexWrap":"wrap"}),
            ]),
        ]),
    ]),

    dcc.Store(id="active-estados",data=list(ESTADOS_BAJIO.keys())),
    dcc.Store(id="active-var",data="empleo"),

    # Serie de tiempo
    html.Div(style=STYLE_CARD,children=[
        html.P("Serie de tiempo trimestral",style=STYLE_SEC_HDR),
        html.P(id="series-sub",style=STYLE_SUB),
        dcc.Graph(id="series-chart",config={"displayModeBar":False}),
    ]),

    # Mapa + Scatter
    html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"14px","marginBottom":"14px"},children=[
        html.Div(style=STYLE_CARD,children=[
            html.P("Mapa del Bajío",style=STYLE_SEC_HDR),
            html.P(id="map-sub",style=STYLE_SUB),
            dcc.Graph(id="map-chart",config={"displayModeBar":False}),
        ]),
        html.Div(style=STYLE_CARD,children=[
            html.P("Empleo vs Variable seleccionada",style=STYLE_SEC_HDR),
            html.P("Dispersión por trimestre y estado",style=STYLE_SUB),
            dcc.Graph(id="scatter-chart",config={"displayModeBar":False}),
        ]),
    ]),

    # Heatmap estacional
    html.Div(style=STYLE_CARD,children=[
        html.P("Patrón estacional por trimestre",style=STYLE_SEC_HDR),
        html.P("Promedio del período seleccionado",style=STYLE_SUB),
        dcc.Graph(id="heatmap-chart",config={"displayModeBar":False}),
    ]),

    # Tabla del panel
    html.Div(style=STYLE_CARD,children=[
        html.P("Tabla del panel",style=STYLE_SEC_HDR),
        html.P(id="tabla-sub",style=STYLE_SUB),
        html.Div(id="tabla-panel",style={"overflowX":"auto"}),
    ]),

    # ──────────────────────────────────────────
    # MÓDULO ECONOMÉTRICO
    # ──────────────────────────────────────────
    html.Div(style={"margin":"28px 0 10px"},children=[
        html.H2("Módulo Econométrico",style={"fontSize":"18px","fontWeight":"500","color":AZUL_OSCURO,"margin":"0 0 4px"}),
        html.P("Modelo de Datos Panel · Efectos Fijos de entidad y tiempo · Errores estándar clusterizados",style={"fontSize":"12px","color":TEXT_SEC,"margin":"0"}),
    ]),

    html.Div(style={**STYLE_CARD,"background":"#EEF4FB","border":f"1px solid {AZUL_CLARO}44"},children=[
        html.P("Modelo estimado:",style={"fontSize":"12px","color":TEXT_SEC,"margin":"0 0 6px"}),
        html.P("Empleo_it = β₀ + β₁·IED_it + β₂·ActInd_it + β₃·Exportaciones_it + μᵢ + λₜ + εᵢₜ",
               style={"fontSize":"14px","fontWeight":"500","color":AZUL_OSCURO,"fontFamily":"monospace","margin":"0 0 4px"}),
        html.P("μᵢ = efectos fijos por entidad   ·   λₜ = efectos fijos temporales   ·   Var. dep: miles de personas",
               style={"fontSize":"11px","color":TEXT_SEC,"margin":"0"}),
    ]),

    html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"14px","marginBottom":"14px"},children=[
        html.Div(style=STYLE_CARD,children=[
            html.P("Coeficientes — Efectos Fijos",style=STYLE_SEC_HDR),
            html.P("*** p<0.01  ** p<0.05  * p<0.1  · Errores clusterizados por entidad",style=STYLE_SUB),
            html.Div(id="tabla-regresion"),
        ]),
        html.Div(style=STYLE_CARD,children=[
            html.P("Pruebas de robustez",style=STYLE_SEC_HDR),
            html.P("Durbin-Watson y VIF (Factor de Inflación de Varianza)",style=STYLE_SUB),
            html.Div(id="robustez-panel"),
        ]),
    ]),

    html.Div(style=STYLE_CARD,children=[
        html.P("Matriz de correlación de Pearson",style=STYLE_SEC_HDR),
        html.P("Entre las 4 variables del modelo para el período seleccionado",style=STYLE_SUB),
        dcc.Graph(id="corr-chart",config={"displayModeBar":False}),
    ]),

    html.P("Fuentes: INEGI (EMIM · BIE-BISE) · Secretaría de Economía · Modelo: PanelOLS (linearmodels 7.0) · Errores clusterizados por entidad",
           style={"fontSize":"11px","color":TEXT_SEC,"textAlign":"center","marginTop":"10px"}),
])

# ══════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════
@app.callback(
    Output("estado-btns","children"),
    Output("active-estados","data"),
    Input({"type":"btn-estado","index":dash.ALL},"n_clicks"),
    State("active-estados","data"),
    prevent_initial_call=False,
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
            style={**STYLE_BTN_BASE,"background":col if ia else "white","color":"#E6F1FB" if ia else "#555","borderColor":col if ia else "#CCC"}))
    return btns,active


@app.callback(
    Output("var-btns","children"),
    Output("active-var","data"),
    Input({"type":"btn-var","index":dash.ALL},"n_clicks"),
    State("active-var","data"),
    prevent_initial_call=False,
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
            style=STYLE_BTN_VAR_ACTIVE if ia else {**STYLE_BTN_BASE,"borderRadius":"6px"}))
    return btns,av


@app.callback(
    Output("metrics-row","children"),
    Output("series-chart","figure"),
    Output("map-chart","figure"),
    Output("scatter-chart","figure"),
    Output("heatmap-chart","figure"),
    Output("tabla-panel","children"),
    Output("tabla-regresion","children"),
    Output("robustez-panel","children"),
    Output("corr-chart","figure"),
    Output("series-sub","children"),
    Output("map-sub","children"),
    Output("tabla-sub","children"),
    Input("active-estados","data"),
    Input("active-var","data"),
    Input("year-from","value"),
    Input("year-to","value"),
)
def update_all(estados,variable,yr_from,yr_to):
    # Siempre desde caché — sin tocar APIs
    panel, geojson = get_datos()
    df=panel[(panel["Estado"].isin(estados))&(panel["Año"]>=yr_from)&(panel["Año"]<=yr_to)].copy()
    n_obs=len(df)

    emp_p=int(df["Empleo_Manufacturero"].mean()) if n_obs else 0
    ied_p=round(df["IED"].mean(),1) if n_obs else 0
    act_p=round(df["ActInd"].mean(),1) if n_obs else 0
    exp_p=round(df["Exportaciones"].mean(),1) if n_obs else 0
    metrics=[
        ("Observaciones",f"{n_obs:,}","trim. × estados"),
        ("Empleo prom.",f"{emp_p:,}","personas / trim."),
        ("IED prom.",f"${ied_p}M","USD / trim."),
        ("Act. Industrial",f"{act_p}","índice 2013=100"),
        ("Exportaciones",f"${exp_p}M","USD prom. / trim."),
    ]
    metric_cards=[html.Div(style=STYLE_METRIC,children=[
        html.P(l,style={"fontSize":"10px","color":TEXT_SEC,"textTransform":"uppercase","letterSpacing":"0.05em","margin":"0 0 3px"}),
        html.P(v,style={"fontSize":"18px","fontWeight":"500","color":TEXT_PRIM,"margin":"0","lineHeight":"1.1"}),
        html.P(s,style={"fontSize":"10px","color":TEXT_SEC,"margin":"2px 0 0"}),
    ]) for l,v,s in metrics]

    f_series=fig_series(df,variable,estados)
    f_mapa=fig_mapa(df,variable,estados,yr_to,geojson)
    f_scat=fig_scatter(df,variable,estados)
    f_heat=fig_heatmap(df,variable,estados)

    # Tabla
    show=df.sort_values(["Año","Trimestre","Estado"],ascending=[False,False,True]).head(60)
    ths=["Estado","Año","Trim.","Empleo mfr.","IED (M USD)","Act.Ind","Exportac.(M USD)","Var.Emp %"]
    header=html.Tr([html.Th(c,style={"fontWeight":"500","fontSize":"10px","color":TEXT_SEC,"textAlign":"left","padding":"5px 8px","borderBottom":"1px solid #DDD"}) for c in ths])
    rows_h=[]
    for _,r in show.iterrows():
        c=COLORES_ESTADOS.get(r["Estado"],"#888")
        ve=r.get("Var_Empleo_pct",float("nan"))
        ve_str=(f"+{ve:.1f}%" if pd.notna(ve) and ve>=0 else f"{ve:.1f}%") if pd.notna(ve) else "—"
        ve_col="#0F6E56" if pd.notna(ve) and ve>=0 else "#993C1D"
        rows_h.append(html.Tr([
            html.Td(html.Span(r["Estado"],style={"background":c+"22","color":c,"border":f"0.5px solid {c}55","fontSize":"10px","padding":"2px 7px","borderRadius":"10px"}),style={"padding":"4px 8px"}),
            html.Td(str(int(r["Año"])),style={"padding":"4px 8px","fontSize":"11px"}),
            html.Td(f"Q{int(r['Trimestre'])}",style={"padding":"4px 8px","fontSize":"11px"}),
            html.Td(f"{int(r['Empleo_Manufacturero']):,}",style={"padding":"4px 8px","fontSize":"11px"}),
            html.Td(f"${r['IED']:.1f}M",style={"padding":"4px 8px","fontSize":"11px"}),
            html.Td(f"{r['ActInd']:.1f}" if pd.notna(r.get("ActInd")) else "—",style={"padding":"4px 8px","fontSize":"11px"}),
            html.Td(f"${r['Exportaciones']:.1f}M" if pd.notna(r.get("Exportaciones")) else "—",style={"padding":"4px 8px","fontSize":"11px"}),
            html.Td(ve_str,style={"padding":"4px 8px","fontSize":"11px","color":ve_col,"fontWeight":"500"}),
        ],style={"borderBottom":"0.5px solid #EEE"}))
    tabla=html.Table([html.Thead(header),html.Tbody(rows_h)],style={"width":"100%","borderCollapse":"collapse"})

    # Econometría
    eco=calcular_econometria(df)
    no_data=html.P("Amplía el período o selecciona más estados para estimar el modelo.",style={"fontSize":"12px","color":TEXT_SEC,"padding":"10px"})
    if eco is None:
        return (metric_cards,f_series,f_mapa,f_scat,f_heat,tabla,no_data,no_data,go.Figure(),
                VAR_LABEL[variable],f"Promedio {yr_to} · {VAR_LABEL[variable]}",f"{n_obs} obs · últimas 60")

    # Tabla regresión
    var_names={"IED":"IED (M USD)","ActInd":"Actividad Industrial","Exportaciones":"Exportaciones (M USD)"}
    hdr_r=html.Tr([html.Th(c,style={"fontWeight":"500","fontSize":"10px","color":TEXT_SEC,"padding":"5px 10px","borderBottom":"1px solid #DDD","textAlign":al}) for c,al in [("Variable","left"),("Coeficiente","right"),("P-value","right")]])
    filas_r=[]
    for vk,vn in var_names.items():
        coef=eco["coefs"].get(vk,float("nan")); pval=eco["pvals"].get(vk,float("nan"))
        stars="***" if pd.notna(pval) and pval<0.01 else ("**" if pd.notna(pval) and pval<0.05 else ("*" if pd.notna(pval) and pval<0.1 else ""))
        pcol="#0F6E56" if pd.notna(pval) and pval<0.05 else TEXT_SEC
        filas_r.append(html.Tr([
            html.Td(vn,style={"padding":"6px 10px","fontSize":"12px","fontWeight":"500"}),
            html.Td(f"{coef:,.2f}{stars}",style={"padding":"6px 10px","fontSize":"12px","textAlign":"right"}),
            html.Td(f"{pval:.4f}",style={"padding":"6px 10px","fontSize":"12px","textAlign":"right","color":pcol,"fontWeight":"500"}),
        ]))
    summary=[
        html.Tr([html.Td("R² within",style={"padding":"6px 10px","fontSize":"12px","color":TEXT_SEC}),html.Td("",style={"padding":"6px 10px"}),html.Td(f"{eco['r2_within']:.4f}",style={"padding":"6px 10px","fontSize":"12px","textAlign":"right","fontWeight":"500"})]),
        html.Tr([html.Td("N observaciones",style={"padding":"6px 10px","fontSize":"12px","color":TEXT_SEC}),html.Td("",style={"padding":"6px 10px"}),html.Td(str(eco["n_obs"]),style={"padding":"6px 10px","fontSize":"12px","textAlign":"right"})]),
        html.Tr([html.Td("Efectos entidad",style={"padding":"6px 10px","fontSize":"12px","color":TEXT_SEC}),html.Td("",style={"padding":"6px 10px"}),html.Td("✓ Sí",style={"padding":"6px 10px","fontSize":"12px","textAlign":"right","color":"#0F6E56"})]),
        html.Tr([html.Td("Efectos tiempo",style={"padding":"6px 10px","fontSize":"12px","color":TEXT_SEC}),html.Td("",style={"padding":"6px 10px"}),html.Td("✓ Sí",style={"padding":"6px 10px","fontSize":"12px","textAlign":"right","color":"#0F6E56"})]),
    ]
    tabla_reg=html.Table([html.Thead(hdr_r),html.Tbody(filas_r+summary)],style={"width":"100%","borderCollapse":"collapse"})

    # Robustez
    dw=eco["dw"]
    dw_txt="Sin autocorrelación" if 1.5<dw<2.5 else ("Autocorrelación positiva posible" if dw<1.5 else "Autocorrelación negativa posible")
    dw_col="#0F6E56" if 1.5<dw<2.5 else NARANJA
    vif_items=[]
    for vn,vv in eco["vifs"].items():
        vc="#993C1D" if vv>10 else ("#D97B2A" if vv>5 else "#0F6E56")
        vl="Alta multicolinealidad" if vv>10 else ("Moderada" if vv>5 else "OK ✓")
        vif_items.append(html.Div(style={"display":"flex","justifyContent":"space-between","alignItems":"center","padding":"6px 10px","borderBottom":"0.5px solid #EEE"},children=[
            html.Span(vn,style={"fontSize":"12px"}),
            html.Span(f"VIF = {vv:.2f}  ({vl})",style={"fontSize":"11px","color":vc,"fontWeight":"500"}),
        ]))
    robustez=html.Div([
        html.P("Durbin-Watson",style={"fontSize":"11px","color":TEXT_SEC,"margin":"0 0 4px","padding":"0 10px"}),
        html.Div(style={"display":"flex","alignItems":"center","gap":"10px","padding":"6px 10px","marginBottom":"12px","borderBottom":"1px solid #EEE"},children=[
            html.Span(f"{dw:.3f}",style={"fontSize":"22px","fontWeight":"500","color":dw_col}),
            html.Span(dw_txt,style={"fontSize":"11px","color":dw_col}),
        ]),
        html.P("VIF — Factor de inflación de varianza",style={"fontSize":"11px","color":TEXT_SEC,"margin":"0 0 4px","padding":"0 10px"}),
        html.Div(vif_items),
    ])

    f_corr=fig_correlacion(eco["corr"])

    return (metric_cards,f_series,f_mapa,f_scat,f_heat,tabla,tabla_reg,robustez,f_corr,
            VAR_LABEL[variable],f"Intensidad promedio {yr_to} · {VAR_LABEL[variable]}",f"{n_obs} obs · últimas 60")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
if __name__ == "__main__":
    port=int(os.environ.get("PORT",8050))
    app.run(debug=False,host="0.0.0.0",port=port)
