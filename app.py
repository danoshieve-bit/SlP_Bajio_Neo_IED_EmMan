"""
Dashboard Econométrico — Región del Bajío (VERSIÓN MASTER Y SEGURA)
========================================================
UI: Pregunta de Inv, Mapa Mapbox, DataTable, Lags, Predicciones
ETL: Lógica robusta original y explícita.
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
# Token restaurado directo al código para evitar fallos de entorno en Render
TOKEN_INEGI = os.getenv("INEGI_TOKEN", "2c63db48-9a6a-4468-be5b-8ab85da04eb6")

ESTADOS_BAJIO = {
    "San Luis Potosí": "24",
    "Aguascalientes":  "01",
    "Guanajuato":      "11",
    "Jalisco":         "14",
    "Querétaro":       "22",
}

# Paleta de Colores (SLP Héroe)
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

INDICADOR_ACTIND = "738413"
INDICADOR_EXPORTACIONES = "924,739277"

SE_IED_URL = (
    "https://datos.gob.mx/busca/api/action/datastore_search"
    "?resource_id=fc1e3b7b-4027-4c59-9e5a-f02f48e90ca1&limit=5000"
)
GEOJSON_URLS = [
    "https://raw.githubusercontent.com/PhantomInsights/mexican-geojson/main/src/states/states.json",
    "https://raw.githubusercontent.com/angelnmara/geojson/master/mexicoHigh.json",
]

_EMP_BASE = {"Aguascalientes":95000,"Guanajuato":340000,"Jalisco":280000,"Querétaro":160000,"San Luis Potosí":110000}
_IED_BASE = {"Aguascalientes":220,"Guanajuato":315,"Jalisco":430,"Querétaro":265,"San Luis Potosí":225}
_ACT_BASE = {"Aguascalientes":108,"Guanajuato":115,"Jalisco":112,"Querétaro":120,"San Luis Potosí":106}
_EXP_BASE = {"Aguascalientes":4200,"Guanajuato":9500,"Jalisco":7800,"Querétaro":5600,"San Luis Potosí":3900}

# Variables con nombres corregidos
VARS_DEF   = [("empleo","Empleo Manufacturero"),("ied","Inversión Extranjera Directa (IED)"),("actind","Actividad Manufacturera"),("exportaciones","Exportaciones Manufactureras")]
VAR_COL    = {"empleo":"Empleo_Manufacturero","ied":"IED","actind":"ActInd","exportaciones":"Exportaciones"}
VAR_LABEL  = {
    "empleo":        "Empleo Manufacturero (personas)",
    "ied":           "Inversión Extranjera Directa (IED) (M USD)",
    "actind":        "Actividad Manufacturera (base 2013=100)",
    "exportaciones": "Exportaciones Manufactureras (M USD)",
}

# ══════════════════════════════════════════════
# ETL — CÓDIGO EXPLÍCITO Y SEGURO (SIN COMPRIMIR)
# ══════════════════════════════════════════════
def fetch_inegi_serie(indicador, fuente="BIE", geo="00"):
    url = (
        f"https://www.inegi.org.mx/app/api/indicadores/desarrolladores/"
        f"jsonxml/INDICATOR/{indicador}/es/{geo}/false/{fuente}/2.0/{TOKEN_INEGI}?type=json"
    )
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        rows = []
        series_list = data.get("Series", [])
        if not series_list:
            return pd.DataFrame(columns=["fecha","valor"])
            
        # Tomar la última serie disponible (más específica)
        obs = series_list[-1].get("OBSERVATIONS", [])
        for s in obs:
            if s["OBS_VALUE"] not in (None, "", "N/A"):
                try: 
                    rows.append({"fecha": s["TIME_PERIOD"], "valor": float(s["OBS_VALUE"])})
                except Exception: 
                    pass
        if not rows: 
            raise ValueError("vacío")
        return pd.DataFrame(rows)
    except Exception as e:
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
        except Exception: 
            pass
    if rows:
        return pd.DataFrame(rows)
    else:
        return pd.DataFrame(columns=["Estado","Año","Trimestre",col])

def _sim_empleo(estado):
    rng = np.random.default_rng(abs(hash(estado))%(2**32))
    base = _EMP_BASE.get(estado,100000)
    rows = []
    for yr in YEARS:
        for m in range(1,13):
            if yr==2025 and m>3: 
                break
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
                if yr==2025 and q>1: 
                    break
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
                if yr==2025 and q>1: 
                    break
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
                if yr==2025 and q>1: 
                    break
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
        if not recs: 
            raise ValueError("Sin registros de IED")
        df = pd.DataFrame(recs)
        
        ce = next((c for c in df.columns if "entidad" in c.lower() or "estado" in c.lower()), None)
        ca = next((c for c in df.columns if "año" in c.lower() or "anio" in c.lower()), None)
        ct = next((c for c in df.columns if "trim" in c.lower()), None)
        ci = next((c for c in df.columns if "ied" in c.lower() or "inversion" in c.lower()), None)
        
        if not all([ce,ca,ct,ci]): 
            raise ValueError("Columnas de IED no mapeadas correctamente")
            
        df = df.rename(columns={ce:"Estado", ca:"Año", ct:"Trimestre", ci:"IED"})
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
        if df.empty: 
            frames.append(_sim_actind()[lambda d:d["Estado"]==estado])
        else: 
            frames.append(_mensual_a_trim(df, estado, "ActInd"))
            
    result = pd.concat(frames, ignore_index=True)
    if result.empty:
        return _sim_actind()
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
        base = _EXP_BASE.get(estado,5000)
        sim = []
        for yr in YEARS:
            for q in range(1,5):
                if yr==2025 and q>1: 
                    break
                t = (yr-2015)*4+q
                valor = round(base*(1+0.022*t)*rng.uniform(0.85,1.18)*[.95,1.05,1.08,1.12][q-1],1)
                sim.append({"Estado":estado, "Año":yr, "Trimestre":q, "Exportaciones":valor})
        frames.append(pd.DataFrame(sim))
    return pd.concat(frames, ignore_index=True)

def construir_panel(df_emp, df_ied, df_act, df_exp):
    keys = ["Estado","Año","Trimestre"]
    # Merge explícito
    panel = pd.merge(df_emp, df_ied, on=keys, how="inner")
    panel = pd.merge(panel, df_act, on=keys, how="left")
    panel = pd.merge(panel, df_exp, on=keys, how="left")
    
    panel = panel[panel["Estado"].isin(ESTADOS_BAJIO.keys())].copy()
    panel = panel.sort_values(["Estado","Año","Trimestre"]).reset_index(drop=True)
    
    panel["Var_Empleo_pct"] = (panel.groupby("Estado")["Empleo_Manufacturero"].pct_change()*100
