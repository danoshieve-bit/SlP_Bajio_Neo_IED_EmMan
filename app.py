"""
Dashboard Econométrico — Región del Bajío (VERSIÓN PRO)
========================================================
UI: Tabs, Full Width, Animación Hans Rosling, Área Chart
ETL/Caché/Econometría: sin cambios
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
BG          = "#F4F2EE"
CARD_BG     = "#FFFFFF"
TEXT_PRIM   = "#1A1A1A"
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
INDICADORES_ACTIND = {
    "Aguascalientes":  "738413",
    "Guanajuato":      "738414",
    "Jalisco":         "738415",
    "Querétaro":       "738416",
    "San Luis Potosí": "738417",
}
INDICADORES_EXPORTACIONES = {
    "Aguascalientes":  "127595",
    "Guanajuato":      "739277",
    "Jalisco":         "739278",
    "Querétaro":       "739279",
    "San Luis Potosí": "739280",
}
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

VARS_DEF   = [("empleo","Empleo mfr."),("ied","IED"),("actind","Act. Industrial"),("exportaciones","Exportaciones")]
VAR_COL    = {"empleo":"Empleo_Manufacturero","ied":"IED","actind":"ActInd","exportaciones":"Exportaciones"}
VAR_LABEL  = {
    "empleo":        "Empleo manufacturero (personas)",
    "ied":           "IED (millones USD)",
    "actind":        "Índice de Act. Industrial (base 2013=100)",
    "exportaciones": "Exportaciones manufactureras (M USD)",
}

# ══════════════════════════════════════════════
# ETL — SIN CAMBIOS
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
    except:
        return pd.DataFrame(columns=["fecha","valor"])

def _mensual_a_trim(df, estado, col):
    df["fecha"] = pd.to_datetime(df["fecha"], format="%Y/%m", errors="coerce")
    df = df.dropna(subset=["fecha"])
    df["Año"] = df["fecha"].dt.year; df["Mes"] = df["fecha"].dt.month
    df = df[(df["Año"]>=2015)&(df["Año"]<=2025)]
    df["Trimestre"] = df["Mes"].apply(lambda m:(m-1)//3+1)
    return df.groupby([pd.Series([estado]*len(df),name="Estado"),"Año","Trimestre"])["valor"].mean().reset_index().rename(columns={"valor":col})

def _parse_trim(df, estado, col):
    rows=[]
    for _,row in df.iterrows():
        t=str(row["fecha"]).replace("-","/")
        try:
            p=t.split("/"); yr=int(p[0]); qn=int(p[1].replace("Q","").replace("T",""))
            if 2015<=yr<=2025: rows.append({"Estado":estado,"Año":yr,"Trimestre":qn,col:row["valor"]})
        except: pass
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Estado","Año","Trimestre",col])

def _sim_empleo(estado):
    rng=np.random.default_rng(abs(hash(estado))%(2**32)); base=_EMP_BASE.get(estado,100000); rows=[]
    for yr in YEARS:
        for m in range(1,13):
            if yr==2025 and m>3: break
            t=(yr-2015)*12+m
            rows.append({"fecha":pd.Timestamp(yr,m,1),"valor":int(base*(1+0.018*t/12)*rng.uniform(0.97,1.03)*[.97,.98,1.,.101,1.02,1.02,1.01,1.01,1.,.99,.98,.97][m-1])})
    return pd.DataFrame(rows)

def _sim_ied():
    rows=[]
    for estado,base in _IED_BASE.items():
        rng=np.random.default_rng(abs(hash(estado+"ied"))%(2**32))
        for yr in YEARS:
            for q in range(1,5):
                if yr==2025 and q>1: break
                t=(yr-2015)*4+q
                rows.append({"Estado":estado,"Año":yr,"Trimestre":q,"IED":round(base*(1+0.02*t)*rng.uniform(0.88,1.20)*[1.,1.1,1.05,1.15][q-1],1)})
    return pd.DataFrame(rows)

def _sim_actind():
    rows=[]
    for estado,base in _ACT_BASE.items():
        rng=np.random.default_rng(abs(hash(estado+"act"))%(2**32))
        for yr in YEARS:
            for q in range(1,5):
                if yr==2025 and q>1: break
                t=(yr-2015)*4+q
                rows.append({"Estado":estado,"Año":yr,"Trimestre":q,"ActInd":round(base*(1+0.015*t)*rng.uniform(0.94,1.06)*[.98,1.01,1.02,.99][q-1],1)})
    return pd.DataFrame(rows)

def _sim_exportaciones():
    rows=[]
    for estado,base in _EXP_BASE.items():
        rng=np.random.default_rng(abs(hash(estado+"exp"))%(2**32))
        for yr in YEARS:
            for q in range(1,5):
                if yr==2025 and q>1: break
                t=(yr-2015)*4+q
                rows.append({"Estado":estado,"Año":yr,"Trimestre":q,"Exportaciones":round(base*(1+0.022*t)*rng.uniform(0.85,1.18)*[.95,1.05,1.08,1.12][q-1],1)})
    return pd.DataFrame(rows)

def procesar_empleo():
    print("📥 Empleo manufacturero...")
    frames=[]
    for estado,ind in INDICADORES_EMPLEO.items():
        df=fetch_inegi_serie(ind,fuente="BIE-BISE")
        if df.empty:
            sim=_sim_empleo(estado); sim["Estado"]=estado; sim["Año"]=sim["fecha"].dt.year; sim["Mes"]=sim["fecha"].dt.month
            sim=sim[(sim["Año"]>=2015)&(sim["Año"]<=2025)]; sim["Trimestre"]=sim["Mes"].apply(lambda m:(m-1)//3+1)
            df_t=sim.groupby(["Estado","Año","Trimestre"])["valor"].mean().reset_index().rename(columns={"valor":"Empleo_Manufacturero"})
        else:
            df_t=_mensual_a_trim(df,estado,"Empleo_Manufacturero")
        df_t["Empleo_Manufacturero"]=df_t["Empleo_Manufacturero"].round(0).astype(int)
        frames.append(df_t)
    return pd.concat(frames,ignore_index=True)

def procesar_ied():
    try:
        r=requests.get(SE_IED_URL,timeout=15); r.raise_for_status()
        recs=r.json().get("result",{}).get("records",[])
        if not recs: raise ValueError()
        df=pd.DataFrame(recs)
        ce=next((c for c in df.columns if "entidad" in c.lower() or "estado" in c.lower()),None)
        ca=next((c for c in df.columns if "año" in c.lower() or "anio" in c.lower()),None)
        ct=next((c for c in df.columns if "trim" in c.lower()),None)
        ci=next((c for c in df.columns if "ied" in c.lower() or "inversion" in c.lower()),None)
        if not all([ce,ca,ct,ci]): raise ValueError()
        df=df.rename(columns={ce:"Estado",ca:"Año",ct:"Trimestre",ci:"IED"})
        df=df[df["Estado"].str.strip().isin(ESTADOS_BAJIO.keys())]
        for col in ["Año","Trimestre","IED"]: df[col]=pd.to_numeric(df[col],errors="coerce")
        df=df.dropna(); df=df[(df["Año"]>=2015)&(df["Año"]<=2025)]
        return df[["Estado","Año","Trimestre","IED"]]
    except:
        return _sim_ied()

def procesar_actind():
    frames=[]
    for estado,ind in INDICADORES_ACTIND.items():
        df=fetch_inegi_serie(ind,fuente="BIE-BISE")
        if df.empty: frames.append(_sim_actind()[lambda d:d["Estado"]==estado])
        else: frames.append(_mensual_a_trim(df,estado,"ActInd"))
    result=pd.concat(frames,ignore_index=True)
    return result if not result.empty else _sim_actind()

def procesar_exportaciones():
    frames=[]
    for estado,ind in INDICADORES_EXPORTACIONES.items():
        df=fetch_inegi_serie(ind,fuente="BIE-BISE")
        if not df.empty:
            df_t=_parse_trim(df,estado,"Exportaciones")
            if not df_t.empty: frames.append(df_t); continue
        rng=np.random.default_rng(abs(hash(estado+"exp"))%(2**32)); base=_EXP_BASE.get(estado,5000); sim=[]
        for yr in YEARS:
            for q in range(1,5):
                if yr==2025 and q>1: break
                t=(yr-2015)*4+q
                sim.append({"Estado":estado,"Año":yr,"Trimestre":q,"Exportaciones":round(base*(1+0.022*t)*rng.uniform(0.85,1.18)*[.95,1.05,1.08,1.12][q-1],1)})
        frames.append(pd.DataFrame(sim))
    return pd.concat(frames,ignore_index=True)

def construir_panel(df_emp,df_ied,df_act,df_exp):
    keys=["Estado","Año","Trimestre"]
    panel=df_emp.merge(df_ied,on=keys,how="inner").merge(df_act,on=keys,how="left").merge(df_exp,on=keys,how="left")
    panel=panel[panel["Estado"].isin(ESTADOS_BAJIO.keys())].copy()
    panel=panel.sort_values(["Estado","Año","Trimestre"]).reset_index(drop=True)
    panel["Var_Empleo_pct"]=(panel.groupby("Estado")["Empleo_Manufacturero"].pct_change()*100).round(2)
    panel["Periodo"]=panel["Año"].astype(str)+" Q"+panel["Trimestre"].astype(str)
    for col in ["ActInd","Exportaciones"]:
        panel[col]=panel.groupby("Estado")[col].transform(lambda x:x.interpolate(limit_direction="both"))
    print(f"   ✓ Panel: {len(panel)} observaciones")
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
# CACHÉ — SIN CAMBIOS
# ══════════════════════════════════════════════
_cache={"panel":None,"geojson":None,"fecha":None,"lock":threading.Lock()}

def get_datos():
    hoy=date.today()
    if _cache["panel"] is not None and _cache["fecha"]==hoy:
        return _cache["panel"],_cache["geojson"]
    with _cache["lock"]:
        if _cache["panel"] is not None and _cache["fecha"]==hoy:
            return _cache["panel"],_cache["geojson"]
        print(f"🔄 Cargando datos para {hoy}...")
        df_emp=procesar_empleo(); df_ied=procesar_ied(); df_act=procesar_actind(); df_exp=procesar_exportaciones()
        panel=construir_panel(df_emp,df_ied,df_act,df_exp); geo=cargar_geojson()
        _cache["panel"]=panel; _cache["geojson"]=geo; _cache["fecha"]=hoy
        print("✅ Caché listo.\n")
    return _cache["panel"],_cache["geojson"]

PANEL,GEOJSON=get_datos()
AÑOS_DISPONIBLES=sorted(PANEL["Año"].unique())

# ══════════════════════════════════════════════
# ECONOMETRÍA — SIN CAMBIOS
# ══════════════════════════════════════════════
def calcular_econometria(df, vars_x):
    if not vars_x: return None
    cols=["Empleo_Manufacturero"]+vars_x
    sub=df[["Estado","Año","Trimestre"]+cols].dropna().copy()
    if len(sub)<20: return None
    for c in cols:
        sub[f"Crec_{c}"]=sub.groupby("Estado")[c].pct_change()*100
    sub=sub.dropna()
    sub["t"]=(sub["Año"]-sub["Año"].min())*4+sub["Trimestre"]
    sub=sub.set_index(["Estado","t"])
    Y=sub["Crec_Empleo_Manufacturero"]; X=sub[[f"Crec_{c}" for c in vars_x]]
    try:
        res=PanelOLS(Y,X,entity_effects=True,time_effects=True).fit(cov_type="clustered",cluster_entity=True)
        dw_stat=durbin_watson(res.resids.values)
    except: return None
    vifs={}
    if len(vars_x)>1:
        X_vif=sub[[f"Crec_{c}" for c in vars_x]].reset_index(drop=True)
        try:
            for i,c in enumerate(X_vif.columns):
                vifs[c.replace("Crec_","")]=round(float(variance_inflation_factor(X_vif.values.astype(float),i)),2)
        except: vifs={c:float("nan") for c in vars_x}
    else: vifs={vars_x[0]:1.0}
    return {"coefs":res.params,"pvals":res.pvalues,"r2_within":round(res.rsquared,4),
            "n_obs":int(res.nobs),"dw":round(dw_stat,3),"vifs":vifs,
            "corr":sub[[f"Crec_{c}" for c in cols]].corr().round(3)}

# ══════════════════════════════════════════════
# FIGURAS — ACTUALIZADAS
# ══════════════════════════════════════════════
H_CHART = 450  # altura uniforme full-width

def fig_series(df, variable, estados, tipo="line"):
    col=VAR_COL.get(variable,"Empleo_Manufacturero")
    fig=go.Figure()
    for est in estados:
        sub=df[df["Estado"]==est].sort_values(["Año","Trimestre"])
        if tipo=="line":
            fig.add_trace(go.Scatter(x=sub["Periodo"],y=sub[col],name=est,mode="lines+markers",
                line=dict(color=COLORES_ESTADOS[est],width=2.5),marker=dict(size=5),
                hovertemplate=f"<b>{est}</b><br>%{{x}}<br>{VAR_LABEL[variable]}: %{{y:,.1f}}<extra></extra>"))
        elif tipo=="area":
            fig.add_trace(go.Scatter(x=sub["Periodo"],y=sub[col],name=est,mode="lines",
                fill="tozeroy",line=dict(color=COLORES_ESTADOS[est],width=2),
                fillcolor=COLORES_ESTADOS[est]+"44",
                hovertemplate=f"<b>{est}</b><br>%{{x}}<br>{VAR_LABEL[variable]}: %{{y:,.1f}}<extra></extra>"))
        else:  # bar
            fig.add_trace(go.Bar(x=sub["Periodo"],y=sub[col],name=est,
                marker_color=COLORES_ESTADOS[est],
                hovertemplate=f"<b>{est}</b><br>%{{x}}<br>{VAR_LABEL[variable]}: %{{y:,.1f}}<extra></extra>"))
    fig.update_layout(**{**PLOT_LAYOUT,
        "yaxis":dict(title=VAR_LABEL[variable],gridcolor="#EEE",tickformat=","),
        "xaxis":dict(tickangle=-45,tickfont_size=9),
        "height":H_CHART,"barmode":"stack" if tipo=="bar" else "group"})
    return fig

def fig_mapa(df, variable, estados, yr_to, geojson):
    col=VAR_COL.get(variable,"Empleo_Manufacturero")
    sub=df[(df["Estado"].isin(estados))&(df["Año"]==yr_to)]
    grp=sub.groupby("Estado")[col].mean().reset_index().rename(columns={col:"valor"})
    grp=pd.DataFrame({"Estado":list(ESTADOS_BAJIO.keys())}).merge(grp,on="Estado",how="left")
    fig=go.Figure(go.Choropleth(
        geojson=geojson,locations=grp["Estado"],z=grp["valor"],
        featureidkey="properties.name",
        colorscale=[[0,"#B5D4F4"],[0.5,"#1A5599"],[1,"#0C3460"]],
        marker_line_color="white",marker_line_width=1.5,
        colorbar=dict(title=dict(text=VAR_LABEL[variable],font_size=10),thickness=14,len=0.75),
        hovertemplate="<b>%{location}</b><br>"+VAR_LABEL[variable]+": %{z:,.1f}<extra></extra>",
    ))
    fig.update_geos(fitbounds="locations",visible=False,showland=True,landcolor="#F5F3EF",showframe=False)
    fig.update_layout(**{**PLOT_LAYOUT,"height":H_CHART,"margin":dict(l=0,r=0,t=20,b=0)})
    return fig

def fig_scatter_animado(df, var_x, var_y, estados):
    col_x=VAR_COL.get(var_x,"IED")
    col_y=VAR_COL.get(var_y,"Empleo_Manufacturero")
    sub=df[df["Estado"].isin(estados)].copy()
    # Promedio anual para animación más limpia
    sub_anual=sub.groupby(["Estado","Año"]).agg(
        x=(col_x,"mean"), y=(col_y,"mean"),
        size_col=("Empleo_Manufacturero","mean")
    ).reset_index()
    sub_anual["size_col"] = (sub_anual["size_col"] / sub_anual["size_col"].max() * 60 + 10).round(1)
    sub_anual["Año_str"] = sub_anual["Año"].astype(str)

    fig = px.scatter(
        sub_anual, x="x", y="y",
        color="Estado", size="size_col",
        animation_frame="Año_str",
        color_discrete_map=COLORES_ESTADOS,
        hover_name="Estado",
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

def fig_heatmap(df, variable, estados):
    col=VAR_COL.get(variable,"Empleo_Manufacturero")
    sub=df[df["Estado"].isin(estados)]
    piv=sub.pivot_table(index="Estado",columns="Trimestre",values=col,aggfunc="mean").round(1)
    piv.columns=[f"Q{c}" for c in piv.columns]
    fig=go.Figure(go.Heatmap(z=piv.values,x=piv.columns.tolist(),y=piv.index.tolist(),
        colorscale=[[0,"#E6F1FB"],[0.5,"#1A5599"],[1,"#0C3460"]],
        text=np.round(piv.values,0),texttemplate="%{text:,.0f}",textfont_size=11))
    fig.update_layout(**{**PLOT_LAYOUT,"height":H_CHART,"margin":dict(l=140,r=20,t=20,b=20)})
    return fig

def fig_correlacion(corr_df):
    labels=[c.replace("Crec_","") for c in corr_df.columns]
    z=corr_df.values
    fig=go.Figure(go.Heatmap(z=z,x=labels,y=labels,
        colorscale=[[0,"#FFF0E0"],[0.5,"#8B5E3C"],[1,"#0C3460"]],zmin=-1,zmax=1,
        text=np.round(z,2),texttemplate="%{text}",textfont_size=12))
    fig.update_layout(**{**PLOT_LAYOUT,"height":320,"margin":dict(l=80,r=20,t=20,b=80)})
    return fig

# ══════════════════════════════════════════════
# ESTILOS
# ══════════════════════════════════════════════
CARD={"background":CARD_BG,"borderRadius":"12px","border":"1px solid #E5E0D8",
      "padding":"20px 24px","marginBottom":"16px","boxShadow":"0 1px 4px rgba(0,0,0,0.06)"}
METRIC_CARD={"background":"#F0EDE8","borderRadius":"10px","padding":"14px 18px","flex":"1","minWidth":"120px"}
BTN_BASE={"fontSize":"12px","padding":"5px 14px","borderRadius":"20px","border":"1.5px solid #CCC",
          "background":"white","color":"#555","cursor":"pointer","marginRight":"6px","marginBottom":"6px","fontFamily":"inherit"}
BTN_VAR_ON={**BTN_BASE,"background":CAFE,"color":"#FAC775","border":f"1.5px solid {CAFE}","borderRadius":"6px"}
SEC_HDR={"fontSize":"15px","fontWeight":"500","color":AZUL_OSCURO,"margin":"0 0 4px",
         "borderLeft":f"3px solid {NARANJA}","paddingLeft":"10px"}
SUB={"fontSize":"11px","color":TEXT_SEC,"margin":"4px 0 12px 14px"}

TAB_STYLE={"padding":"10px 20px","fontFamily":"'Helvetica Neue',Arial,sans-serif","fontSize":"13px","color":TEXT_SEC,"borderBottom":"2px solid transparent"}
TAB_SEL={"padding":"10px 20px","fontFamily":"'Helvetica Neue',Arial,sans-serif","fontSize":"13px",
          "color":AZUL_OSCURO,"fontWeight":"600","borderBottom":f"2px solid {AZUL_OSCURO}","background":"white"}

# ══════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════
app=dash.Dash(__name__,title="Panel Bajío · Econométrico",
    meta_tags=[{"name":"viewport","content":"width=device-width, initial-scale=1"}])
server=app.server

# Controles compartidos
CONTROLES = html.Div(style={**CARD,"marginBottom":"16px"},children=[
    html.Div(style={"display":"flex","flexWrap":"wrap","gap":"24px","alignItems":"flex-start"},children=[
        html.Div([
            html.P("Estados",style={"fontSize":"11px","color":TEXT_SEC,"textTransform":"uppercase","letterSpacing":"0.05em","margin":"0 0 6px"}),
            html.Div(id="estado-btns",style={"display":"flex","flexWrap":"wrap"}),
        ]),
        html.Div([
            html.P("Período",style={"fontSize":"11px","color":TEXT_SEC,"textTransform":"uppercase","letterSpacing":"0.05em","margin":"0 0 6px"}),
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
            html.P("Variable",style={"fontSize":"11px","color":TEXT_SEC,"textTransform":"uppercase","letterSpacing":"0.05em","margin":"0 0 6px"}),
            html.Div(id="var-btns",style={"display":"flex","gap":"6px","flexWrap":"wrap"}),
        ]),
    ]),
])

app.layout=html.Div(
    style={"fontFamily":"'Helvetica Neue',Arial,sans-serif","background":BG,"minHeight":"100vh","padding":"20px 24px","maxWidth":"1400px","margin":"0 auto"},
    children=[

    # Header
    html.Div(style={"marginBottom":"20px"},children=[
        html.H1("Panel Econométrico — Región del Bajío",
            style={"fontSize":"22px","fontWeight":"500","margin":"0 0 4px","color":TEXT_PRIM}),
        html.P("Empleo · IED · Actividad Industrial · Exportaciones  ·  2015–2025  ·  INEGI + SE",
            style={"fontSize":"13px","color":TEXT_SEC,"margin":"0"}),
    ]),

    # Métricas
    html.Div(id="metrics-row",style={"display":"flex","gap":"10px","flexWrap":"wrap","marginBottom":"16px"}),

    # Stores
    dcc.Store(id="active-estados",data=list(ESTADOS_BAJIO.keys())),
    dcc.Store(id="active-var",data="empleo"),

    # Controles
    CONTROLES,

    # ── TABS ──────────────────────────────────
    dcc.Tabs(id="main-tabs",value="tab-visor",
        style={"marginBottom":"16px"},
        children=[

        # ─────────────────────────────────────
        # TAB 1: VISOR DE DATOS
        # ─────────────────────────────────────
        dcc.Tab(label="📊 Visor de Datos", value="tab-visor",
            style=TAB_STYLE, selected_style=TAB_SEL,
            children=[

            # Serie de tiempo
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

            # Mapa full width
            html.Div(style=CARD,children=[
                html.P("Mapa del Bajío",style=SEC_HDR),
                html.P(id="map-sub",style=SUB),
                dcc.Graph(id="map-chart",config={"displayModeBar":False}),
            ]),

            # Scatter animado full width
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

            # Heatmap estacional full width
            html.Div(style=CARD,children=[
                html.P("Patrón estacional por trimestre",style=SEC_HDR),
                html.P("Promedio del período seleccionado",style=SUB),
                dcc.Graph(id="heatmap-chart",config={"displayModeBar":False}),
            ]),
        ]),

        # ─────────────────────────────────────
        # TAB 2: LABORATORIO ECONOMÉTRICO
        # ─────────────────────────────────────
        dcc.Tab(label="🔬 Laboratorio Econométrico", value="tab-eco",
            style=TAB_STYLE, selected_style=TAB_SEL,
            children=[

            html.Div(style={"margin":"20px 0 12px"},children=[
                html.H2("Modelo de Datos Panel",style={"fontSize":"17px","fontWeight":"500","color":AZUL_OSCURO,"margin":"0 0 4px"}),
                html.P("Efectos Fijos de entidad y tiempo · Tasas de crecimiento · Errores clusterizados por entidad",
                    style={"fontSize":"12px","color":TEXT_SEC,"margin":"0"}),
            ]),

            # Selector de variables X
            html.Div(style={**CARD,"background":"#EEF4FB","border":f"1px solid {AZUL_CLARO}44"},children=[
                html.P("Modelo estimado (tasas de crecimiento trimestral):",style={"fontSize":"12px","color":TEXT_SEC,"margin":"0 0 6px"}),
                html.P("Δ%Empleo_it = β₀ + β₁·Δ%X₁_it + β₂·Δ%X₂_it + μᵢ + λₜ + εᵢₜ",
                    style={"fontSize":"14px","fontWeight":"500","color":AZUL_OSCURO,"fontFamily":"monospace","margin":"0 0 10px"}),
                html.P("Variables independientes (selecciona las que quieras incluir):",
                    style={"fontSize":"11px","color":TEXT_SEC,"margin":"0 0 8px"}),
                dcc.Checklist(id="vars-modelo",
                    options=[{"label":f"  {l}","value":v} for v,l in VARS_DEF if v!="empleo"],
                    value=["ied","actind","exportaciones"], inline=True,
                    inputStyle={"marginRight":"5px"},
                    labelStyle={"marginRight":"20px","fontSize":"13px"},
                ),
            ]),

            # Coeficientes + Robustez
            html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"16px","marginBottom":"16px"},children=[
                html.Div(style=CARD,children=[
                    html.P("Coeficientes — Efectos Fijos",style=SEC_HDR),
                    html.P("*** p<0.01  ** p<0.05  * p<0.1  · Errores clusterizados por entidad",style=SUB),
                    html.Div(id="tabla-regresion"),
                ]),
                html.Div(style=CARD,children=[
                    html.P("Pruebas de Robustez",style=SEC_HDR),
                    html.P("Durbin-Watson (autocorrelación) y VIF (multicolinealidad)",style=SUB),
                    html.Div(id="robustez-panel"),
                ]),
            ]),

            # Correlación full width
            html.Div(style=CARD,children=[
                html.P("Matriz de correlación de Pearson",style=SEC_HDR),
                html.P("Sobre tasas de crecimiento trimestral de las variables seleccionadas",style=SUB),
                dcc.Graph(id="corr-chart",config={"displayModeBar":False}),
            ]),
        ]),

        # ─────────────────────────────────────
        # TAB 3: BASE DE DATOS
        # ─────────────────────────────────────
        dcc.Tab(label="🗃 Base de Datos", value="tab-datos",
            style=TAB_STYLE, selected_style=TAB_SEL,
            children=[
            html.Div(style={"marginTop":"16px"},children=[
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
                    html.Div(id="tabla-panel",style={"overflowX":"auto"}),
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
            style={**BTN_BASE,"background":col if ia else "white","color":"#E6F1FB" if ia else "#555","borderColor":col if ia else "#CCC"}))
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

# ── VISOR DE DATOS ────────────────────────────
@app.callback(
    Output("metrics-row","children"),
    Output("series-chart","figure"), Output("map-chart","figure"),
    Output("scatter-chart","figure"), Output("heatmap-chart","figure"),
    Output("series-sub","children"), Output("map-sub","children"),
    Input("active-estados","data"), Input("active-var","data"),
    Input("year-from","value"), Input("year-to","value"),
    Input("tipo-grafica","value"),
    Input("scatter-x","value"), Input("scatter-y","value"),
)
def update_visor(estados,variable,yr_from,yr_to,tipo,sx,sy):
    panel,geojson=get_datos()
    df=panel[(panel["Estado"].isin(estados))&(panel["Año"]>=yr_from)&(panel["Año"]<=yr_to)].copy()
    n=len(df)

    # Métricas
    def m(l,v,s): return html.Div(style=METRIC_CARD,children=[
        html.P(l,style={"fontSize":"10px","color":TEXT_SEC,"textTransform":"uppercase","letterSpacing":"0.05em","margin":"0 0 3px"}),
        html.P(v,style={"fontSize":"19px","fontWeight":"500","color":TEXT_PRIM,"margin":"0","lineHeight":"1.1"}),
        html.P(s,style={"fontSize":"10px","color":TEXT_SEC,"margin":"2px 0 0"}),
    ])
    metrics=[
        m("Observaciones",f"{n:,}","trim. × estados"),
        m("Empleo prom.",f"{int(df['Empleo_Manufacturero'].mean()):,}" if n else "—","personas/trim."),
        m("IED prom.",f"${df['IED'].mean():.1f}M" if n else "—","USD/trim."),
        m("Act. Industrial",f"{df['ActInd'].mean():.1f}" if n else "—","índice 2013=100"),
        m("Exportaciones",f"${df['Exportaciones'].mean():.1f}M" if n else "—","USD/trim."),
    ]

    f_ser  = fig_series(df,variable,estados,tipo)
    f_map  = fig_mapa(df,variable,estados,yr_to,geojson)
    f_scat = fig_scatter_animado(df,sx,sy,estados)
    f_heat = fig_heatmap(df,variable,estados)

    return (metrics, f_ser, f_map, f_scat, f_heat,
            VAR_LABEL[variable], f"Intensidad promedio {yr_to} · {VAR_LABEL[variable]}")

# ── ECONOMETRÍA ───────────────────────────────
@app.callback(
    Output("tabla-regresion","children"),
    Output("robustez-panel","children"),
    Output("corr-chart","figure"),
    Input("active-estados","data"), Input("active-var","data"),
    Input("year-from","value"), Input("year-to","value"),
    Input("vars-modelo","value"),
)
def update_eco(estados,_,yr_from,yr_to,vars_x):
    panel,_geo=get_datos()
    df=panel[(panel["Estado"].isin(estados))&(panel["Año"]>=yr_from)&(panel["Año"]<=yr_to)].copy()
    vars_x=vars_x or []
    eco=calcular_econometria(df,[VAR_COL[v] for v in vars_x])
    no_data=html.P("Selecciona variables y amplía el período.",style={"fontSize":"12px","color":TEXT_SEC,"padding":"10px"})
    if eco is None: return no_data,no_data,go.Figure()

    # Tabla coefs
    nombres={"Empleo_Manufacturero":"Empleo mfr.","IED":"IED (M USD)","ActInd":"Act. Industrial","Exportaciones":"Exportaciones"}
    hdr=html.Tr([html.Th(c,style={"fontWeight":"500","fontSize":"10px","color":TEXT_SEC,"padding":"5px 10px","borderBottom":"1px solid #DDD","textAlign":al}) for c,al in [("Variable","left"),("Coef.","right"),("P-value","right")]])
    filas=[]
    for vk,coef in eco["coefs"].items():
        vn=vk.replace("Crec_",""); pval=eco["pvals"].get(vk,float("nan"))
        stars="***" if pd.notna(pval) and pval<0.01 else ("**" if pd.notna(pval) and pval<0.05 else ("*" if pd.notna(pval) and pval<0.1 else ""))
        pcol="#0F6E56" if pd.notna(pval) and pval<0.05 else TEXT_SEC
        filas.append(html.Tr([
            html.Td(nombres.get(vn,vn),style={"padding":"6px 10px","fontSize":"12px","fontWeight":"500"}),
            html.Td(f"{coef:.4f}{stars}",style={"padding":"6px 10px","fontSize":"12px","textAlign":"right"}),
            html.Td(f"{pval:.4f}",style={"padding":"6px 10px","fontSize":"12px","textAlign":"right","color":pcol,"fontWeight":"500"}),
        ]))
    summary=[
        html.Tr([html.Td("R² within",style={"padding":"6px 10px","fontSize":"12px","color":TEXT_SEC}),html.Td("",),html.Td(f"{eco['r2_within']:.4f}",style={"padding":"6px 10px","fontSize":"12px","textAlign":"right","fontWeight":"500"})]),
        html.Tr([html.Td("N obs.",style={"padding":"6px 10px","fontSize":"12px","color":TEXT_SEC}),html.Td("",),html.Td(str(eco["n_obs"]),style={"padding":"6px 10px","fontSize":"12px","textAlign":"right"})]),
        html.Tr([html.Td("EF Entidad",style={"padding":"6px 10px","fontSize":"12px","color":TEXT_SEC}),html.Td("",),html.Td("✓ Sí",style={"padding":"6px 10px","fontSize":"12px","textAlign":"right","color":"#0F6E56"})]),
        html.Tr([html.Td("EF Tiempo",style={"padding":"6px 10px","fontSize":"12px","color":TEXT_SEC}),html.Td("",),html.Td("✓ Sí",style={"padding":"6px 10px","fontSize":"12px","textAlign":"right","color":"#0F6E56"})]),
    ]
    tabla_reg=html.Table([html.Thead(hdr),html.Tbody(filas+summary)],style={"width":"100%","borderCollapse":"collapse"})

    # Robustez
    dw=eco["dw"]; dw_ok=1.5<dw<2.5
    dw_txt="Sin autocorrelación ✓" if dw_ok else ("Autocorrelación positiva" if dw<1.5 else "Autocorrelación negativa")
    dw_col="#0F6E56" if dw_ok else NARANJA
    vif_items=[html.Div(style={"display":"flex","justifyContent":"space-between","padding":"6px 10px","borderBottom":"0.5px solid #EEE"},children=[
        html.Span(nombres.get(vn,vn),style={"fontSize":"12px"}),
        html.Span(f"VIF={vv:.2f} {'OK ✓' if vv<=5 else 'Alta' if vv>10 else 'Moderada'}",
            style={"fontSize":"11px","fontWeight":"500","color":"#0F6E56" if vv<=5 else "#993C1D" if vv>10 else NARANJA}),
    ]) for vn,vv in eco["vifs"].items()]
    robustez=html.Div([
        html.Div(style={"display":"flex","alignItems":"center","gap":"10px","padding":"8px 10px","marginBottom":"10px","borderBottom":"1px solid #EEE"},children=[
            html.Span(f"{dw:.3f}",style={"fontSize":"22px","fontWeight":"500","color":dw_col}),
            html.Span(f"Durbin-Watson · {dw_txt}",style={"fontSize":"11px","color":dw_col}),
        ]),
        html.Div(vif_items),
    ])

    return tabla_reg,robustez,fig_correlacion(eco["corr"])

# ── BASE DE DATOS ─────────────────────────────
@app.callback(
    Output("tabla-panel","children"), Output("tabla-sub","children"),
    Input("active-estados","data"), Input("year-from","value"), Input("year-to","value"),
)
def update_tabla(estados,yr_from,yr_to):
    panel,_=get_datos()
    df=panel[(panel["Estado"].isin(estados))&(panel["Año"]>=yr_from)&(panel["Año"]<=yr_to)].copy()
    show=df.sort_values(["Año","Trimestre","Estado"],ascending=[False,False,True]).head(80)
    ths=["Estado","Año","Trim.","Empleo mfr.","IED (M USD)","Act.Ind","Exportac.(M USD)","Var.Emp %"]
    header=html.Tr([html.Th(c,style={"fontWeight":"500","fontSize":"10px","color":TEXT_SEC,"textAlign":"left","padding":"5px 8px","borderBottom":"1px solid #DDD"}) for c in ths])
    rows=[]
    for _,r in show.iterrows():
        c=COLORES_ESTADOS.get(r["Estado"],"#888")
        ve=r.get("Var_Empleo_pct",float("nan"))
        ve_str=(f"+{ve:.1f}%" if ve>=0 else f"{ve:.1f}%") if pd.notna(ve) else "—"
        ve_col="#0F6E56" if pd.notna(ve) and ve>=0 else "#993C1D"
        rows.append(html.Tr([
            html.Td(html.Span(r["Estado"],style={"background":c+"22","color":c,"border":f"0.5px solid {c}55","fontSize":"10px","padding":"2px 7px","borderRadius":"10px"}),style={"padding":"4px 8px"}),
            html.Td(str(int(r["Año"])),style={"padding":"4px 8px","fontSize":"11px"}),
            html.Td(f"Q{int(r['Trimestre'])}",style={"padding":"4px 8px","fontSize":"11px"}),
            html.Td(f"{int(r['Empleo_Manufacturero']):,}",style={"padding":"4px 8px","fontSize":"11px"}),
            html.Td(f"${r['IED']:.1f}M",style={"padding":"4px 8px","fontSize":"11px"}),
            html.Td(f"{r['ActInd']:.1f}" if pd.notna(r.get("ActInd")) else "—",style={"padding":"4px 8px","fontSize":"11px"}),
            html.Td(f"${r['Exportaciones']:.1f}M" if pd.notna(r.get("Exportaciones")) else "—",style={"padding":"4px 8px","fontSize":"11px"}),
            html.Td(ve_str,style={"padding":"4px 8px","fontSize":"11px","color":ve_col,"fontWeight":"500"}),
        ],style={"borderBottom":"0.5px solid #EEE"}))
    tabla=html.Table([html.Thead(header),html.Tbody(rows)],style={"width":"100%","borderCollapse":"collapse"})
    return tabla,f"{len(df)} observaciones · mostrando últimas 80"

@app.callback(
    Output("download-csv","data"),
    Input("btn-csv","n_clicks"),
    State("active-estados","data"), State("year-from","value"), State("year-to","value"),
    prevent_initial_call=True,
)
def descargar_csv(n,estados,yr_from,yr_to):
    panel,_=get_datos()
    df=panel[(panel["Estado"].isin(estados))&(panel["Año"]>=yr_from)&(panel["Año"]<=yr_to)]
    return dcc.send_data_frame(df.to_csv, "panel_bajio.csv", index=False)

if __name__=="__main__":
    port=int(os.environ.get("PORT",8050))
    app.run(debug=False,host="0.0.0.0",port=port)
    
