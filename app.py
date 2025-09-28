import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================
# 1. Cargar datos
# ==============================
@st.cache_data
def load_data():
    df = pd.read_csv("datos_saber11_clean.csv")  # ⚠️ aquí cargas el dataset limpio (o la muestra)
    return df

df = load_data()

st.title("📊 Análisis de Resultados Saber 11")
st.markdown("Explora los datos, descubre insights y prueba modelos predictivos interactivos.")

# ==============================
# 2. Tabs principales
# ==============================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Exploración de Datos", 
    "🤖 Modelos Predictivos", 
    "📌 Hipótesis", 
    "🔮 Predicción Personalizada"
])

# ==============================
# TAB 1: Exploración de Datos
# ==============================
with tab1:
    st.subheader("Exploración del Dataset")

    st.write("Vista previa del dataset:")
    st.dataframe(df.head())

    # Variables categóricas para explorar
    col_cat = st.selectbox("Selecciona una variable categórica para explorar", 
                           ["FAMI_ESTRATOVIVIENDA", "FAMI_EDUCACIONPADRE", "FAMI_EDUCACIONMADRE",
                            "FAMI_TIENEINTERNET", "FAMI_TIENECOMPUTADOR", "COLE_AREA_UBICACION", "COLE_BILINGUE"])

    fig = px.box(df, x=col_cat, y="PUNT_GLOBAL", points="all", 
                 title=f"Distribución del Puntaje Global según {col_cat}")
    st.plotly_chart(fig, use_container_width=True)

    # Scatter interactivo
    st.write("Relación entre dos variables numéricas y el puntaje:")
    x_var = st.selectbox("Eje X", ["PUNT_LECTURA_CRITICA", "PUNT_MATEMATICAS", "PUNT_SOCIALES_CIUDADANAS", "PUNT_C_NATURALES"])
    fig2 = px.scatter(df, x=x_var, y="PUNT_GLOBAL", color="FAMI_ESTRATOVIVIENDA", 
                      title=f"Puntaje Global vs {x_var}")
    st.plotly_chart(fig2, use_container_width=True)

# ==============================
# Preparación de datos para modelos
# ==============================
X = df[['FAMI_ESTRATOVIVIENDA', 'FAMI_EDUCACIONPADRE', 'FAMI_EDUCACIONMADRE',
        'FAMI_TIENEINTERNET', 'FAMI_TIENECOMPUTADOR', 
        'COLE_AREA_UBICACION', 'COLE_BILINGUE']]
y = df['PUNT_GLOBAL']

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================
# TAB 2: Modelos Predictivos
# ==============================
with tab2:
    st.subheader("Comparación de Modelos")

    # -------- Modelo 1: Regresión Lineal
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred_lin = lin_reg.predict(X_test)

    st.markdown("### 📌 Regresión Lineal")
    st.write("**R²:**", r2_score(y_test, y_pred_lin))
    st.write("**MAE:**", mean_absolute_error(y_test, y_pred_lin))
    st.write("**RMSE:**", np.sqrt(mean_squared_error(y_test, y_pred_lin)))

    coef = pd.DataFrame({"Variable": X.columns, "Coeficiente": lin_reg.coef_}).sort_values(by="Coeficiente", ascending=False)
    fig_coef = px.bar(coef, x="Coeficiente", y="Variable", orientation="h", title="Coeficientes Regresión Lineal")
    st.plotly_chart(fig_coef, use_container_width=True)

    # -------- Modelo 2: Random Forest
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    st.markdown("### 🌲 Random Forest (Optimizado)")
    st.write("**R²:**", r2_score(y_test, y_pred_rf))
    st.write("**MAE:**", mean_absolute_error(y_test, y_pred_rf))
    st.write("**RMSE:**", np.sqrt(mean_squared_error(y_test, y_pred_rf)))

    importances = pd.DataFrame({"Variable": X.columns, "Importancia": rf.feature_importances_}).sort_values(by="Importancia", ascending=False)
    fig_imp = px.bar(importances, x="Importancia", y="Variable", orientation="h", title="Importancia de Variables en Random Forest")
    st.plotly_chart(fig_imp, use_container_width=True)

# ==============================
# TAB 3: Hipótesis
# ==============================
with tab3:
    st.subheader("Validación de Hipótesis")

    st.markdown("""
    ### 📌 Hipótesis planteadas:
    1. **El estrato socioeconómico influye significativamente en el puntaje global.**  
       ✅ Confirmada: Los estudiantes con acceso a computador e internet (más frecuente en estratos altos) tienen mayores puntajes.  

    2. **El nivel educativo de los padres impacta en el puntaje global.**  
       ✅ Confirmada parcialmente: El Random Forest muestra alta importancia de educación de los padres, especialmente **postgrado**.  

    3. **El acceso a recursos (internet y computador) se relaciona con un mejor desempeño.**  
       ✅ Confirmada: Son las variables con mayor peso en el modelo Random Forest.  

    4. **La ubicación del colegio (urbano vs rural) determina diferencias en puntajes.**  
       ⚠️ Relación débil: Sí hay diferencias, pero no es la variable más importante.  

    """)

# ==============================
# TAB 4: Predicción Personalizada
# ==============================
with tab4:
    st.subheader("🔮 Predicción Personalizada del Puntaje Global")

    estrato = st.selectbox("Estrato de vivienda", df['FAMI_ESTRATOVIVIENDA'].unique())
    edu_padre = st.selectbox("Educación del padre", df['FAMI_EDUCACIONPADRE'].unique())
    edu_madre = st.selectbox("Educación de la madre", df['FAMI_EDUCACIONMADRE'].unique())
    internet = st.selectbox("¿Tiene internet en casa?", df['FAMI_TIENEINTERNET'].unique())
    computador = st.selectbox("¿Tiene computador en casa?", df['FAMI_TIENECOMPUTADOR'].unique())
    bilingue = st.selectbox("¿Colegio bilingüe?", df['COLE_BILINGUE'].unique())
    ubicacion = st.selectbox("Ubicación del colegio", df['COLE_AREA_UBICACION'].unique())

    if st.button("Predecir Puntaje Global"):
        new_data = pd.DataFrame({
            "FAMI_ESTRATOVIVIENDA": [estrato],
            "FAMI_EDUCACIONPADRE": [edu_padre],
            "FAMI_EDUCACIONMADRE": [edu_madre],
            "FAMI_TIENEINTERNET": [internet],
            "FAMI_TIENECOMPUTADOR": [computador],
            "COLE_BILINGUE": [bilingue],
            "COLE_AREA_UBICACION": [ubicacion]
        })

        # Encoding igual que en el entrenamiento
        new_data = pd.get_dummies(new_data, drop_first=True).reindex(columns=X.columns, fill_value=0)

        prediction = rf.predict(new_data)[0]
        st.success(f"✨ El puntaje global estimado es: **{prediction:.2f}**")



