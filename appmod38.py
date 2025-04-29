import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import io

st.set_page_config(layout="wide")

@st.cache_data
def carregar_dados():
    df = pd.read_feather('credit_scoring.ftr')
    return df

st.title("📊 Credit Scoring - App Interativo com Streamlit")

df = carregar_dados()

aba = st.sidebar.radio("Navegação", ['Visão Geral', 'Análise Exploratória', 'Modelagem'])

# ===============================
# ABA 1 - Visão Geral
# ===============================
if aba == 'Visão Geral':
    st.subheader("🔍 Visão Geral dos Dados")
    st.dataframe(df.head())

    datas = sorted(df['data_ref'].unique())
    st.markdown("**Datas únicas mais recentes:**")
    st.write(datas[-5:])

    st.write(f"Total de linhas no dataset: {df.shape[0]}")
    st.write("Número de registros por mês:")
    st.bar_chart(df['data_ref'].value_counts().sort_index())

# ===============================
# ABA 2 - Análise Exploratória
# ===============================
elif aba == 'Análise Exploratória':
    st.subheader("📈 Análise Exploratória")

    variaveis_quant = ['idade', 'tempo_emprego', 'qt_pessoas_residencia', 'renda']
    variaveis_qualitativas = ['sexo', 'posse_de_veiculo', 'posse_de_imovel',
                              'tipo_renda', 'educacao', 'estado_civil',
                              'tipo_residencia', 'mau']

    tipo = st.selectbox("Escolha o tipo de variável para explorar:", ['Quantitativa', 'Qualitativa'])

    if tipo == 'Quantitativa':
        var = st.selectbox("Escolha uma variável:", variaveis_quant)
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x='mau', y=var, ax=ax)
        ax.set_title(f'{var} por Inadimplência')
        st.pyplot(fig)

    else:
        var = st.selectbox("Escolha uma variável categórica:", variaveis_qualitativas)
        tab = pd.crosstab(df[var], df['mau'], margins=True)
        prop = pd.crosstab(df[var], df['mau'], normalize='index') * 100
        st.markdown("**Contagem absoluta:**")
        st.dataframe(tab)
        st.markdown("**Proporção de inadimplentes (%):**")
        st.dataframe(prop[True].round(2).astype(str) + '%')

# ===============================
# ABA 3 - Modelagem
# ===============================
elif aba == 'Modelagem':
    st.subheader("🤖 Treinamento e Avaliação do Modelo")

    colunas_numericas = ['idade', 'tempo_emprego', 'qt_pessoas_residencia', 'renda', 'qtd_filhos']
    colunas_categoricas = ['sexo', 'posse_de_veiculo', 'posse_de_imovel',
                           'tipo_renda', 'educacao', 'estado_civil', 'tipo_residencia']

    df_modelagem = df.copy()
    df_modelagem['tempo_emprego'] = df_modelagem['tempo_emprego'].fillna(0)
    df_modelagem['qt_pessoas_residencia'] = df_modelagem['qt_pessoas_residencia'].fillna(df_modelagem['qt_pessoas_residencia'].median())

    df_modelagem['educacao'] = df_modelagem['educacao'].replace({
        'Fundamental incompleto': 'Fundamental',
        'Superior incompleto': 'Superior',
        'Pós-graduação': 'Superior'
    })
    df_modelagem['tipo_renda'] = df_modelagem['tipo_renda'].replace({
        'Servidor público': 'Assalariado',
        'Empresário': 'Outros',
        'Pensionista': 'Outros',
        'Bolsista': 'Outros',
        'Outro': 'Outros'
    })

    X = df_modelagem.drop(columns=['mau', 'data_ref', 'index'])
    y = df_modelagem['mau']

    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), colunas_numericas),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ]), colunas_categoricas)
    ])

    pipeline_modelo = Pipeline(steps=[
        ('preprocessamento', preprocessor),
        ('modelo', LogisticRegression(max_iter=1000))
    ])

    if st.button("Treinar Modelo"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
        pipeline_modelo.fit(X_train, y_train)
        y_pred = pipeline_modelo.predict(X_test)
        y_proba = pipeline_modelo.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)

        st.markdown(f"**Acurácia:** {acc:.4f}")
        st.markdown(f"**AUC:** {auc:.4f}")
        st.markdown("**Relatório de Classificação:**")
        st.text(classification_report(y_test, y_pred))
