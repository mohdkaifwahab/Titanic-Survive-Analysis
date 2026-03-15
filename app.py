import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# page config
st.set_page_config(page_title='Titanic Suvived Analysis',layout='wide',page_icon='📈')

# Make dashboard text and metrics larger for better readability
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; max-width: 98%;}
    h1 {font-size: 3rem !important;}
    h2, h3 {font-size: 2rem !important;}
    p, li, label, .stMarkdown {font-size: 1.15rem !important;}
    [data-testid="stMetricValue"] {font-size: 2.2rem !important;}
    [data-testid="stMetricLabel"] {font-size: 1.1rem !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
})

# cache data
@st.cache_data
def load_data():
    df = pd.read_csv('titanic.csv')
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    return df

df = load_data()


## Tittle 
st.title('Titanic Survived Analysis Dashboard')
st.markdown("### Analyzing passenger demographics and survival rates from the Titanic dataset.")


# KPIs
st.subheader('Key Performance Indicators (KPIs)')
col1, col2, col3 = st.columns(3)

# Total passengers
with col1:
    st.metric("👥 Total Passengers", df.shape[0])

# Total survived
with col2:
    survived_count = df['Survived'].sum()
    st.metric("✅ Total Survived", survived_count)

# Survival rate
with col3:
    survival_rate = survived_count / df.shape[0] * 100
    st.metric("📊 Survival Rate (%)", f"{survival_rate:.2f}")


# first distribution of data 
col4, col5, col6 = st.columns(3)

with col4:
    st.subheader("Survival Count")

    fig, ax = plt.subplots(figsize=(7,6))
    sns.countplot(ax=ax,data=df,x='Survived')
    for p in ax.patches:
        h = p.get_height()
        ax.text(
            p.get_x() + p.get_width()/2,
            h/2,
            f'{int(h)}',
            ha='center',
            va='center',
            color='white'
        )

    st.pyplot(fig, use_container_width=True)


with col5:
    st.subheader('Survival Count According to Gender')
    fig , ax = plt.subplots(figsize=(7,6))
    sns.countplot(data=df, x='Survived',hue='Sex',ax=ax)
    for p in ax.patches:
        h = p.get_height()
        ax.text(
            p.get_x() + p.get_width()/2,
            h/2,
            f'{int(h)}',
            ha='center',
            va='center',
            color='white'
        )
    st.pyplot(fig, use_container_width=True)



with col6:
    st.subheader('Survival Count vs Passenger class')
    fig , ax = plt.subplots(figsize=(7,6))
    sns.countplot(data=df, x='Survived',hue='Pclass',ax=ax)
    for p in ax.patches:
        h = p.get_height()
        ax.text(
            p.get_x() + p.get_width()/2,
            h/2,
            f'{int(h)}',
            ha='center',
            va='center',
            color='white'
        )
    st.pyplot(fig, use_container_width=True)


# row 2 
col7 , col8 = st.columns(2)

with col7:
    st.subheader('Age Distribution')
    fig, ax = plt.subplots(figsize=(10,6))
    sns.histplot(x='Age',data=df,bins=30, ax=ax)
    st.pyplot(fig, use_container_width=True)


with col8:
    st.subheader('Fare vs Survival')
    fig, ax = plt.subplots(figsize=(10,6))
    sns.boxplot(x='Survived', y='Fare', data=df,ax=ax)
    st.pyplot(fig, use_container_width=True)



# row 3
st.subheader('Survival Correlation')
center_col = st.columns([1, 2, 1])[1]
with center_col:
    fig, ax = plt.subplots(figsize=(9, 6))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, ax=ax)
    st.pyplot(fig, use_container_width=True)









# Recommendations
st.subheader('Project Recommendations')

sex_survival = df.groupby('Sex')['Survived'].mean().sort_values(ascending=False)
class_survival = df.groupby('Pclass')['Survived'].mean().sort_values(ascending=False)
fare_median_survived = df[df['Survived'] == 1]['Fare'].median()
fare_median_not_survived = df[df['Survived'] == 0]['Fare'].median()

best_sex_group = sex_survival.index[0]
best_sex_rate = sex_survival.iloc[0] * 100
best_class_group = int(class_survival.index[0])
best_class_rate = class_survival.iloc[0] * 100

st.markdown('1. Improve emergency response planning by giving extra support to passenger groups with historically lower survival rates.')
st.markdown(f'2. Gender insight: **{best_sex_group.title()}** passengers have the highest survival rate at **{best_sex_rate:.2f}%**.')
st.markdown(f'3. Class insight: **Passenger Class {best_class_group}** records the highest survival rate at **{best_class_rate:.2f}%**.')
st.markdown(
    f'4. Fare insight: the median fare for survivors (**{fare_median_survived:.2f}**) is higher than for non-survivors '
    f'(**{fare_median_not_survived:.2f}**), indicating that ticket class and affordability likely influenced outcomes.'
)
st.markdown('5. For future prediction models, add engineered features such as FamilySize, title extraction from Name, and Embarked encoding to improve accuracy.')


## dataset
if st.checkbox('Show Dataset'):
    st.dataframe(df, use_container_width=True, height=500)