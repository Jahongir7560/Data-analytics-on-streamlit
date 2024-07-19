import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import io


df = pd.read_csv("exam.csv", index_col=0)


st.title('CREDIT CARD ANALYSIS')
st.subheader('DataFrame Head')
st.write(df.head())


st.sidebar.title('Navigation')
option = st.sidebar.radio(
    'Select a section',
    [
        'Data Overview',
        'Missing Values',
        'Descriptive Statistics',
        'Marital Status by Gender',
        'Education Level vs. Months on Book',
        'Customer Age by Income',
        'Total Relationship Count by Card Category',
        'Standardization vs Robust Scaling',
        'Customer Age Distribution',
        'Customer Age vs. Dependent Count by Gender',
        'Average Customer Age by Card Category and Gender',
        'Customer Age by Total Relationship Count',
        'Correlation Heatmap'
    ]
)


if option == 'Data Overview':
    st.subheader('DataFrame Head')
    st.write(df.head())


elif option == 'Missing Values':
    st.subheader('Missing Values')
    st.write(df.isnull().sum())

    columns_to_fill_float = [
        'CLIENTNUM', 'Customer_Age', 'Dependent_count', 'Months_on_book',
        'Total_Relationship_Count', 'Months_Inactive_12_mon', 'Contacts_Count_12_mon',
        'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
    ]

    df[columns_to_fill_float] = df[columns_to_fill_float].apply(lambda x: x.fillna(x.median()))

    columns_to_fill_object = [
        'Attrition_Flag', 'Gender', 'Education_Level', 'Marital_Status',
        'Income_Category', 'Card_Category'
    ]

    df[columns_to_fill_object] = df[columns_to_fill_object].apply(lambda x: x.fillna(x.mode()[0]))

    st.subheader("Processed Data Overview")
    st.write("Dataset with missing values filled using median(float) and mode(object):")
    st.write(df)


elif option == 'Descriptive Statistics':
    st.subheader('Describe Function')
    st.write(df.select_dtypes("number").describe())


elif option == 'Marital Status by Gender':
    st.subheader('Count of Marital Status by Gender')
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Marital_Status', hue='Gender', data=df)
    plt.xlabel('Marital Status')
    plt.ylabel('Count')
    st.pyplot(plt)


elif option == 'Education Level vs. Months on Book':
    st.subheader('Scatter Plot of Education Level vs. Months on Book')
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Education_Level', y='Months_on_book', data=df)
    plt.xlabel('Education Level')
    plt.ylabel('Months on Book')
    plt.xticks(rotation=45)
    st.pyplot(plt)


elif option == 'Customer Age by Income':
    st.subheader("Customer Age by Customer Income")
    g = sns.catplot(x="Income_Category", y="Customer_Age", data=df, hue="Gender", kind="box", col="Card_Category")
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(60)
    g.figure.tight_layout()
    st.pyplot(g.figure)


elif option == 'Total Relationship Count by Card Category':
    st.subheader('Histogram of Total Relationship Count by Card Category')
    plt.figure(figsize=(10, 6))
    sns.histplot(x="Total_Relationship_Count", y="Card_Category", data=df, kde=True)
    plt.xlabel('Total Relationship Count')
    plt.ylabel('Card Category')
    st.pyplot(plt)


elif option == 'Standardization vs Robust Scaling':
    st.subheader('Standardisation')
    df["Credit_Limit_stand"] = (df["Credit_Limit"] - df["Credit_Limit"].mean()) / df["Credit_Limit"].std()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(df["Credit_Limit"], ax=ax[0], kde=True, color='r')
    ax[0].set_title('Original Credit Limit')
    sns.histplot(df["Credit_Limit_stand"], ax=ax[1], kde=True, color='b')
    ax[1].set_title('Standardized Credit Limit')
    st.pyplot(fig)

    st.subheader('Robust Scaling')
    df["Months_on_book_Robust"] = (df["Months_on_book"] - df["Months_on_book"].median()) / (df["Months_on_book"].quantile(0.75) - df["Months_on_book"].quantile(0.25))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(df["Months_on_book"], ax=ax[0], kde=True, color='r')
    ax[0].set_title('Original Months_on_book')
    sns.histplot(df["Months_on_book_Robust"], ax=ax[1], kde=True, color='y')
    ax[1].set_title('Robust Months_on_book')
    st.pyplot(fig)


elif option == 'Customer Age Distribution':
    st.subheader('Customer Age Distribution')
    fig, ax = plt.subplots()
    sns.histplot(df["Customer_Age"], kde=True, bins=10, ax=ax)
    plt.xlabel('Customer Age')
    plt.ylabel('Frequency')
    plt.title('Distribution of Customer Age')
    st.pyplot(fig)


elif option == 'Customer Age vs. Dependent Count by Gender':
    st.subheader('Customer Age vs. Dependent Count by Gender')
    buffer = io.BytesIO()
    g = sns.lmplot(data=df, x="Dependent_count", y="Customer_Age", col="Gender")
    g.savefig(buffer, format='png')
    buffer.seek(0)
    st.image(buffer, use_column_width=True)


elif option == 'Average Customer Age by Card Category and Gender':
    st.subheader('Average Customer Age by Card Category and Gender')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x="Card_Category", y="Customer_Age", hue="Gender", ax=ax)
    ax.set_xlabel('Card Category')
    ax.set_ylabel('Average Age')
    st.pyplot(fig)


elif option == 'Customer Age by Total Relationship Count':
    st.subheader('Customer Age by Total Relationship Count')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df, x="Total_Relationship_Count", y="Customer_Age", ax=ax)
    ax.set_xlabel('Total Relationship Count')
    ax.set_ylabel('Customer Age')
    st.pyplot(fig)


elif option == 'Correlation Heatmap':
    st.subheader('Correlation Heatmap of Numerical Columns')
    st.write(df.select_dtypes("number").corr())
    correlation_matrix = df.select_dtypes(include='number').corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig)
