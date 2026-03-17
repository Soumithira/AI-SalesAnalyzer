import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from database import create_db
from auth import register_user, login_user
import calendar
import sqlite3
import io

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="AI Sales Analyzer", layout="wide")
create_db()

# ================= SESSION =================
if "logged" not in st.session_state:
    st.session_state.logged = False

if "user" not in st.session_state:
    st.session_state.user = ""

# ================= LOGIN PAGE =================
def login_page():

    st.title("🔐 AI Sales Analyzer")

    option = st.selectbox("Choose Option", ["Login", "Register"])

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if option == "Register":
        if st.button("Register"):
            if register_user(username, password):
                st.success("Registered Successfully")
            else:
                st.error("User already exists")

    if option == "Login":
        if st.button("Login"):
            if login_user(username, password):
                st.session_state.logged = True
                st.session_state.user = username
                st.rerun()
            else:
                st.error("Invalid Credentials")

# ================= LOAD DATA =================
# ================= COLUMN DETECTION =================
def detect_column(df, possible_names):

    for col in possible_names:
        if col in df.columns:
            return col
    return None


def ensure_column(df, column_name, default_value):

    if column_name not in df.columns:
        df[column_name] = default_value

    return df
def load_data(file):

    df = pd.read_csv(file, low_memory=False)

    df.columns = df.columns.str.strip()

    # Detect important columns
    date_col = detect_column(df, ["OrderDate","Date","SaleDate","TransactionDate","InvoiceDate","Order_Date"])
    product_col = detect_column(df, ["ProductName","Product","Item","Product_Name"])
    country_col = detect_column(df, ["Country","Region","Market","State","Location"])
    quantity_col = detect_column(df, ["Quantity","Qty","Units"])
    revenue_col = detect_column(df, ["TotalAmount","Revenue","Sales"])
    shipping_col = detect_column(df, ["ShippingCost","Cost","DeliveryCost"])

    # Create standardized columns
    if date_col:
        df["OrderDate"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
        df = df.dropna(subset=["OrderDate"])
    else:
        st.error("❌ No date column detected in dataset.")
        st.stop()

    if product_col:
        df["ProductName"] = df[product_col]

    if country_col:
        df["Country"] = df[country_col]

    if quantity_col:
        df["Quantity"] = pd.to_numeric(df[quantity_col], errors="coerce").fillna(0)

    if revenue_col:
        df["Revenue"] = df[revenue_col].astype(str).str.replace("₹","").str.replace(",","")
        df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce").fillna(0)
    else:
        st.error("❌ No revenue/sales column detected in dataset.")
        st.stop()

    if shipping_col:
        df["Cost"] = pd.to_numeric(df[shipping_col], errors="coerce").fillna(0)
    else:
        df["Cost"] = 0

    # Ensure required columns exist
    df = ensure_column(df,"ProductName","Unknown Product")
    df = ensure_column(df,"Country","Unknown")

    # Business metrics
    df["Profit"] = df["Revenue"] - df["Cost"]
    df["Month"] = df["OrderDate"].dt.strftime("%B")

    return df

# ================= KPI =================
def kpi_section(df):

    st.subheader("📊 Business KPIs")

    total_revenue = df["Revenue"].sum()
    total_orders = len(df)
    best_product = df.groupby("ProductName")["Revenue"].sum().idxmax() if not df.empty else "N/A"
    countries = df["Country"].nunique()

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("💰 Total Revenue", f"₹{total_revenue:,.0f}")
    col2.metric("📦 Total Orders", total_orders)
    col3.metric("🏆 Best Product", best_product)
    col4.metric("🌍 Countries", countries)

# ================= BASIC INFO =================
def basic_info(df):

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📊 Dataset Info")

    st.write("Columns:", df.columns.tolist())
    st.write("Shape:", df.shape)

    st.dataframe(df.describe())

# ================= CHARTS =================
def charts_section(df):

    st.subheader("📈 Sales Trend")

    trend = df.groupby("OrderDate")["Revenue"].sum()

    st.line_chart(trend)

    st.subheader("📦 Product Sales")

    product_sales = df.groupby("ProductName")["Revenue"].sum().sort_values(ascending=False)

    st.bar_chart(product_sales)

    st.subheader("🌍 Country Sales")

    country_sales = df.groupby("Country")["Revenue"].sum().sort_values(ascending=False)

    st.bar_chart(country_sales)

# ================= PROFIT =================
def profit_analysis(df):

    st.subheader("💵 Profit Analysis")

    product_profit = df.groupby("ProductName")["Profit"].sum().sort_values(ascending=False)

    st.bar_chart(product_profit)

    best = product_profit.idxmax()
    worst = product_profit.idxmin()

    st.success(f"🏆 Most Profitable Product: {best}")
    st.warning(f"⚠ Least Profitable Product: {worst}")

# ================= PREDICTION =================
def prediction_section(df):

    st.subheader("🔮 Smart Monthly Product Intelligence")

    month_input = st.selectbox("Select Month", list(calendar.month_name)[1:])

    monthly_product = df.groupby(["Month","ProductName"])["Revenue"].sum().reset_index()

    month_data = monthly_product[monthly_product["Month"]==month_input]

    if not month_data.empty:

        best_product = month_data.sort_values("Revenue",ascending=False).iloc[0]

        lowest_product = month_data.sort_values("Revenue").iloc[0]

        st.success(f"🏆 Best Product in {month_input}: {best_product['ProductName']} | ₹{best_product['Revenue']:,.2f}")

        st.warning(f"📉 Lowest Product: {lowest_product['ProductName']} | ₹{lowest_product['Revenue']:,.2f}")

        fig,ax = plt.subplots(figsize=(12,5))

        ax.bar(month_data["ProductName"],month_data["Revenue"])

        plt.xticks(rotation=90)

        ax.grid(True)

        st.pyplot(fig)

    else:
        st.warning("No data for this month")


# ================= PRODUCT MONTHLY PERFORMANCE =================
    st.subheader("📅 Product Monthly Performance Intelligence")

    product_choice = st.selectbox("Select Product", df["ProductName"].unique())

    product_monthly = df[df["ProductName"] == product_choice] \
        .groupby("Month")["Revenue"].sum() \
        .reindex(list(calendar.month_name)[1:])

    best_month = product_monthly.idxmax()
    max_value = product_monthly.max()

    fig2, ax2 = plt.subplots(figsize=(12,5))
    ax2.plot(product_monthly.index, product_monthly.values, marker="o")
    ax2.set_title(f"{product_choice} Monthly Sales Trend")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Revenue")
    ax2.set_xticklabels(product_monthly.index, rotation=90)
    ax2.grid(True)
    st.pyplot(fig2)

    st.success(f"""
🔥 Peak Month for {product_choice}: {best_month}
💰 Revenue: ₹{max_value:,.2f}
""")

# ================= FORECASTING =================
def forecasting_section(df):

    st.subheader("📊 Prophet Forecast")

    revenue_trend = df.groupby("OrderDate")["Revenue"].sum().reset_index()

    revenue_trend.columns=["ds","y"]

    if len(revenue_trend)>20:

        model = Prophet()

        model.fit(revenue_trend)

        future = model.make_future_dataframe(periods=30)

        forecast = model.predict(future)

        fig = model.plot(forecast)

        st.pyplot(fig)
        future_only = forecast.tail(30)
        best_day = future_only.loc[future_only["yhat"].idxmax()]

        st.success(f"""
📈 Highest Forecasted Sales Day: {best_day['ds'].date()}
💰 Predicted Revenue: ₹{best_day['yhat']:,.2f}
""")

    else:
        st.warning("Need at least 20 rows for forecasting")

    # ================= LINEAR REGRESSION =================
    st.subheader("📈 Linear Regression Forecast with Insights")

    if len(revenue_trend) > 10:

        revenue_trend["date_ordinal"] = revenue_trend["ds"].map(pd.Timestamp.toordinal)

        X = revenue_trend[["date_ordinal"]]
        y = revenue_trend["y"]

        lr = LinearRegression()
        lr.fit(X, y)

        last_date = revenue_trend["ds"].max()
        future_dates = pd.date_range(start=last_date, periods=30)
        future_ord = future_dates.map(pd.Timestamp.toordinal)

        predictions = lr.predict(np.array(future_ord).reshape(-1, 1))

        fig2, ax = plt.subplots(figsize=(12,5))
        ax.plot(revenue_trend["ds"], y, label="Historical")
        ax.plot(future_dates, predictions, linestyle="--", label="Predicted")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig2)

        best_future_day = future_dates[np.argmax(predictions)]
        max_prediction = max(predictions)

        st.success(f"""
🚀 Highest Predicted Sales (Next 30 Days):
📅 {best_future_day.date()}
💰 ₹{max_prediction:,.2f}
""")


# ================= EXCEL REPORT =================
def generate_excel_report(df):

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:

        df.to_excel(writer, sheet_name="Raw Data", index=False)

        product_sales = df.groupby("ProductName")["Revenue"].sum().reset_index()
        product_sales.to_excel(writer, sheet_name="Product Sales", index=False)

        country_sales = df.groupby("Country")["Revenue"].sum().reset_index()
        country_sales.to_excel(writer, sheet_name="Country Sales", index=False)

        profit = df.groupby("ProductName")["Profit"].sum().reset_index()
        profit.to_excel(writer, sheet_name="Profit Analysis", index=False)

    return output.getvalue()

# ================= PDF REPORT =================
def generate_pdf_report(df):

    buffer = io.BytesIO()

    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph("AI Sales Analyzer Report", styles['Title']))

    elements.append(Spacer(1,20))

    total_revenue = df["Revenue"].sum()
    total_profit = df["Profit"].sum()

    elements.append(Paragraph(f"Total Revenue: ₹{total_revenue:,.2f}", styles['Normal']))
    elements.append(Paragraph(f"Total Profit: ₹{total_profit:,.2f}", styles['Normal']))

    elements.append(Spacer(1,20))

    product_sales = df.groupby("ProductName")["Revenue"].sum().reset_index()

    table_data=[["Product","Revenue"]]

    for i,row in product_sales.iterrows():
        table_data.append([row["ProductName"], f"₹{row['Revenue']:,.2f}"])

    table = Table(table_data)

    elements.append(table)

    doc = SimpleDocTemplate(buffer)

    doc.build(elements)

    return buffer.getvalue()

# ================= ADMIN DASHBOARD =================
def admin_dashboard():

    st.sidebar.subheader("🛠 Admin Panel")

    conn = sqlite3.connect("sales.db")
    c = conn.cursor()

    # Total Users
    c.execute("SELECT COUNT(*) FROM users")
    total_users = c.fetchone()[0]

    # Total uploads
    c.execute("SELECT COUNT(*) FROM system_logs")
    total_uploads = c.fetchone()[0]

    # Total sales records processed
    c.execute("SELECT SUM(records) FROM system_logs")
    total_records = c.fetchone()[0]

    if total_records is None:
        total_records = 0

    col1, col2, col3 = st.columns(3)

    col1.metric("👥 Total Users", total_users)
    col2.metric("📂 Total Uploads", total_uploads)
    col3.metric("📊 Records Processed", total_records)

    st.divider()

    # Recent Users
    st.subheader("👤 Registered Users")

    users_df = pd.read_sql_query(
        "SELECT id, username FROM users ORDER BY id DESC",
        conn
    )

    st.dataframe(users_df)

    st.divider()

    # Upload History
    st.subheader("📊 Dataset Upload History")

    uploads_df = pd.read_sql_query(
        "SELECT username, action, records, log_time FROM system_logs ORDER BY log_time DESC",
        conn
    )

    st.dataframe(uploads_df)

    conn.close()

# ================= DASHBOARD =================
def dashboard():

    st.title("📊 AI Sales Dashboard")

    st.write(f"Welcome {st.session_state.user} 👋")

    if st.session_state.user == "admin":
        st.subheader("⚙ System Monitoring")
        admin_dashboard()

    file = st.file_uploader("Upload Dataset", type=["csv"])

    if file:

        df = load_data(file)

        # Save dataset activity
        conn = sqlite3.connect("sales.db")
        c = conn.cursor()

        c.execute(
            "INSERT INTO system_logs (username, action, records) VALUES (?,?,?)",
            (st.session_state.user, "dataset_upload", len(df))
        )
        conn.commit()
        conn.close()

        # Filters
        st.sidebar.subheader("Filters")

        country_filter = st.sidebar.multiselect("Select Country", df["Country"].unique())
        product_filter = st.sidebar.multiselect("Select Product", df["ProductName"].unique())

        if country_filter:
            df = df[df["Country"].isin(country_filter)]

        if product_filter:
            df = df[df["ProductName"].isin(product_filter)]

        # KPIs
        kpi_section(df)

        # Menu
        menu = st.sidebar.radio(
            "Select Option",
            ["Basic Info","Charts","Prediction","Forecasting","Profit Analysis"]
        )

        if menu=="Basic Info":
            basic_info(df)

        elif menu=="Charts":
            charts_section(df)

        elif menu=="Prediction":
            prediction_section(df)

        elif menu=="Forecasting":
            forecasting_section(df)

        elif menu=="Profit Analysis":
            profit_analysis(df)

            # Reports
            st.subheader("📑 Reports")

            excel_data = generate_excel_report(df)

            st.download_button(
                label="📥 Download Excel Report",
                data=excel_data,
                file_name="sales_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            pdf_data = generate_pdf_report(df)

            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_data,
                file_name="sales_report.pdf",
                mime="application/pdf"
            )

            # Download processed data
            csv = df.to_csv(index=False)

            st.download_button(
                label="📥 Download Processed Data",
                data=csv,
                file_name="processed_sales.csv",
                mime="text/csv"
            )
            
        

        st.sidebar.button("Logout", on_click=logout)

# ================= LOGOUT =================
def logout():
    st.session_state.logged=False
    st.session_state.user=""
    st.rerun()

# ================= MAIN =================
if not st.session_state.logged:
    login_page()
else:
    dashboard()