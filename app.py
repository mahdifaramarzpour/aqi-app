import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import random
from datetime import datetime


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AQI Predictor",
    page_icon="🌫️",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("xgboost.pkl")


try:
    model = load_model()
except Exception as e:
    st.error("❌ خطا در بارگذاری فایل xgboost.pkl")
    st.exception(e)
    st.stop()


# =========================================================
# MODEL FEATURES
# =========================================================

if not hasattr(model, "feature_names_in_"):
    st.error(
        "❌ مدل ذخیره‌شده اطلاعات feature_names_in_ را ندارد."
    )
    st.stop()


MODEL_FEATURES = list(model.feature_names_in_)


# =========================================================
# HARD-CODED DATA
# =========================================================

HARDCODED_DATA = [
    {
        "pm2_5": 14.558333,
        "pm10": 24.816667,
        "co": 466.333333,
        "no2": 60.270833,
        "so2": 5.0,
        "o3": 68.541667,
        "dust": 12.583333,
        "uv_index": 2.427083,
        "aqi_today": 46.5,
        "aqi_yesterday": 47.0,
        "aqi_2days_ago": 46.152778
    },
    {
        "pm2_5": 16.245833,
        "pm10": 25.7125,
        "co": 722.541667,
        "no2": 77.1625,
        "so2": 5.0,
        "o3": 67.5,
        "dust": 7.791667,
        "uv_index": 2.504167,
        "aqi_today": 54.791667,
        "aqi_yesterday": 46.5,
        "aqi_2days_ago": 46.597222
    },
    {
        "pm2_5": 19.391667,
        "pm10": 29.658333,
        "co": 822.458333,
        "no2": 76.083333,
        "so2": 5.0,
        "o3": 74.583333,
        "dust": 6.125,
        "uv_index": 2.410417,
        "aqi_today": 58.208333,
        "aqi_yesterday": 54.791667,
        "aqi_2days_ago": 49.430556
    },
    {
        "pm2_5": 64.670833,
        "pm10": 94.8125,
        "co": 3333.833333,
        "no2": 400.570833,
        "so2": 5.0,
        "o3": 9.5,
        "dust": 5.125,
        "uv_index": 0.389583,
        "aqi_today": 111.25,
        "aqi_yesterday": 95.541667,
        "aqi_2days_ago": 85.638889
    },
    {
        "pm2_5": 18.4375,
        "pm10": 18.833333,
        "co": 2691.583333,
        "no2": 43.754167,
        "so2": 5.0,
        "o3": 24.291667,
        "dust": 0.333333,
        "uv_index": 0.50625,
        "aqi_today": 27.583333,
        "aqi_yesterday": 66.875,
        "aqi_2days_ago": 72.333333
    }
]


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def compute_features(
    pm2_5,
    pm10,
    co,
    no2,
    so2,
    o3,
    dust,
    uv_index,
    aqi_today,
    aqi_yesterday,
    aqi_2days_ago
):

    particle_pollution = pm10 + dust + 0.2

    gas_pollution = (
        o3
        + no2
        + so2
        + co
    )

    co_dot_no2 = co * no2

    trend = aqi_today - aqi_yesterday

    ma3 = (
        aqi_today
        + aqi_yesterday
        + aqi_2days_ago
    ) / 3

    return {
        "pm2_5": pm2_5,
        "pm10": pm10,
        "european_aqi": aqi_today,
        "particle_pollution": particle_pollution,
        "gas_pollution": gas_pollution,
        "carbon_monoxide": co,
        "uv_index": uv_index,
        "ozone": o3,
        "nitrogen_dioxide": no2,
        "dust": dust,
        "lag1": aqi_yesterday,
        "co dot no2": co_dot_no2,
        "trend": trend,
        "ma3": ma3
    }


# =========================================================
# PREDICTION
# =========================================================

def make_prediction(features):

    # -----------------------------------------------------
    # بررسی featureهای گمشده
    # -----------------------------------------------------

    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in features
    ]

    if missing_features:
        raise ValueError(
            "Featureهای زیر برای مدل وجود ندارند: "
            + str(missing_features)
        )

    # -----------------------------------------------------
    # خیلی مهم:
    #
    # featureهای تکراری MODEL_FEATURES را حذف نمی‌کنیم.
    #
    # اگر مدل مثلاً این را داشته باشد:
    #
    # ['ozone', 'uv_index', 'ozone', 'uv_index']
    #
    # باید دقیقاً 4 مقدار به مدل بدهیم.
    # -----------------------------------------------------

    input_values = [
        features[feature]
        for feature in MODEL_FEATURES
    ]

    # -----------------------------------------------------
    # تبدیل مستقیم به NumPy
    #
    # این کار باعث می‌شود sklearn/narwhals روی نام
    # ستون‌های تکراری DataFrame خطا ندهد.
    # -----------------------------------------------------

    X_input = np.asarray(
        [input_values],
        dtype=np.float64
    )

    # -----------------------------------------------------
    # کنترل تعداد featureها
    # -----------------------------------------------------

    expected_count = len(MODEL_FEATURES)

    actual_count = X_input.shape[1]

    if actual_count != expected_count:
        raise ValueError(
            f"تعداد featureهای ورودی {actual_count} است "
            f"اما مدل {expected_count} feature انتظار دارد."
        )

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    prediction = model.predict(X_input)

    return float(prediction[0])


# =========================================================
# ADVICE
# =========================================================

advice_map = {
    "خوب":
        "هوای عالی برای پیاده‌روی! از فضای باز لذت ببرید. 🏃‍♂️",

    "متوسط":
        "کیفیت هوا قابل قبول است، اما افراد بسیار حساس احتیاط کنند. 😷",

    "ناسالم برای حساس‌ها":
        "غلظت آلاینده‌ها کمی بالاست، افراد حساس کمتر در فضای باز بمانند. 😷",

    "ناسالم":
        "هوا ناسالم است! از تردد غیرضروری و فعالیت سنگین خودداری کنید. 🛑",

    "بسیار ناسالم":
        "وضعیت نامطلوب است. فعالیت در فضای باز را محدود کنید. ⛔"
}


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "pm2_5": 20.0,
    "pm10": 30.0,
    "co": 300.0,
    "no2": 40.0,
    "so2": 10.0,
    "o3": 50.0,
    "dust": 5.0,
    "uv_index": 1.5,
    "aqi_today": 50.0,
    "aqi_yesterday": 48.0,
    "aqi_2days_ago": 45.0
}


for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# HISTORY
# =========================================================

if "history" not in st.session_state:

    st.session_state.history = pd.DataFrame(
        columns=[
            "زمان",
            "PM2.5",
            "PM10",
            "CO",
            "NO2",
            "SO2",
            "O3",
            "Dust",
            "UV Index",
            "AQI امروز",
            "AQI دیروز",
            "AQI دو روز پیش",
            "پیش‌بینی AQI",
            "وضعیت"
        ]
    )


# =========================================================
# ABOUT
# =========================================================

if "show_modal" not in st.session_state:
    st.session_state.show_modal = True


if st.session_state.show_modal:

    with st.popover(
        "ℹ️ درباره سامانه",
        use_container_width=True
    ):

        st.markdown(
            """
            ### ℹ️ درباره سامانه پیش‌بینی

            🤖 **این سیستم با استفاده از مدل XGBoost
            و داده‌های آلایندگی روزهای گذشته،
            شاخص کیفیت هوا (AQI) را برای فردا
            پیش‌بینی می‌کند.**

            شما می‌توانید مقادیر را به صورت دستی
            وارد کنید یا با دکمه **«انتخاب تصادفی»**
            یک نمونه واقعی از داده‌های تاریخی را امتحان کنید.

            ---

            ### 📊 عملکرد مدل روی داده‌های تست

            | معیار | مقدار |
            |---|---:|
            | MAE | 3.571 |
            | RMSE | 4.719 |
            """
        )

        if st.button(
            "✓ متوجه شدم",
            use_container_width=True
        ):

            st.session_state.show_modal = False
            st.rerun()


# =========================================================
# TITLE
# =========================================================

st.markdown(
    """
    <h1 style="
        text-align:center;
        margin-bottom:5px;
    ">
        🌫️ پیش‌بینی هوشمند کیفیت هوای فردا
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="
        text-align:center;
        color:#999999;
        font-size:17px;
        margin-bottom:25px;
    ">
        سامانه پیش‌بینی شاخص کیفیت هوا با استفاده از XGBoost
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# AQI TABLE
# =========================================================

st.markdown(
    "### 📊 رنج‌های شاخص کیفیت هوا (AQI)"
)

st.markdown(
    """
    <table style="
        width:100%;
        text-align:center;
        border-collapse:collapse;
        overflow:hidden;
        border-radius:12px;
        font-size:17px;
    ">

        <tr style="
            background-color:#333333;
            color:white;
        ">
            <th style="padding:12px;">
                وضعیت
            </th>

            <th style="padding:12px;">
                بازه AQI
            </th>
        </tr>

        <tr style="
            background-color:#00e400;
            color:black;
        ">
            <td style="padding:10px;">
                خوب
            </td>

            <td style="padding:10px;">
                0 – 50
            </td>
        </tr>

        <tr style="
            background-color:#ffff00;
            color:black;
        ">
            <td style="padding:10px;">
                متوسط
            </td>

            <td style="padding:10px;">
                51 – 100
            </td>
        </tr>

        <tr style="
            background-color:#ff7e00;
            color:white;
        ">
            <td style="padding:10px;">
                ناسالم برای حساس‌ها
            </td>

            <td style="padding:10px;">
                101 – 150
            </td>
        </tr>

        <tr style="
            background-color:#ff0000;
            color:white;
        ">
            <td style="padding:10px;">
                ناسالم
            </td>

            <td style="padding:10px;">
                151 – 200
            </td>
        </tr>

        <tr style="
            background-color:#800080;
            color:white;
        ">
            <td style="padding:10px;">
                بسیار ناسالم
            </td>

            <td style="padding:10px;">
                201 – 300
            </td>
        </tr>

    </table>
    """,
    unsafe_allow_html=True
)


st.markdown("---")


# =========================================================
# AIR QUALITY INPUTS
# =========================================================

st.subheader(
    "📊 متغیرهای کیفیت هوا (امروز)"
)

cols1 = st.columns(8)


with cols1[0]:

    st.number_input(
        "PM2.5",
        key="pm2_5",
        step=1.0,
        format="%.2f"
    )


with cols1[1]:

    st.number_input(
        "PM10",
        key="pm10",
        step=1.0,
        format="%.2f"
    )


with cols1[2]:

    st.number_input(
        "CO",
        key="co",
        step=10.0,
        format="%.2f"
    )


with cols1[3]:

    st.number_input(
        "NO₂",
        key="no2",
        step=1.0,
        format="%.2f"
    )


with cols1[4]:

    st.number_input(
        "SO₂",
        key="so2",
        step=1.0,
        format="%.2f"
    )


with cols1[5]:

    st.number_input(
        "O₃",
        key="o3",
        step=1.0,
        format="%.2f"
    )


with cols1[6]:

    st.number_input(
        "Dust",
        key="dust",
        step=1.0,
        format="%.2f"
    )


with cols1[7]:

    st.number_input(
        "UV Index",
        key="uv_index",
        step=0.1,
        format="%.2f"
    )


# =========================================================
# PREVIOUS AQI
# =========================================================

st.subheader(
    "📅 شاخص AQI روزهای گذشته"
)

cols2 = st.columns(3)


with cols2[0]:

    st.number_input(
        "AQI امروز (european_aqi)",
        key="aqi_today",
        step=1.0,
        format="%.2f"
    )


with cols2[1]:

    st.number_input(
        "AQI دیروز (lag1)",
        key="aqi_yesterday",
        step=1.0,
        format="%.2f"
    )


with cols2[2]:

    st.number_input(
        "AQI دو روز پیش (lag2)",
        key="aqi_2days_ago",
        step=1.0,
        format="%.2f"
    )


st.markdown("---")


# =========================================================
# BUTTONS
# =========================================================

col_btn1, col_btn2, col_btn3 = st.columns(3)


with col_btn1:

    random_clicked = st.button(
        "🎲 انتخاب تصادفی از دیتاست",
        use_container_width=True
    )


with col_btn2:

    predict_clicked = st.button(
        "🚀 اجرای پیش‌بینی",
        use_container_width=True
    )


# =========================================================
# RANDOM DATA
# =========================================================

if random_clicked:

    random_data = random.choice(
        HARDCODED_DATA
    )

    for key, value in random_data.items():

        st.session_state[key] = value

    st.rerun()


# =========================================================
# PREDICTION
# =========================================================

if predict_clicked:

    # -----------------------------------------------------
    # Compute features
    # -----------------------------------------------------

    features = compute_features(
        pm2_5=st.session_state.pm2_5,
        pm10=st.session_state.pm10,
        co=st.session_state.co,
        no2=st.session_state.no2,
        so2=st.session_state.so2,
        o3=st.session_state.o3,
        dust=st.session_state.dust,
        uv_index=st.session_state.uv_index,
        aqi_today=st.session_state.aqi_today,
        aqi_yesterday=st.session_state.aqi_yesterday,
        aqi_2days_ago=st.session_state.aqi_2days_ago
    )

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    try:

        pred = make_prediction(
            features
        )

    except Exception as e:

        st.error(
            "❌ خطا هنگام اجرای مدل"
        )

        st.exception(e)

        with st.expander(
            "🔍 اطلاعات فنی مدل"
        ):

            st.write(
                "تعداد featureهای مدل:",
                len(MODEL_FEATURES)
            )

            st.write(
                "Featureهای مدل به ترتیب:"
            )

            st.write(
                MODEL_FEATURES
            )

            st.write(
                "تعداد featureهای تولیدشده:",
                len(features)
            )

            st.write(
                "Featureهای تولیدشده:"
            )

            st.write(
                list(features.keys())
            )

        st.stop()


    # -----------------------------------------------------
    # AQI status
    # -----------------------------------------------------

    if pred <= 50:

        status = "خوب"
        color = "#00e400"
        msg = "هوای بسیار پاک و عالی."

    elif pred <= 100:

        status = "متوسط"
        color = "#ffff00"
        msg = (
            "کیفیت قابل قبول است، "
            "افراد حساس احتیاط کنند."
        )

    elif pred <= 150:

        status = "ناسالم برای حساس‌ها"
        color = "#ff7e00"
        msg = (
            "گروه‌های حساس کمتر "
            "در فضای باز بمانند."
        )

    elif pred <= 200:

        status = "ناسالم"
        color = "#ff0000"
        msg = (
            "هوا ناسالم است. "
            "از فعالیت سنگین خودداری کنید."
        )

    else:

        status = "بسیار ناسالم"
        color = "#800080"
        msg = (
            "وضعیت نامطلوب است. "
            "فعالیت در فضای باز را محدود کنید."
        )


    # =====================================================
    # GAUGE
    # =====================================================

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pred,

            number={
                "font": {
                    "size": 45
                }
            },

            gauge={
                "axis": {
                    "range": [0, 300]
                },

                "bar": {
                    "color": color
                },

                "steps": [

                    {
                        "range": [0, 50],
                        "color": "#00e400"
                    },

                    {
                        "range": [50, 100],
                        "color": "#ffff00"
                    },

                    {
                        "range": [100, 150],
                        "color": "#ff7e00"
                    },

                    {
                        "range": [150, 200],
                        "color": "#ff0000"
                    },

                    {
                        "range": [200, 300],
                        "color": "#800080"
                    }
                ]
            }
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # =====================================================
    # STATUS CARD
    # =====================================================

    text_color = (
        "black"
        if pred <= 100
        else "white"
    )

    status_html = f"""
<div style="
    background-color:{color};
    padding:25px;
    border-radius:15px;
    color:{text_color};
    text-align:center;
    margin-top:15px;
    margin-bottom:20px;
">

    <h2 style="
        margin:0;
        font-size:40px;
        font-weight:bold;
    ">
        وضعیت: {status} — AQI: {pred:.1f}
    </h2>

    <p style="
        font-size:26px;
        font-weight:bold;
        margin-top:15px;
        margin-bottom:0;
    ">
        {msg}
    </p>

</div>
"""

    st.markdown(
        status_html,
        unsafe_allow_html=True
    )


    # =====================================================
    # ADVICE
    # =====================================================

    advice = advice_map[status]

    advice_html = f"""
<div style="
    margin-top:20px;
    padding:18px;
    border-left:6px solid #1f77b4;
    border-radius:10px;
    background-color:#f0f8ff;
    text-align:center;
">

    <p style="
        font-size:24px;
        color:#333333;
        font-weight:bold;
        margin:0;
    ">
        💡 توصیه هوشمند: {advice}
    </p>

</div>
"""

    st.markdown(
        advice_html,
        unsafe_allow_html=True
    )


    # =====================================================
    # SAVE HISTORY
    # =====================================================

    new_record = pd.DataFrame(
        [
            {
                "زمان":
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "PM2.5":
                    st.session_state.pm2_5,

                "PM10":
                    st.session_state.pm10,

                "CO":
                    st.session_state.co,

                "NO2":
                    st.session_state.no2,

                "SO2":
                    st.session_state.so2,

                "O3":
                    st.session_state.o3,

                "Dust":
                    st.session_state.dust,

                "UV Index":
                    st.session_state.uv_index,

                "AQI امروز":
                    st.session_state.aqi_today,

                "AQI دیروز":
                    st.session_state.aqi_yesterday,

                "AQI دو روز پیش":
                    st.session_state.aqi_2days_ago,

                "پیش‌بینی AQI":
                    round(pred, 2),

                "وضعیت":
                    status
            }
        ]
    )


    st.session_state.history = pd.concat(
        [
            st.session_state.history,
            new_record
        ],
        ignore_index=True
    )


# =========================================================
# HISTORY
# =========================================================

st.markdown("---")

st.subheader(
    "📜 تاریخچه پیش‌بینی‌ها"
)


if not st.session_state.history.empty:

    st.dataframe(
        st.session_state.history,
        use_container_width=True,
        height=300
    )

else:

    st.info(
        "هنوز پیش‌بینی انجام نشده است. "
        "با کلیک روی دکمه «اجرای پیش‌بینی» "
        "اولین رکورد ثبت می‌شود."
    )
