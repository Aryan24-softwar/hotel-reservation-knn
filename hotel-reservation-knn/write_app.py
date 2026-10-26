# coding: utf-8
"""Helper: writes app.py for hotel-reservation-knn project"""
import os

APP = r'c:\Users\Lenovo\OneDrive\Desktop\VS CODE\Assistant\hotel-reservation-knn\app.py'

code = """\
# coding: utf-8
import os, pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title='Hotel Reservation - KNN', page_icon='H', layout='wide', initial_sidebar_state='expanded')

BASE   = os.path.dirname(os.path.abspath(__file__))
MODEL  = os.path.join(BASE, 'models')
DATA   = os.path.join(BASE, 'data', 'hotel_reservations.csv')
PLOTS  = os.path.join(BASE, 'plots')

@st.cache_resource
def load_all():
    knn      = pickle.load(open(os.path.join(MODEL, 'knn_model.pkl'), 'rb'))
    scaler   = pickle.load(open(os.path.join(MODEL, 'scaler.pkl'), 'rb'))
    encoders = pickle.load(open(os.path.join(MODEL, 'encoders.pkl'), 'rb'))
    best_k   = pickle.load(open(os.path.join(MODEL, 'best_k.pkl'), 'rb'))
    res_df   = pickle.load(open(os.path.join(MODEL, 'results_df.pkl'), 'rb'))
    return knn, scaler, encoders, best_k, res_df

@st.cache_data
def load_data():
    return pd.read_csv(DATA)

try:
    knn, scaler, encoders, best_k, res_df = load_all()
    df   = load_data()
    ready = True
except Exception as exc:
    ready = False

HEADER = ('<div style="background:linear-gradient(135deg,#0d47a1,#1976d2);'
          'padding:2rem;border-radius:16px;margin-bottom:1.5rem;text-align:center">'
          '<h1 style="color:white;font-size:2.4rem;margin:0">Hotel Reservation System</h1>'
          '<p style="color:#bbdefb;margin:0.4rem 0 0">Cancellation Prediction using '
          '<strong>KNN Classifier</strong> | Major Project</p></div>')
st.markdown(HEADER, unsafe_allow_html=True)

if not ready:
    st.error('Models not found! Run: python generate_dataset.py  then  python train_model.py')
    st.stop()

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('## New Reservation')
    st.markdown('---')
    hotel          = st.selectbox('Hotel Type', ['City Hotel', 'Resort Hotel'])
    lead_time      = st.slider('Lead Time (days before arrival)', 0, 365, 30)
    arrival_month  = st.selectbox('Arrival Month', ['January','February','March','April','May','June','July','August','September','October','November','December'])
    arrival_week   = st.slider('Arrival Week Number', 1, 53, 25)
    arrival_day    = st.slider('Arrival Day of Month', 1, 31, 15)
    weekend_nights = st.slider('Weekend Nights', 0, 4, 1)
    week_nights    = st.slider('Week Nights', 0, 8, 2)
    st.markdown('---')
    adults   = st.selectbox('Adults', [1, 2, 3, 4])
    children = st.selectbox('Children', [0, 1, 2, 3])
    babies   = st.selectbox('Babies', [0, 1, 2])
    st.markdown('---')
    meal           = st.selectbox('Meal Plan', ['BB', 'HB', 'FB', 'SC'], help='BB=Bed & Breakfast, HB=Half Board, FB=Full Board, SC=Self Catering')
    market_segment = st.selectbox('Market Segment', ['Online TA','Offline TA/TO','Direct','Corporate','Groups','Complementary','Aviation'])
    distribution   = st.selectbox('Distribution Channel', ['TA/TO','Direct','Corporate','GDS','Undefined'])
    deposit_type   = st.selectbox('Deposit Type', ['No Deposit','Non Refund','Refundable'])
    customer_type  = st.selectbox('Customer Type', ['Transient','Contract','Transient-Party','Group'])
    reserved_room  = st.selectbox('Reserved Room Type', ['A','B','C','D','E','F','G','H'])
    assigned_room  = st.selectbox('Assigned Room Type', ['A','B','C','D','E','F','G','H'])
    booking_changes= st.slider('Booking Changes', 0, 5, 0)
    special_req    = st.slider('Special Requests', 0, 5, 1)
    parking        = st.selectbox('Car Parking Spaces', [0, 1, 2])
    adr            = st.slider('Average Daily Rate', 0, 500, 100)
    st.markdown('---')
    is_repeated    = st.selectbox('Repeated Guest', [0, 1], format_func=lambda x: 'Yes' if x else 'No')
    prev_cancel    = st.slider('Previous Cancellations', 0, 5, 0)
    prev_not_cancel= st.slider('Prev Bookings Not Cancelled', 0, 10, 0)
    days_waitlist  = st.slider('Days in Waiting List', 0, 200, 0)

def build_input():
    room_match   = int(reserved_room == assigned_room)
    total_nights = weekend_nights + week_nights
    row = [
        lead_time, weekend_nights, week_nights, total_nights,
        adults, children, babies, is_repeated, prev_cancel, prev_not_cancel,
        room_match, booking_changes, days_waitlist, adr, parking, special_req,
        arrival_week, arrival_day,
        encoders['hotel'].transform([hotel])[0],
        encoders['arrival_date_month'].transform([arrival_month])[0],
        encoders['meal'].transform([meal])[0],
        encoders['market_segment'].transform([market_segment])[0],
        encoders['distribution_channel'].transform([distribution])[0],
        encoders['reserved_room_type'].transform([reserved_room])[0],
        encoders['assigned_room_type'].transform([assigned_room])[0],
        encoders['deposit_type'].transform([deposit_type])[0],
        encoders['customer_type'].transform([customer_type])[0],
    ]
    return np.array(row).reshape(1, -1)

tab1, tab2, tab3, tab4 = st.tabs(['Prediction', 'EDA and Insights', 'KNN Analysis', 'Model Comparison'])

# ── TAB 1: PREDICTION ────────────────────────────────────────────────────────
with tab1:
    X_in = scaler.transform(build_input())
    prob = knn.predict_proba(X_in)[0][1]
    pred = knn.predict(X_in)[0]
    distances, indices = knn.kneighbors(X_in)
    avg_dist = distances[0].mean()

    if prob < 0.35:
        risk, bg, col = 'LOW RISK',    '#E8F5E9', '#2E7D32'
        advice = 'Reservation is likely to be honoured. Low cancellation risk.'
    elif prob < 0.65:
        risk, bg, col = 'MEDIUM RISK', '#FFF3E0', '#E65100'
        advice = 'Moderate risk. Consider sending a confirmation reminder to the guest.'
    else:
        risk, bg, col = 'HIGH RISK',   '#FFEBEE', '#B71C1C'
        advice = 'High cancellation risk! Consider requiring a non-refundable deposit.'

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        card = (f'<div style="background:{bg};border:2px solid {col};border-radius:16px;'
                f'padding:2rem;text-align:center;box-shadow:0 6px 20px rgba(0,0,0,0.1)">'
                f'<div style="font-size:1rem;color:#555">Cancellation Probability</div>'
                f'<div style="font-size:3.8rem;font-weight:900;color:{col}">{prob*100:.1f}%</div>'
                f'<div style="background:{col};color:white;border-radius:25px;display:inline-block;'
                f'padding:0.4rem 1.4rem;font-weight:700;margin-top:0.4rem">{risk}</div>'
                f'<div style="color:{col};margin-top:0.7rem">{advice}</div></div>')
        st.markdown(card, unsafe_allow_html=True)

    st.markdown('---')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Prediction', 'Will Cancel' if pred == 1 else 'Will NOT Cancel')
    c2.metric('Probability', f'{prob*100:.1f}%')
    c3.metric('KNN K Value', f'K = {best_k}')
    c4.metric('Avg Neighbor Distance', f'{avg_dist:.3f}')

    st.markdown('---')
    st.markdown('### Risk Meter')
    fig, ax = plt.subplots(figsize=(10, 1.8))
    for lo, hi, c, lbl in [(0, 0.35, '#2E7D32', 'Low Risk'), (0.35, 0.65, '#E65100', 'Medium Risk'), (0.65, 1.0, '#B71C1C', 'High Risk')]:
        ax.barh(0, hi - lo, left=lo, height=0.5, color=c, alpha=0.85)
        ax.text((lo + hi) / 2, 0, lbl, ha='center', va='center', color='white', fontsize=9, fontweight='bold')
    ax.axvline(prob, color='black', lw=3, zorder=5)
    ax.annotate(f'  {prob*100:.1f}%', xy=(prob, 0.27), fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1); ax.set_ylim(-0.4, 0.4); ax.axis('off')
    fig.patch.set_facecolor('#f8f9fa')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('---')
    st.markdown('### Booking Summary')
    c1, c2 = st.columns(2)
    with c1:
        st.write(f'**Hotel:** {hotel}')
        st.write(f'**Lead Time:** {lead_time} days')
        st.write(f'**Total Nights:** {weekend_nights + week_nights}')
        st.write(f'**Guests:** {adults} Adults, {children} Children, {babies} Babies')
        st.write(f'**Meal Plan:** {meal}')
    with c2:
        st.write(f'**Market Segment:** {market_segment}')
        st.write(f'**Deposit Type:** {deposit_type}')
        st.write(f'**Room Reserved:** {reserved_room}  |  Assigned: {assigned_room}')
        st.write(f'**ADR:** {adr}')
        st.write(f'**Special Requests:** {special_req}')

    st.markdown('---')
    st.markdown('### Management Recommendations')
    recs = []
    if deposit_type == 'No Deposit' and prob > 0.5:
        recs.append('Consider requiring a deposit for this high-risk booking')
    if lead_time > 150:
        recs.append('Long lead time detected - send reminder 30 days before arrival')
    if prev_cancel >= 1:
        recs.append('Guest has prior cancellations - flag for closer monitoring')
    if special_req == 0 and prob > 0.4:
        recs.append('No special requests made - follow up to confirm commitment')
    if market_segment == 'Groups':
        recs.append('Group bookings carry higher cancellation risk - ensure contract is signed')
    if reserved_room != assigned_room:
        recs.append('Room type mismatch detected - try to honour reserved room type')
    if is_repeated == 0 and prob > 0.5:
        recs.append('New guest with high risk - consider overbooking buffer for this date')
    if not recs:
        recs.append('Booking looks reliable. No special action needed.')
    for r in recs:
        st.write(f'- {r}')

# ── TAB 2: EDA ───────────────────────────────────────────────────────────────
with tab2:
    st.markdown('### Dataset Overview')
    total  = len(df)
    cancel = df['is_canceled'].sum()
    stay   = total - cancel
    rate   = cancel / total * 100
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Total Reservations', f'{total:,}')
    c2.metric('Not Cancelled', f'{stay:,}')
    c3.metric('Cancelled', f'{cancel:,}')
    c4.metric('Cancellation Rate', f'{rate:.1f}%')
    st.markdown('---')
    if os.path.exists(PLOTS):
        eda_plots = sorted([f for f in os.listdir(PLOTS) if f.endswith('.png') and not f.startswith(('01','11','12','13','14'))])
        titles = {
            '02_cancellation_distribution.png': 'Cancellation Distribution',
            '03_hotel_type_cancellation.png':   'By Hotel Type',
            '04_lead_time_distribution.png':    'Lead Time Distribution',
            '05_deposit_cancellation.png':      'By Deposit Type',
            '06_market_segment_cancellation.png': 'By Market Segment',
            '07_adr_vs_cancellation.png':       'ADR vs Cancellation',
            '08_special_requests_cancellation.png': 'Special Requests',
            '09_monthly_cancellation.png':      'Monthly Pattern',
            '10_correlation_heatmap.png':       'Correlation Heatmap',
            '15_customer_type_cancellation.png':'By Customer Type',
        }
        cols = st.columns(2)
        for i, pf in enumerate(eda_plots):
            cols[i % 2].markdown(f'#### {titles.get(pf, pf)}')
            cols[i % 2].image(os.path.join(PLOTS, pf), use_container_width=True)
    st.markdown('---')
    st.markdown('#### Sample Dataset (First 20 rows)')
    st.dataframe(df.head(20), use_container_width=True)

# ── TAB 3: KNN ANALYSIS ──────────────────────────────────────────────────────
with tab3:
    st.markdown(f'### KNN Model Analysis (Best K = {best_k})')
    c1, c2, c3 = st.columns(3)
    c1.metric('Best K Value', str(best_k))
    c2.metric('Distance Metric', 'Euclidean (Minkowski p=2)')
    c3.metric('Weight Function', 'Uniform')
    st.markdown('---')
    st.markdown('#### How KNN Works for Hotel Reservations')
    st.markdown(
        f'1. Take new reservation as input (27 features)\\n'
        f'2. Scale all features using StandardScaler\\n'
        f'3. Find **{best_k} nearest reservations** in training data using Euclidean distance\\n'
        f'4. Majority class among K neighbors decides: Cancel or Not Cancel\\n'
        f'5. Probability = fraction of neighbors that are Cancelled'
    )
    st.markdown('---')
    c1, c2 = st.columns(2)
    with c1:
        p = os.path.join(PLOTS, '01_knn_k_selection.png')
        if os.path.exists(p):
            st.markdown('#### K Value Selection (5-Fold CV)')
            st.image(p, use_container_width=True)
    with c2:
        p = os.path.join(PLOTS, '14_knn_cross_validation.png')
        if os.path.exists(p):
            st.markdown('#### 10-Fold Cross-Validation Accuracy')
            st.image(p, use_container_width=True)
    st.markdown('---')
    c1, c2 = st.columns(2)
    with c1:
        p = os.path.join(PLOTS, '12_knn_confusion_matrix.png')
        if os.path.exists(p):
            st.markdown('#### Confusion Matrix')
            st.image(p, use_container_width=True)
    with c2:
        p = os.path.join(PLOTS, '13_roc_curves.png')
        if os.path.exists(p):
            st.markdown('#### ROC Curves (All Models)')
            st.image(p, use_container_width=True)
    st.markdown('---')
    st.markdown('#### KNN Algorithm Pseudocode')
    st.code(
        'ALGORITHM KNN_Predict(new_reservation, K):\\n'
        '  1. Scale features using StandardScaler\\n'
        '  2. Compute Euclidean distance to all training samples:\\n'
        '         dist(x, xi) = sqrt(sum((x_j - xi_j)^2))\\n'
        '  3. Sort training samples by distance (ascending)\\n'
        '  4. Select K nearest neighbors\\n'
        '  5. cancel_votes = count(neighbors where is_canceled = 1)\\n'
        '  6. Prediction  = majority vote class\\n'
        '  7. Probability = cancel_votes / K\\n'
        '  RETURN prediction, probability'
    )

# ── TAB 4: MODEL COMPARISON ──────────────────────────────────────────────────
with tab4:
    st.markdown('### Model Comparison')
    st.dataframe(res_df.reset_index(drop=True), use_container_width=True)
    st.markdown('---')
    p = os.path.join(PLOTS, '11_model_comparison.png')
    if os.path.exists(p):
        st.image(p, use_container_width=True)
    st.markdown('---')
    st.markdown('#### Why KNN for Hotel Reservations?')
    st.markdown(
        '| Reason | Explanation |\\n'
        '|--------|-------------|\\n'
        '| Interpretable | Similar past bookings predict future outcomes |\\n'
        '| No assumptions | KNN makes no data distribution assumptions |\\n'
        '| Pattern-based | Hotel bookings cluster by deposit, lead time, segment |\\n'
        '| Instance-based | Each prediction uses actual historical data points |\\n'
        '| Flexible | Works well with mixed feature types after encoding |'
    )
    st.markdown('---')
    st.markdown('#### Features Used (27 total)')
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('**Booking Info**')
        for f in ['lead_time','stays_in_weekend_nights','stays_in_week_nights','total_nights','booking_changes','days_in_waiting_list']:
            st.write(f'- {f}')
    with c2:
        st.markdown('**Guest Info**')
        for f in ['adults','children','babies','is_repeated_guest','previous_cancellations','previous_bookings_not_canceled','required_car_parking_spaces','total_of_special_requests']:
            st.write(f'- {f}')
    with c3:
        st.markdown('**Categorical (Encoded)**')
        for f in ['hotel','meal','market_segment','distribution_channel','deposit_type','customer_type','reserved_room_type','assigned_room_type','arrival_date_month']:
            st.write(f'- {f}')

st.markdown('---')
footer = ('<div style="text-align:center;color:#9e9e9e;padding:1.5rem;font-size:0.85rem;border-top:1px solid #eee">'
          'Hotel Reservation System | KNN Classifier | Major Project | 2025-26</div>')
st.markdown(footer, unsafe_allow_html=True)
"""

with open(APP, 'w', encoding='utf-8') as f:
    f.write(code)

print(f'app.py written: {os.path.getsize(APP)} bytes')
