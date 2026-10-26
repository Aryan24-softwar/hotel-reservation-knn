# coding: utf-8
"""
Generate synthetic Hotel Reservation dataset
Features: Lead time, room type, deposit, market segment, etc.
Target: is_canceled (0=Not Cancelled, 1=Cancelled)
Major Project - Hotel Reservation System using KNN Classifier
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 5000

def generate_dataset(n=N):
    hotel_type      = np.random.choice(["City Hotel","Resort Hotel"], n, p=[0.66,0.34])
    lead_time       = np.random.randint(0, 365, n)
    arrival_month   = np.random.choice(["January","February","March","April","May","June",
                                        "July","August","September","October","November","December"], n)
    arrival_week    = np.random.randint(1, 54, n)
    arrival_day     = np.random.randint(1, 32, n)
    weekend_nights  = np.random.randint(0, 5, n)
    week_nights     = np.random.randint(0, 9, n)
    adults          = np.random.choice([1,2,3,4], n, p=[0.20,0.60,0.15,0.05])
    children        = np.random.choice([0,1,2,3], n, p=[0.75,0.15,0.08,0.02])
    babies          = np.random.choice([0,1,2],   n, p=[0.94,0.05,0.01])
    meal            = np.random.choice(["BB","HB","FB","SC"], n, p=[0.58,0.22,0.08,0.12])
    market_segment  = np.random.choice(["Online TA","Offline TA/TO","Direct","Corporate",
                                        "Groups","Complementary","Aviation"],
                                        n, p=[0.47,0.20,0.15,0.07,0.06,0.03,0.02])
    distribution    = np.random.choice(["TA/TO","Direct","Corporate","GDS","Undefined"],
                                        n, p=[0.68,0.14,0.09,0.07,0.02])
    is_repeated     = np.random.choice([0,1], n, p=[0.97,0.03])
    prev_cancel     = np.random.choice([0,1,2,3], n, p=[0.87,0.08,0.03,0.02])
    prev_not_cancel = np.random.randint(0, 10, n)
    reserved_room   = np.random.choice(["A","B","C","D","E","F","G","H"], n,
                                        p=[0.40,0.10,0.05,0.22,0.10,0.05,0.05,0.03])
    assigned_room   = np.random.choice(["A","B","C","D","E","F","G","H"], n,
                                        p=[0.38,0.11,0.06,0.21,0.10,0.06,0.05,0.03])
    booking_changes = np.random.choice([0,1,2,3,4,5], n, p=[0.55,0.25,0.10,0.05,0.03,0.02])
    deposit_type    = np.random.choice(["No Deposit","Non Refund","Refundable"],
                                        n, p=[0.87,0.10,0.03])
    days_waitlist   = np.random.choice(
                        np.concatenate([np.zeros(4000, dtype=int), np.arange(1, 1001)]),
                        n, replace=True)
    customer_type   = np.random.choice(["Transient","Contract","Transient-Party","Group"],
                                        n, p=[0.75,0.10,0.12,0.03])
    adr             = np.clip(np.random.normal(100, 50, n), 0, 500).round(2)
    parking         = np.random.choice([0,1,2], n, p=[0.92,0.07,0.01])
    special_req     = np.random.choice([0,1,2,3,4,5], n, p=[0.42,0.28,0.16,0.08,0.04,0.02])
    total_nights    = weekend_nights + week_nights

    # Cancellation probability (realistic logic)
    prob = np.full(n, 0.30)
    prob += (lead_time > 150)              * 0.15
    prob += (deposit_type == "Non Refund") * 0.25
    prob += (prev_cancel >= 1)             * 0.20
    prob += (market_segment == "Groups")   * 0.10
    prob += (days_waitlist > 50)           * 0.08
    prob += (special_req == 0)             * 0.05
    prob -= (is_repeated == 1)             * 0.15
    prob -= (prev_not_cancel > 3)          * 0.10
    prob -= (booking_changes >= 1)         * 0.05
    prob -= (deposit_type == "No Deposit") * 0.05
    prob -= (customer_type == "Contract")  * 0.10
    prob -= (special_req >= 3)             * 0.08
    prob = np.clip(prob, 0.02, 0.97)

    is_canceled = (np.random.rand(n) < prob).astype(int)
    room_type_match = (reserved_room == assigned_room).astype(int)

    df = pd.DataFrame({
        "hotel":                     hotel_type,
        "lead_time":                 lead_time,
        "arrival_date_month":        arrival_month,
        "arrival_date_week_number":  arrival_week,
        "arrival_date_day_of_month": arrival_day,
        "stays_in_weekend_nights":   weekend_nights,
        "stays_in_week_nights":      week_nights,
        "total_nights":              total_nights,
        "adults":                    adults,
        "children":                  children,
        "babies":                    babies,
        "meal":                      meal,
        "market_segment":            market_segment,
        "distribution_channel":      distribution,
        "is_repeated_guest":         is_repeated,
        "previous_cancellations":    prev_cancel,
        "previous_bookings_not_canceled": prev_not_cancel,
        "reserved_room_type":        reserved_room,
        "assigned_room_type":        assigned_room,
        "room_type_match":           room_type_match,
        "booking_changes":           booking_changes,
        "deposit_type":              deposit_type,
        "days_in_waiting_list":      days_waitlist,
        "customer_type":             customer_type,
        "adr":                       adr,
        "required_car_parking_spaces": parking,
        "total_of_special_requests": special_req,
        "is_canceled":               is_canceled,
    })
    return df

if __name__ == "__main__":
    df = generate_dataset()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "hotel_reservations.csv")
    df.to_csv(out, index=False)
    print(f"Dataset saved -> {out}")
    print(f"Shape: {df.shape}")
    print(f"\nCancellation Rate:\n{df['is_canceled'].value_counts()}")
    print(f"\nCancellation %: {df['is_canceled'].mean()*100:.1f}%")
    print(df.head())