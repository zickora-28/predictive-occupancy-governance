import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. SIMULATE DATA (In a real project, you'd use library logs)
# Features: [Hour (0-23), Is_Exam_Season (0 or 1), Day_of_Week (0-6)]
data = {
    'hour': np.random.randint(0, 24, 1000),
    'is_exam_season': np.random.randint(0, 2, 1000),
    'day_of_week': np.random.randint(0, 7, 1000)
}
df = pd.DataFrame(data)

# Target: 1 = Full (Occupied), 0 = Available
# Logic: If it's exam season AND between 10am-4pm, it's likely Full
df['is_full'] = ((df['is_exam_season'] == 1) & (df['hour'].between(10, 16))).astype(int)

# 2. PREPARE MODEL
X = df[['hour', 'is_exam_season', 'day_of_week']]
y = df['is_full']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. TRAIN THE AI
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. PREDICTIVE FUNCTION FOR THE APP
def get_prediction(hour, exam_season, day):
    # Returns probability of being "Full"
    prob = model.predict_proba([[hour, exam_season, day]])[0][1]
    status = "FULL (Red)" if prob > 0.5 else "AVAILABLE (Green)"
    return f"Time {hour}:00 -> {status} (Probability: {prob:.2f})"

# Example Test
print(get_prediction(14, 1, 1)) # 2pm during exam season
print(get_prediction(22, 0, 5)) # 10pm on a weekend, no exams
