# Sales Growth Forecast for Ornaments & Other Categories in 2026

โครงการคาดการณ์ยอดขายและวิเคราะห์ปัจจัยที่มีผลต่อยอดขายของหมวด Ornaments และ Other เพื่อสนับสนุนการวางแผนเชิงกลยุทธ์ด้วยวิธีการทาง Data Analytics และ Predictive Modeling Using Ridge Regression with Polynimial Features

---

## 1. บทนำและความเป็นมา (Introduction & Background)

ธุรกิจอีคอมเมิร์ซในปัจจุบันมีการแข่งขันสูงและต้องอาศัยข้อมูลเชิงลึกในการตัดสินใจ โดยเฉพาะในหมวดสินค้า Ornaments และ Other ซึ่งมีความผันผวนของยอดขายอันเนื่องมาจากฤดูกาล พฤติกรรมผู้บริโภค และปัจจัยด้านราคา

แม้ยอดขายรวมจะอยู่ในระดับที่ดี แต่ยังพบความไม่เสถียรของกำไร ทำให้จำเป็นต้องพัฒนาโมเดลพยากรณ์ยอดขาย เพื่อใช้วางแผนด้านสต๊อก โปรโมชั่น และกลยุทธ์ราคาในปี 2026

โดยมีเป้าหมายหลักคือ เพิ่มยอดขาย 5% จากยอดขายรวมปี 2025 ของหมวด Ornaments และ Other ภายใน Q4/2026 โดยอาศัยข้อมูลปี 2025 ในการปรับกลยุทธ์ target และ marketing campaign

---

## 2. วัตถุประสงค์ของโครงการ (Research Objectives)

- พัฒนาโมเดลพยากรณ์ยอดขายรายเดือนสำหรับปี 2026 ในหมวด Ornaments และ Other  
- วิเคราะห์ตัวแปรทางธุรกิจที่ส่งผลต่อยอดขาย เช่น อายุผู้ซื้อ ค่าจัดส่ง จำนวนสินค้า และเรตติ้ง  
- สร้างกรอบการวิเคราะห์เพื่อสนับสนุนการตัดสินใจด้านการตลาดและการบริหารสินค้าคงคลัง  
- ประเมินประสิทธิภาพของโมเดลด้วยตัวชี้วัด R² และ RMSE เพื่อหาโมเดลที่เหมาะสมที่สุด  

---

## 3. คำถามการวิจัยและสมมติฐาน (Research Questions & Hypotheses)

### Research Questions

1. ปัจจัยใดมีผลต่อยอดขายของหมวด Ornaments และ Other มากที่สุด  
2. พฤติกรรมของผู้ซื้อ เช่น อายุ เพศ ประเทศ หรือช่วงวัน มีอิทธิพลต่อยอดขายหรือไม่  
3. โมเดล Machine Learning สามารถทำนายยอดขายได้แม่นยำเพียงใด  
4. สามารถนำผลลัพธ์ไปใช้กำหนดกลยุทธ์ด้านราคา การตลาด และการบริหารสต๊อกได้หรือไม่  

### Hypotheses

- H1: Shipping Charges และ Quantity มีความสัมพันธ์เชิงบวกกับ Total Sales  
- H2: การใช้ Ridge Regression ร่วมกับ Polynomial Features จะช่วยเพิ่มความสามารถของโมเดลในการอธิบายความแปรปรวนของข้อมูล  

---

## 4. ชุดข้อมูลและตัวแปรที่ใช้ (Dataset & Features)

- จำนวนแถวข้อมูล: 7,394  
- จำนวนตัวแปรทั้งหมด: 15  
- แหล่งที่มาของข้อมูล: https://www.kaggle.com/datasets/adarsh0806/influencer-merchandise-sales/data  

#### Data Dictionary

| Attribute | คำอธิบาย | Data Type | ช่วงค่าที่ถูกต้อง / ตัวอย่าง |
|---|---|---|---|
| Order ID | หมายเลขคำสั่งซื้อ | Ordinal | 189440 |
| Order Date | วันที่มีการสั่งซื้อ | Interval (Date) | 20/07/2024 |
| Product ID | หมายเลขสินค้า | Ordinal | BF1543, BF1544, BFXXXX |
| Product Category | หมวดหมู่สินค้า | Nominal | Clothing, Ornaments, Others |
| Buyer Gender | เพศของผู้ซื้อ | Nominal | Male, Female |
| Buyer Age | อายุผู้ซื้อ | Ratio (Continuous) | 18–100 |
| Order Location | สถานที่สั่งซื้อ | Nominal | Las Vegas, Sydney |
| International Shipping | จัดส่งระหว่างประเทศหรือไม่ | Nominal (Binary) | Yes / No |
| Sales Price | ราคาสินค้า | Ratio (Continuous) | 0–∞ |
| Shipping Charges | ค่าจัดส่งสินค้า | Ratio (Continuous) | 0–∞ |
| Sales per Unit | ยอดขายต่อหน่วย | Ratio (Continuous) | 0–∞ |
| Quantity | จำนวนสินค้าที่ซื้อ | Ratio (Discrete) | 1–∞ |
| Total Sales | ยอดขายรวม | Ratio (Continuous) | 0–∞ |
| Rating | คะแนนรีวิว | Interval / Ordinal | 1–5 |
| Review | รีวิวลูกค้า | Nominal (Text) | “The product was delivered quickly.” |


#### ตัวแปรเป้าหมาย (Target Variable)

- Total Sales

```python
y = df["Total Sales"]
```


#### ตัวแปรสำคัญที่ใช้วิเคราะห์ (Key Features)
- Buyer Age  
- Buyer Gender  
- Shipping Charges  
- Rating  
- Product Category  
- Order Date → Month, Quarter  
- DayOfWeek, IsWeekend  
- Quantity  


```python
feature_cols = [
    "Buyer Age",
    "Buyer Gender",
    "Shipping Charges",
    "Rating",
    "Product Category",
    "OrderMonth",
    "Quarter",
    "DayOfWeek",
    "IsWeekend",
    "Quantity"
]
X = df[feature_cols]
```

## 5. ระเบียบวิธีวิจัย (Methodology)

### 5.1 Data Cleaning and Preparation
- ตรวจสอบว่าคอลัมน์ที่จำเป็นสำหรับการสร้างโมเดลมีอยู่ครบ
- แปลงค่าของตัวแปรเชิงตัวเลขทั้งหมดให้อยู่ในรูป numeric
- ลบแถวข้อมูลที่มีค่า NaN หลังจากการแปลงข้อมูล
- ตรวจสอบว่าข้อมูลไม่ว่างเปล่าหลังการทำความสะอาด
- กำหนดตัวแปรอิสระ (X) และตัวแปรเป้าหมาย (y)
- แบ่ง Train/Test Split

#### ตรวจสอบว่าคอลัมน์ที่จำเป็นมีครบหรือไม่
```python
    missing = [c for c in feature_cols + ["Total Sales"] if c not in df_cat.columns]
    if missing:
        raise ValueError(f"Missing columns for modelling: {missing}")
```
#### แปลงข้อมูลฟีเจอร์และ target ให้เป็นตัวเลข
```python
    for col in feature_cols + ["Total Sales"]:
        df_cat[col] = pd.to_numeric(df_cat[col], errors="coerce")
```
#### ลบแถวที่มีค่า NaN ในฟีเจอร์หรือ target
```python
    df_cat = df_cat.dropna(subset=feature_cols + ["Total Sales"])
    if df_cat.empty:
        raise ValueError(f"All rows became NaN after numeric conversion for {category_name}")
```
#### กำหนดตัวแปรอิสระ (X) และตัวแปรเป้าหมาย (y)
```python
    X = df_cat[feature_cols]
    y = df_cat["Total Sales"]
```
#### แบ่ง Train/Test Split
```python
 X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)
```

### 5.2 Feature Engineering

- Age Grouping
- แปลง Order Date เป็นตัวแปร Month และ Quarter  
- สร้าง DayOfWeek และ IsWeekend เพื่อแทนพฤติกรรมการซื้อในวันธรรมดาและวันหยุด  #

#### Age Grouping
```python
bins = [18, 25, 35, 45,200]
labels = ["18-24", "25-34", "35-44","45+"]

df["AgeGroup"] = pd.cut(
    df["Buyer Age"],
    bins=bins,
    labels=labels,
    right=False,        # เปลี่ยนเป็น [18,25), [25,35), [35,45)
    include_lowest=True # ให้ 18 ตกอยู่ในกลุ่มแรกด้วย
)
```

#### สร้าง feature จาก Order Date สำหรับใช้ใน EDA
```python
df["OrderMonth"] = df["Order Date"].dt.month
df["DayOfWeek"]  = df["Order Date"].dt.dayofweek
df["day_of_month"] = df["Order Date"].dt.day
df["Quarter"] = df["Order Date"].dt.quarter
df["week"] = df["Order Date"].dt.isocalendar().week.astype(int)
df["week_start"] = df["Order Date"] - pd.to_timedelta(df["Order Date"].dt.weekday, unit="D")
df["IsWeekend"]  = df["DayOfWeek"].isin([5, 6]).astype(int)
```

#### สร้าง Polynomial Features (degree = 2) และรวมกับ StandardScaler + Ridge Regression ใน Pipeline
```python
  pipe = Pipeline([
        ("poly",   PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("model",  Ridge())
    ])
```

#### GridSearch หา alpha ที่ดีที่สุด
```python
    param_grid = {
        "model__alpha": [0.1, 1, 5, 10, 20, 50]
    }
    gs = GridSearchCV(
        pipe,
        param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )
```
    
### 5.3 Exploratory Data Analysis (EDA)

- วิเคราะห์การกระจายตัวของยอดขาย (Distribution of Total Sales)  
- วิเคราะห์แนวโน้มตามเวลา (Time Series Trend)  
- ตรวจสอบความสัมพันธ์ด้วย Correlation Matrix  
- วิเคราะห์ผู้ซื้อแยกตามประเทศ

![Correlation Matrix](https://github.com/LN1616/Sales-growth-forecast-ornaments-other-DS514-515/blob/main/image/Ornaments%20-%20Other%20Correlation%20Matrix.png)

ตรวจสอบข้อมูลการทำ EDA เพิ่มเติมได้ที่
https://github.com/LN1616/Sales-growth-forecast-ornaments-other-DS512-513.git


### 5.4 Machine Learning Modeling
- โมเดลที่ใช้: Ridge Regression ร่วมกับ Polynomial Features  
- เหตุผลการเลือก:
  - Ridge Regression ช่วยลดปัญหา Multicollinearity ในตัวแปรอิสระ  
  - Polynomial Features ช่วยให้โมเดลสามารถจับความสัมพันธ์ที่ไม่เป็นเส้นตรงระหว่างตัวแปรกับยอดขายได้ดีขึ้น  
- ตัวชี้วัดที่ใช้ประเมิน (Evaluation Metrics):
  - R² (Coefficient of Determination)  
  - RMSE (Root Mean Squared Error)  

#### Train Model
```python
    gs.fit(X_train, y_train)
```

#### ดึง best model
```python
    best_model = gs.best_estimator_
```

#### Train metrics
```python
    y_train_pred = best_model.predict(X_train)
    train_r2   = r2_score(y_train, y_train_pred)
    train_rmse = mean_squared_error(y_train, y_train_pred) ** 0.5
```
#### Test metrics
```python
    y_test_pred = best_model.predict(X_test)
    test_r2   = r2_score(y_test, y_test_pred)
    test_rmse = mean_squared_error(y_test, y_test_pred) ** 0.5
```

```python
 print(f"===== Category (Poly degree={degree}): {category_name} =====")
    print("Best parameters:", gs.best_params_)
    print(f"Train R^2  : {train_r2:.4f}")
    print(f"Train RMSE : {train_rmse:.4f}")
    print(f"Test R^2   : {test_r2:.4f}")
    print(f"Test RMSE  : {test_rmse:.4f}")

    return gs, (X_train, X_test, y_train, y_test)
```


## 6. ผลลัพธ์ของโมเดล (Model Performance)
#### Model Evaluation
```python
# Ornaments (Polynomial)
model_orn_poly, data_orn_poly = train_ridge_poly_for_category(df, "Ornaments", degree=2)

# Other (Polynomial)
model_oth_poly, data_oth_poly = train_ridge_poly_for_category(df, "Other", degree=2)
```

#### หมวด Ornaments

| Metric | Score |
|---|---:|
| Train R² | 0.3759 |
| Test R² | 0.3896 |
| Train RMSE | 56.7367 |
| Test RMSE | 61.2070 |

### หมวด Other

| Metric | Score |
|---|---:|
| Train R² | 0.5610 |
| Test R² | 0.6046 |
| Train RMSE | 39.6635 |
| Test RMSE | 37.1657 |



## 7. ข้อค้นพบเชิงลึก (Findings & Insights)

โมเดลแสดงให้เห็นถึง generalization ที่ดี เนื่องจากค่า Train และ Test performance มีความใกล้เคียงกัน โดยค่า R² ของ Ornaments อยู่ที่ประมาณ 0.39 และ Other ประมาณ 0.60 แม้ค่า R² จะอยู่ในระดับปานกลาง แต่โมเดลสามารถอธิบาย ทิศทางและแนวโน้มเชิงธุรกิจ (forecast direction/trend) ได้อย่างเหมาะสม ทั้งนี้ ข้อมูลมีลักษณะ right-skew และมี outliers สูง ซึ่งจำกัดเพดานของค่า R² ส่งผลให้ผลการพยากรณ์ชี้ว่ายอดขายปี 2026 มีแนวโน้มทรงตัวถึงลดลงเล็กน้อย จึงจำเป็นต้องอาศัย กลยุทธ์ทางธุรกิจเพิ่มเติม เพื่อให้บรรลุเป้าหมายการเติบโต +5%

---

### Ornaments

- Train R² = 0.3759  
- Test R² = 0.3896  
ยอดขายมีความผันผวนสูง ทำให้ความแม่นยำในการพยากรณ์ต่ำกว่า  
- Train RMSE = 56.74  
- Test RMSE = 61.21  
สะท้อนว่าโมเดลมีความคลาดเคลื่อนในการทำนายค่อนข้างสูง สอดคล้องกับลักษณะยอดขายที่มีความผันผวน (high volatility)

### Other

- Train R² = 0.5610  
- Test R² = 0.6046  
มีค่า R² สูงกว่า Ornaments แสดงถึงความสามารถในการ generalize ที่ดีกว่า  
- Train RMSE = 39.66  
- Test RMSE = 37.17  
ค่า RMSE ต่ำกว่า Ornaments อย่างชัดเจน แสดงว่าโมเดลสามารถทำนายยอดขายของหมวด Other ได้แม่นยำกว่า

---

### ผลของการเพิ่ม Polynomial Features

การเพิ่ม Polynomial Degree = 2 ทำให้โมเดลสามารถจับ **ความสัมพันธ์แบบไม่เชิงเส้น (non-linear patterns)** ได้ดีขึ้น เช่น  
- ค่าจัดส่งสูงเกิน threshold ทำให้ยอดขายลดลงทันที  
- จำนวนสินค้าที่เพิ่มขึ้นส่งผลต่อยอดขายแบบโค้ง (diminishing returns)

การเพิ่ม Polynomial Features ช่วยเพิ่มค่า R² ในทั้งสองหมวดสินค้า อย่างไรก็ตาม ความแม่นยำในการพยากรณ์เชิงตัวเลขยังอยู่ในระดับไม่สูงพอสำหรับการคาดการณ์แบบละเอียด จึงเหมาะกับการใช้เพื่อทำความเข้าใจโครงสร้างความสัมพันธ์ของข้อมูลมากกว่า

---

### บทบาทของ Ridge Regression ในคุณภาพของโมเดล

ข้อมูลธุรกิจมักมีลักษณะ multicollinearity ระหว่างตัวแปร เช่น:
- Sales Price, Quantity และ Shipping Charges  
- Price ↔ Rating  
- Quantity ↔ Total Sales (อาจเกิด target leakage หากไม่จัดการให้ดี)

Ridge Regression ช่วยให้โมเดล:
- มีความเสถียร (coefficient stability)  
- ลดความแกว่งของค่าสัมประสิทธิ์  
- ลดความเสี่ยง overfitting  
- เพิ่มความสามารถในการ generalize เมื่อพบข้อมูลใหม่  

---

### สรุปภาพรวม

โมเดล Ridge Regression with Polynomial Features ช่วยให้เข้าใจปัจจัยที่มีผลต่อยอดขายได้ดีในระดับหนึ่ง โดยเฉพาะในหมวด Other ที่มีรูปแบบข้อมูลเสถียรและสอดคล้องกับโมเดลมากกว่า แม้ผลการพยากรณ์เชิงปริมาณจะยังไม่สูง แต่โมเดลนี้มีประโยชน์ต่อการวิเคราะห์เชิงกลยุทธ์ และสามารถต่อยอดไปยังโมเดลที่ซับซ้อนขึ้นเพื่อเพิ่มความแม่นยำได้ในอนาคต

![Scatter Plot](https://github.com/LN1616/Sales-growth-forecast-ornaments-other-DS514-515/blob/main/image/Ornaments%20-%20Other%20Scatter%20plot%20-%20Actual%20vs%20Predicted.png)
---

## 8. ข้อเสนอเชิงกลยุทธ์ (Strategic Recommendations)
![Forecast Ornaments](https://github.com/LN1616/Sales-growth-forecast-ornaments-other-DS514-515/blob/main/image/Ornament-Total%20Sales%20-%20Actual%2C%20Forcast-%20Monthly%20Gap%20goal.png)
![Forecast Other](https://github.com/LN1616/Sales-growth-forecast-ornaments-other-DS514-515/blob/main/image/Other-Total%20Sales%20-%20Actual%2C%20Forcast-%20Monthly%20Gap%20goal.png)
1. Forecast Indicates Natural Sales Decline in 2026  
โมเดลคาดการณ์ว่ายอดขายหมวด Ornaments (-1.24%) และ Other (-0.30%) มีแนวโน้มทรงตัวถึงลดลง หากไม่มีกลยุทธ์เชิงรุกเพิ่มเติม

2. +5% Growth Target Requires Proactive Monthly Uplift  
การบรรลุเป้าหมาย +5% YoY จำเป็นต้องปิดช่องว่างรายเดือน  
โดยเฉพาะหมวด Ornaments (+$810 ต่อเดือน) ที่ต้องเพิ่มยอดขายเฉลี่ยสูงกว่า Other (+$280 ต่อเดือน)

3. Data-Driven Actions Needed to Close the Growth Gap  
ผลการวิเคราะห์ชี้ว่า การเพิ่ม Quantity ผ่าน targeted campaigns และ market-specific strategies เป็นกลไกหลักในการขับเคลื่อนการเติบโต

เนื่องจากผลการพยากรณ์จากโมเดลชี้ให้เห็นว่า ยอดขายหมวด Ornaments มีแนวโน้มลดลงประมาณ −1.24% และ หมวด Other ลดลงประมาณ −0.3% เมื่อเทียบกับปี 2025 เพื่อให้บรรลุ SMART Objective ที่กำหนดไว้ จึงจำเป็นต้องผลักดันการเติบโตเพิ่มเติม โดยคาดว่าต้องสร้าง uplift เพิ่มประมาณ +5.9% สำหรับหมวด Ornaments และ ประมาณ +5% สำหรับหมวด Other จากระดับที่โมเดลคาดการณ์ไว้ จาก EDA Insights สามารถกำหนดแนวทางเชิงกลยุทธ์เพื่อปิดช่องว่างจากเป้าหมายการเติบโตได้ดังนี้

---

### 8.1 Customer Strategy: Target High-Value Segment (Male, Age 25–34)

ผลการวิเคราะห์ระบุว่าผู้ชายอายุ 25–34 ปีเป็นกลุ่มที่สร้างยอดซื้อสูงที่สุดในทั้งสองหมวดสินค้า  
จึงควรดำเนินกลยุทธ์ดังนี้:

- ใช้ **Targeted Campaigns** เพื่อสื่อสารกับกลุ่มลูกค้าเป้าหมายอย่างเฉพาะเจาะจง  
- นำ **Personalization Strategy** มาใช้ เช่น แนะนำสินค้าเฉพาะบุคคล (Personalized Product Recommendations)  
- พัฒนา **Bundle Promotion** ที่ตอบโจทย์ความต้องการของกลุ่มนี้ เช่นเซ็ตสินค้า หรือสินค้าคู่กัน  

**Expected Impact**  
การเจาะกลุ่มลูกค้าหลักโดยตรงมีแนวโน้มเพิ่ม Conversion Rate และ Customer Lifetime Value (CLV) ได้อย่างมีนัยสำคัญ

---

### 8.2 Sales Strategy: Leverage Quantity as the Key Driver of Total Sales

จาก Correlation Analysis พบว่า **Quantity เป็นตัวแปรที่ส่งผลต่อยอดขายมากที่สุดในหมวด Ornaments (Correlation = 0.62)**  
จึงควรใช้กลยุทธ์ที่มุ่งเพิ่มจำนวนสินค้าในแต่ละคำสั่งซื้อ ดังนี้:

- **Bundle Promotion** เช่น ซื้อคู่ราคาพิเศษ  
- **Multi-buy Offer** เช่น ซื้อ 2 ชิ้นลด 15%, ซื้อ 3 ชิ้นลด 25%  
- โปรโมชั่น “**Buy More, Save More**” เพื่อกระตุ้นให้ผู้ซื้อเพิ่มจำนวนสินค้าในตะกร้า  

**Expected Impact**  
เพิ่มยอดขายรวม (Total Sales) โดยตรง เนื่องจากโมเดลชี้ให้เห็นว่าโครงสร้างรายได้ของ Ornaments และ Other เป็นแบบ **quantity-driven** มากกว่า price-driven

---

### 8.3 Market Strategy: Focus on the United States (US Market Optimization)

ข้อมูลยอดขายชี้ว่า **ตลาดสหรัฐอเมริกาเป็นตลาดที่สร้างรายได้หลัก** ของสินค้าในทั้งสองหมวด  
กลยุทธ์ที่ควรดำเนินการ ได้แก่:

- กำหนด **Free Shipping Threshold** สำหรับตลาดสหรัฐ เนื่องจากลูกค้ามีแนวโน้มซื้อหลายชิ้นต่อคำสั่งซื้อ → ช่วยเพิ่ม basket size  
- จัดทำ **Localized Promotion** เช่น  
  - แคมเปญ “U.S. Limited Edition”  
  - โปรโมชั่นตามเทศกาลสำคัญของสหรัฐ เช่น Black Friday, Independence Day  
- ปรับปรุงข้อมูลสินค้าและกลยุทธ์สื่อสารให้สอดคล้องกับพฤติกรรมผู้บริโภคในสหรัฐ

**Expected Impact**  
เพิ่มความสามารถในการแข่งขันในตลาดหลัก เพิ่ม Repeat Purchase Rate และลดต้นทุนด้านโลจิสติกส์ผ่านการบริหารสต๊อกที่เหมาะสม

---

### 8.4 Summary of Business Impact

- การเจาะกลุ่มลูกค้าที่มีมูลค่าสูงช่วยเพิ่มยอดขายแบบมีประสิทธิภาพ  
- กลยุทธ์ที่เพิ่มจำนวนสินค้าในตะกร้ามีผลโดยตรงต่อยอดขายรวม  
- ตลาดสหรัฐมีศักยภาพสูงสุดและควรได้รับความสำคัญในเชิงกลยุทธ์  
- การบูรณาการกลยุทธ์ด้านราคา โปรโมชั่น และการปรับแต่งตามพื้นที่ จะช่วยเพิ่มยอดขายอย่างยั่งยืนในระยะยาว


## 9. เครื่องมือที่ใช้ (Tech Stack)

- Python: pandas, numpy, matplotlib, seaborn  
- scikit-learn  
- Google Colab  
- GitHub สำหรับ Version Control  

---

## 10. รายชื่อผู้จัดทำ (Contributors)

- Pimwipa Leesongsak (68199160287)  
- Poonyapa Sansupo (68199160283)


ตรวจสอบข้อมูลการทำ EDA เพิ่มเติมได้ที่
https://github.com/LN1616/Sales-growth-forecast-ornaments-other-DS512-513.git

