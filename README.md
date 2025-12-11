# Sales Growth Forecast for Ornaments & Other Categories in 2026
โครงการคาดการณ์ยอดขายและวิเคราะห์ปัจจัยที่มีผลต่อยอดขายของหมวด Ornaments และ Other เพื่อสนับสนุนการวางแผนเชิงกลยุทธ์ด้วยวิธีการทาง Data Analytics และ Predictive Modeling

---

## 1. บทนำและความเป็นมา (Introduction & Background)

ธุรกิจอีคอมเมิร์ซในปัจจุบันมีการแข่งขันสูงและต้องอาศัยข้อมูลเชิงลึกในการตัดสินใจ โดยเฉพาะในหมวดสินค้า Ornaments และ Other
ซึ่งมีความผันผวนของยอดขายอันเนื่องมาจากฤดูกาล พฤติกรรมผู้บริโภค และปัจจัยด้านราคา

แม้ยอดขายรวมจะอยู่ในระดับที่ดี แต่ยังพบความไม่เสถียรของกำไร ทำให้จำเป็นต้องพัฒนาโมเดลพยากรณ์ยอดขาย เพื่อใช้วางแผนด้านสต๊อก โปรโมชั่น และกลยุทธ์ราคาในปี 2026

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

### Data Dictionary

| Attribute             | คำอธิบาย                        | Data Type               | ช่วงค่าที่ถูกต้อง / ตัวอย่าง                |
|-----------------------|----------------------------------|-------------------------|----------------------------------------------|
| Order ID              | หมายเลขคำสั่งซื้อ               | Ordinal                 | 189440                                       |
| Order Date            | วันที่มีการสั่งซื้อ              | Interval (Date)         | 20/07/2024                                   |
| Product ID            | หมายเลขสินค้า                   | Ordinal                 | BF1543, BF1544, BFXXXX                       |
| Product Category      | หมวดหมู่สินค้า                  | Nominal                 | Clothing, Ornaments, Others                  |
| Buyer Gender          | เพศของผู้ซื้อ                   | Nominal                 | Male, Female                                 |
| Buyer Age             | อายุผู้ซื้อ                     | Ratio (Continuous)      | 18–100                                       |
| Order Location        | สถานที่สั่งซื้อ                 | Nominal                 | Las Vegas, Sydney                            |
| International Shipping| จัดส่งระหว่างประเทศหรือไม่     | Nominal (Binary)        | Yes / No                                     |
| Sales Price           | ราคาสินค้า                      | Ratio (Continuous)      | 0–∞                                          |
| Shipping Charges      | ค่าจัดส่งสินค้า                 | Ratio (Continuous)      | 0–∞                                          |
| Sales per Unit        | ยอดขายต่อหน่วย                  | Ratio (Continuous)      | 0–∞                                          |
| Quantity              | จำนวนสินค้าที่ซื้อ               | Ratio (Discrete)        | 1–∞                                          |
| Total Sales           | ยอดขายรวม                       | Ratio (Continuous)      | 0–∞                                          |
| Rating                | คะแนนรีวิว                      | Interval / Ordinal      | 1–5                                          |
| Review                | รีวิวลูกค้า                     | Nominal (Text)          | “The product was delivered quickly.”         |

### ตัวแปรเป้าหมาย (Target Variable)

- Total Sales

### ตัวแปรสำคัญที่ใช้วิเคราะห์ (Key Features)

- Buyer Age  
- Buyer Gender  
- Shipping Charges  
- Rating  
- Product Category  
- Order Date → Month, Quarter  
- DayOfWeek, IsWeekend  
- Quantity  

---

## 5. ระเบียบวิธีวิจัย (Methodology)

### 5.1 Data Cleaning

- ไม่มี Missing Values  
- ไม่มี Duplicate Records  
- ตรวจสอบและปรับรูปแบบวันที่ (Date Format) ให้เป็นชนิด datetime  

### 5.2 Exploratory Data Analysis (EDA)

- วิเคราะห์การกระจายตัวของยอดขาย (Distribution of Total Sales)  
- วิเคราะห์แนวโน้มตามเวลา (Time Series Trend)  
- ตรวจสอบความสัมพันธ์ด้วย Correlation Matrix  
- วิเคราะห์พฤติกรรมผู้ซื้อแยกตาม Product Category  

### 5.3 Feature Engineering

- แปลง Order Date เป็นตัวแปร Month และ Quarter  
- สร้าง DayOfWeek และ IsWeekend เพื่อแทนพฤติกรรมการซื้อในวันธรรมดาและวันหยุด  
- สร้าง Polynomial Features (Degree = 2) เพื่อจับความสัมพันธ์แบบไม่เชิงเส้น (Nonlinear Relationship)  

### 5.4 Machine Learning Modeling

- โมเดลที่ใช้: Ridge Regression ร่วมกับ Polynomial Features  
- เหตุผลการเลือก:
  - Ridge Regression ช่วยลดปัญหา Multicollinearity ในตัวแปรอิสระ  
  - Polynomial Features ช่วยให้โมเดลสามารถจับความสัมพันธ์ที่ไม่เป็นเส้นตรงระหว่างตัวแปรกับยอดขายได้ดีขึ้น  

- ตัวชี้วัดที่ใช้ประเมิน (Evaluation Metrics):
  - R² (Coefficient of Determination)  
  - RMSE (Root Mean Squared Error)  

---

## 6. ผลลัพธ์ของโมเดล (Model Performance)

### หมวด Ornaments

| Metric    | Score   |
|-----------|---------|
| Train R²  | 0.3759  |
| Test R²   | 0.3896  |
| Train RMSE| 56.7367 |
| Test RMSE | 61.2070 |

### หมวด Other

| Metric    | Score   |
|-----------|---------|
| Train R²  | 0.5610  |
| Test R²   | 0.6046  |
| Train RMSE| 39.6635 |
| Test RMSE | 37.1657 |

---

## 7. ข้อค้นพบเชิงลึก (Findings & Insights)

จากการทดลองพัฒนาโมเดลหลายรูปแบบ ได้แก่ Linear Regression, Polynomial Regression, และ Regularized Models เช่น Ridge Regression พบว่าโมเดล Ridge Regression ร่วมกับ Polynomial Features ให้ผลลัพธ์ที่มีความสมดุลที่สุดระหว่าง
ความซับซ้อนของโมเดล (Model Complexity) และความสามารถในการอธิบายข้อมูล (Explanatory Power) พร้อมทั้งคงไว้ซึ่ง generalization ที่เหมาะสม เมื่อทดสอบกับชุดข้อมูลที่ไม่เคยเห็นมาก่อน (Test Set)
แม้โมเดลจะยังไม่สามารถทำนายยอดขายได้อย่างแม่นยำในระดับสูง แต่ก็สามารถสะท้อนรูปแบบความสัมพันธ์เชิงธุรกิจได้ดี โดยเฉพาะในหมวด Other ซึ่งมีข้อมูลที่เสถียรกว่าและมีสัญญาณ (signal) ชัดเจนกว่าหมวด Ornaments
1) โมเดลในหมวด “Other” มี generalization ดีกว่า Ornaments

ค่า Test R² = 0.6046 แสดงว่าโมเดลสามารถอธิบายความแปรปรวนของยอดขายได้ประมาณ 60%
ถือว่าอยู่ในระดับ “ปานกลางค่อนข้างดี” สำหรับข้อมูลธุรกิจที่มี noise สูง

ในขณะเดียวกันค่า RMSE = 37.17 บ่งบอกว่าความคลาดเคลื่อนเฉลี่ยระหว่างค่าจริงและค่าทำนายอยู่ที่ประมาณ 37 หน่วย ซึ่งถือว่า “ยอมรับได้” ในบริบทข้อมูลนี้

ในทางกลับกัน หมวด Ornaments

มี R² เพียง 0.39 → โมเดลอธิบายความแปรปรวนได้เพียงบางส่วน

RMSE สูงถึง 61 → ความผันผวนของยอดขายสูง ทำให้โมเดลคาดการณ์ได้ยาก

สิ่งนี้สะท้อนว่า โพรงพฤติกรรมยอดขายของ Ornaments มีความไม่เสถียรสูงกว่า จึงเหมาะกับโมเดลที่ซับซ้อนขึ้น เช่น Tree-Based Models (RF, XGBoost) หรือ Time Series Models

2) การเพิ่ม Polynomial Features ช่วยให้โมเดลจับ non-linear patterns ได้ดีขึ้น

ผลลัพธ์ของ R² ทั้งสองหมวดดีขึ้นหลังเพิ่ม Polynomial Degree = 2
เนื่องจากความสัมพันธ์ของยอดขายมักเป็นแบบไม่เชิงเส้น เช่น

ค่าจัดส่งสูงเกิน threshold → คนไม่ซื้อ

ปริมาณมากขึ้น → ยอดขายเพิ่มแบบโค้ง (diminishing returns)

อย่างไรก็ตาม แม้ R² จะดีขึ้น แต่ระดับการพยากรณ์ยังไม่เพียงพอสำหรับการนำไปใช้ทำนายยอดขายแบบละเอียดเชิงตัวเลข

3) Ridge Regression ให้ความสมดุลของ bias–variance และช่วยลด multicollinearity

ข้อมูลธุรกิจมักมีตัวแปรที่สัมพันธ์กัน เช่น

Sales Price, Quantity, Shipping Charges

Price ↔ Rating

Quantity ↔ Total Sales (target leakage ถ้าไม่จัดการให้ดี)

Ridge Regression ทำให้โมเดล

คงความเสถียร (stability)

ลดการแกว่งของค่าสัมประสิทธิ์

เพิ่มความสามารถในการ generalize เมื่อเจอข้อมูลใหม่

ดังนั้นโมเดลนี้จึงเหมาะสำหรับ “การอธิบายปัจจัยที่มีผลต่อยอดขาย” แม้จะไม่ใช่โมเดลที่แม่นยำที่สุดสำหรับการคาดการณ์ตัวเลขจริง

---

## 8. ข้อเสนอเชิงกลยุทธ์ (Strategic Recommendations)

### ด้านการตลาด (Marketing Strategy)

- ทำ Targeted Campaign สำหรับลูกค้ากลุ่มอายุ 25–34 ปีที่เป็นกลุ่มกำลังซื้อหลัก  
- ใช้โปรโมชั่นแบบ Bundle หรือ Multi-Buy เพื่อผลักดันหมวดสินค้าที่มียอดขายต่ำแต่มีศักยภาพเติบโต  

### ด้านบริหารสินค้า (Inventory Strategy)

- เพิ่มสต๊อกสินค้าในรหัสที่มียอดซื้อซ้ำสูง เช่น BF1551, BF1544 (ตัวอย่าง)  
- ลดสต๊อกสินค้าที่มี Demand ต่ำและความผันผวนสูง  

### ด้านธุรกิจระหว่างประเทศ (Regional Optimization)

- ทำ Localized Promotion ตามลักษณะพฤติกรรมลูกค้าแต่ละประเทศ  
- ใช้ข้อมูลยอดขายต่อภูมิภาคประกอบการวางแผนสต๊อกและการจัดส่ง  

---

## 9. เครื่องมือที่ใช้ (Tech Stack)

- Python: pandas, numpy, matplotlib, seaborn  
- scikit-learn  
- Google Colab  
- GitHub สำหรับ Version Control  

---

## 10. รายชื่อผู้จัดทำ (Contributors)

- Pimwipa Leesongsak (68199160287)  
- Poonyapa Sansupo (68199160283)

---
