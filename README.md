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

### Data Dictionary

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
- วิเคราะห์ผู้ซื้อแยกตามประเทศ

![Correlation Matrix](https://github.com/LN1616/Sales-growth-forecast-ornaments-other-DS514-515/blob/main/image/Ornaments%20-%20Other%20Correlation%20Matrix.png)

ตรวจสอบ ข้อมูลการทำ EDA เพิ่มเติมได้ที่
https://github.com/LN1616/Sales-growth-forecast-ornaments-other-DS512-513.git

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

---

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

### แนวทางการส่งเสริมการขายด้านการตลาด (Marketing Strategy)

1. กลุ่มลูกค้าที่มีมูลค่าในการซื้อสูงที่สุดคือ เพศชายอายุ 25–34 ปีควรใช้ targeted campaigns, personalization และ bundle promotion (+2%)

2. Quantity เป็นปัจจัยที่ส่งผลต่อยอดขายมากที่สุด จึงควรใช้กลยุทธ์ที่มุ่งเพิ่มจำนวนสินค้าที่ลูกค้าซื้อในแต่ละออเดอร์ เช่น Bundle Promotion, Multi-buy Offer และโปรโมชันแบบ “ซื้อ 2 ชิ้นลดเพิ่ม” เพื่อเร่งการเติบโตของยอดขาย (+2%)

3. Targeted campaign for US Market  
ใช้ Free Shipping Threshold สำหรับ US เพราะลูกค้ามักสั่งซื้อหลายชิ้น เพื่อ เพิ่ม basket size ทำ Localized Promotion เช่น “U.S. Limited Edition” (+2%)

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
