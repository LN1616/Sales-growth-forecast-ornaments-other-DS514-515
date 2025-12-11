Sales Growth Forecast for Ornaments & Other Categories in 2026
โครงการคาดการณ์ยอดขายและวิเคราะห์ปัจจัยที่มีผลต่อยอดขายของหมวด Ornaments และ Other เพื่อสนับสนุนการวางแผนเชิงกลยุทธ์ด้วยวิธีการทาง Data Analytics และ Predictive Modeling

1. บทนำและความเป็นมา (Introduction & Background)
ธุรกิจอีคอมเมิร์ซมีความต้องการใช้ข้อมูลเชิงลึกเพื่อเพิ่มประสิทธิภาพการขายและการวางแผนทางการตลาด โดยเฉพาะผลิตภัณฑ์ในหมวด Ornaments และ Other ที่มีความผันผวนของยอดขายสูงตามฤดูกาล พฤติกรรมผู้บริโภค และปัจจัยด้านราคา
แม้จะมียอดขายรวมอยู่ในระดับสูง แต่ความสามารถในการทำกำไรยังไม่เสถียร ทำให้การพัฒนาโมเดลพยากรณ์ยอดขายเป็นสิ่งจำเป็น เพื่อนำไปใช้วางแผนสต๊อก โปรโมชั่น และการตั้งราคาอย่างเหมาะสม

2. วัตถุประสงค์ของโครงการ (Research Objectives)
พัฒนาโมเดลเชิงพยากรณ์เพื่อทำนายยอดขายรายเดือนสำหรับหมวด Ornaments และ Other ในปี 2026
วิเคราะห์ความสัมพันธ์ของตัวแปรทางธุรกิจ (Business Drivers) ที่มีผลต่อยอดขาย เช่น อายุผู้ซื้อ ค่าขนส่ง เรตติ้งสินค้า และจำนวนสินค้า
สร้างกรอบการวิเคราะห์เชิงปริมาณเพื่อสนับสนุนการตัดสินใจของผู้บริหารด้านการตลาดและการจัดการสินค้าคงคลัง
ประเมินประสิทธิภาพของโมเดลภายใต้ตัวชี้วัดทางสถิติ เช่น R², RMSE เพื่อเลือกโมเดลที่เหมาะสมที่สุด

3. คำถามการวิจัยและสมมติฐาน (Research Questions & Hypotheses)
คำถามการวิจัย (Research Questions)
ปัจจัยใดส่งผลต่อยอดขายของหมวด Ornaments และ Other มากที่สุด
พฤติกรรมผู้ซื้อ เช่น อายุ ประเทศ หรือช่วงวัน มีอิทธิพลต่อยอดขายหรือไม่
โมเดล Machine Learning สามารถทำนายยอดขายล่วงหน้าได้แม่นยำเพียงใด
สามารถนำผลลัพธ์ไปใช้วางแผนธุรกิจหรือกำหนดกลยุทธ์ราคาได้หรือไม่

สมมติฐาน (Hypotheses)
H1: Shipping Charges และ Quantity เป็นตัวแปรที่มีความสัมพันธ์เชิงบวกกับ Total Sales
H2: การใช้ Ridge Regression ร่วมกับ Polynomial Features จะเพิ่มความสามารถของโมเดลในการอธิบายความแปรปรวนของข้อมูล


4. ชุดข้อมูลและตัวแปรที่ใช้ (Dataset & Features)
รายละเอียดชุดข้อมูล
จำนวนแถวข้อมูล: 7,394
จำนวนตัวแปรทั้งหมด: 15
แหล่งที่มาของข้อมูล: https://www.kaggle.com/datasets/adarsh0806/influencer-merchandise-sales/data


Data Dictionary 
| Attribute             | คำอธิบาย         | Data Type               | ช่วงค่าที่ถูกต้อง / ตัวอย่าง                |
|-----------------------|----------------------------|-------------------------|----------------------------------------------|
| Order ID              | หมายเลขคำสั่งซื้อ         | Ordinal                 | 189440                                       |
| Order Date            | วันที่มีการสั่งซื้อ        | Interval (Date)         | 20/07/2024                                   |
| Product ID            | หมายเลขสินค้า             | Ordinal                 | BF1543, BF1544, BFXXXX                       |
| Product Category      | หมวดหมู่สินค้า            | Nominal                 | Clothing, Ornaments, Others                  |
| Buyer Gender          | เพศของผู้ซื้อ             | Nominal                 | Male, Female                                 |
| Buyer Age             | อายุของผู้ซื้อ            | Ratio (Continuous)      | [18, 100]                                    |
| Order Location        | สถานที่สั่งซื้อ           | Nominal                 | Las Vegas, Sydney                            |
| International Shipping| จัดส่งระหว่างประเทศหรือไม่| Nominal (Binary)        | Yes / No                                     |
| Sales Price           | ราคาสินค้า                | Ratio (Continuous)      | [0, ∞)                                       |
| Shipping Charges      | ค่าจัดส่งสินค้า            | Ratio (Continuous)      | [0, ∞)                                       |
| Sales per Unit        | ยอดขายต่อหน่วย            | Ratio (Continuous)      | [0, ∞)                                       |
| Quantity              | จำนวนสินค้าที่ซื้อ         | Ratio (Discrete)        | [1, ∞)                                       |
| Total Sales           | ยอดขายรวม                 | Ratio (Continuous)      | [0, ∞)                                       |
| Rating                | คะแนนรีวิว                | Interval (Ordinal/Disc.)| 1–5                                          |
| Review                | ข้อความรีวิวลูกค้า        | Nominal (Text)          | The product was delivered quickly.          |

ตัวแปรเป้าหมาย (Target Variable)
Total Sales
ตัวแปรสำคัญที่ใช้วิเคราะห์
Buyer Age
Buyer Gender
Shipping Charges
Rating
Product Category
Order Date → Month, Quarter
DayOfWeek, IsWeekend
Quantity

5. ระเบียบวิธีวิจัย (Methodology)
5.1 การสำรวจข้อมูลเบื้องต้น (Exploratory Data Analysis: EDA)
วิเคราะห์การกระจายตัวของยอดขาย (Distribution)
วิเคราะห์แนวโน้มตามเวลา (Time Series Trend)
ตรวจสอบความสัมพันธ์ด้วย Correlation Matrix

วิเคราะห์พฤติกรรมผู้ซื้อแยกตาม Category

5.2 Feature Engineering
แยก Order Date เป็น Month และ Quarter
สร้างตัวแปรวันในสัปดาห์ (DayOfWeek) และ IsWeekend
สร้าง Polynomial Features (Degree = 2) เพื่อเพิ่ม nonlinear effect

5.3 Machine Learning Modeling

โมเดลที่ใช้: Ridge Regression พร้อม Polynomial Features
เหตุผลการเลือก:
Ridge ลดปัญหา multicollinearity ที่พบเด่นชัดในข้อมูลเชิงธุรกิจ
Polynomial ช่วยให้โมเดลจับความสัมพันธ์แบบ nonlinear ได้ดีขึ้น

ตัวชี้วัด (Evaluation Metrics)
R² Score: ความสามารถของโมเดลในการอธิบายความแปรปรวน
RMSE: ค่าคลาดเคลื่อนที่วัดความแม่นยำในการทำนาย

6. ผลลัพธ์ของโมเดล (Model Performance)
หมวด Ornaments
Train R²: 0.3759
Test R²: 0.3896
Train RMSE: 56.7367
Test RMSE: 61.2070

หมวด Other
Train R²: 0.5610
Test R²: 0.6046
Train RMSE: 39.6635
Test RMSE: 37.1657

สรุปผลเชิงสถิติ:


7. ข้อค้นพบเชิงลึก (Findings & Insights)
Shipping Charges และ Quantity เป็นตัวแปรที่มีอิทธิพลต่อยอดขายมากที่สุด
Rating ของสินค้าไม่มีนัยสำคัญต่อยอดขาย → อาจสะท้อนว่าผู้ซื้อให้ความสำคัญกับราคาและโปรโมชั่นมากกว่า
กลุ่มอายุผู้ซื้อที่มียอดซื้อมากที่สุดคือช่วง 25–34 ปี
ยอดขายมี seasonal pattern ชัดเจน ส่งผลให้การพยากรณ์บางช่วงมี error สูง
หมวด Ornaments มี volatility สูงกว่า ทำให้ความแม่นยำของโมเดลต่ำกว่าหมวด Other

8. ข้อเสนอเชิงกลยุทธ์ (Strategic Recommendations)
ด้านการตลาด (Marketing Strategy)
ทำ targeted campaign สำหรับกลุ่มอายุ 25–34 ปี
ใช้ bundle และ multi-buy promotion เพื่อผลักดันหมวดที่มียอดขายต่ำ
ด้านบริหารสินค้า (Inventory Strategy)
เพิ่มสต๊อกสินค้ารหัสที่ถูกซื้อซ้ำบ่อย เช่น BF1551, BF1544
ลดสต๊อกสินค้าที่มี demand ต่ำและ volatility สูง
ด้านธุรกิจระหว่างประเทศ (Regional Optimization)
ทำ localized promotion เนื่องจากพฤติกรรมลูกค้าแต่ละประเทศแตกต่างกัน

9. เครื่องมือที่ใช้ (Tech Stack)
Python (pandas, numpy, matplotlib, seaborn)
scikit-learn
Google Colab
GitHub for version control

10. รายชื่อผู้จัดทำ (Contributors)
Pimwipa Leesongsak (68199160287)
Poonyapa Sansupo (68199160283)
