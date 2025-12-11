# Sales-growth-forecast-ornaments-other-DS514-515
# Sales Growth Forecast for Ornaments & Other Categories in 2026
โครงการคาดการณ์ยอดขายและวิเคราะห์ปัจจัยที่มีผลต่อยอดขายของหมวด Ornaments และ Other เพื่อสนับสนุนการวางแผนกลยุทธ์ทางธุรกิจสำหรับปี 2026 โดยใช้เครื่องมือด้าน Data Analytics และ Predictive Modeling

---

## 1. วัตถุประสงค์ของโครงการ  
1. คาดการณ์ยอดขายรายเดือนของหมวด Ornaments และ Other ในปี 2026  
2. วิเคราะห์ปัจจัยที่มีผลต่อยอดขาย เช่น อายุผู้ซื้อ ค่าขนส่ง เรตติ้ง จำนวนสินค้า  
3. นำผลการวิเคราะห์ไปสนับสนุนการตัดสินใจด้านการตลาด การตั้งราคา และการจัดการสต๊อก  
4. พัฒนาโมเดลพยากรณ์ยอดขายโดยใช้ Ridge Regression ร่วมกับ Polynomial Features  

---

## 2. ข้อมูลที่ใช้ในการวิเคราะห์  
- จำนวนข้อมูล: 7,394 แถว  
- จำนวนตัวแปร: 15 ตัว  
- ตัวแปรเป้าหมาย: Total Sales  

ตัวแปรสำคัญที่ใช้ในโมเดล ได้แก่  
- Buyer Age  
- Shipping Charges  
- Rating  
- Order Month, Quarter  
- DayOfWeek, IsWeekend  
- Quantity  

---

## 3. วิธีการดำเนินงาน (Methodology)

### 3.1 การวิเคราะห์ข้อมูลเบื้องต้น (EDA)
- วิเคราะห์แนวโน้มยอดขาย (Time Series Analysis)  
- วิเคราะห์ความสัมพันธ์ระหว่างตัวแปร (Correlation Analysis)  
- วิเคราะห์พฤติกรรมการซื้อของลูกค้าแยกตาม Category  

### 3.2 การทำ Feature Engineering  
- สร้างตัวแปร OrderMonth, Quarter  
- สร้างตัวแปร DayOfWeek, IsWeekend  
- ปรับโครงสร้างตัวแปรสำหรับนำเข้าโมเดล  

### 3.3 การสร้างโมเดล Machine Learning  
- Ridge Regression (Polynomial Degree = 2)  
- ตัวชี้วัดผลการพยากรณ์:  
  - R² (ความสามารถในการอธิบายความแปรปรวน)  
  - RMSE (ค่าคาดเคลื่อนของผลการทำนาย)  

---

## 4. ผลลัพธ์ของโมเดล (Model Performance)

### หมวด Ornaments  
- Best Model: Ridge + Polynomial (degree=2)  
- Train R²: 0.3759  
- Test R²: 0.3896  
- Test RMSE: 61.2070  

### หมวด Other  
- Best Model: Ridge + Polynomial (degree=2)  
- Train R²: 0.5610  
- Test R²: 0.6046  
- Test RMSE: 37.1657  


