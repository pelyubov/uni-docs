---
share: "true"
---

# Abstract
* Bài đánh giá:
	* ==Tự tạo dữ liệu or dữ liệu chưa xử lý== or dữ liệu đã xử lý
	* Mô hình (học máy) mới
	1. weka và khai phá luật kết hợp
	2. thực hành phân phối lớp
	3. phân cụm
- Nội dung
	1. Tổng quan
		1. Dữ liệu và tri thức
		2. Các mô hình tổ chức dữ liệu
		3. Mô hình biểu diễn tri thức
		4. Học có giám sát và học có giám sát
		5. Giới thiệu 
		6. Ứng dụng
	2. Khai phá luật kết hợp
	3. Bài toán phân lớp
	4. Cây quyết định
	5. Naive Bayes
	6. Neural network
	7. SVM
	8. KNN
	9. Phương pháp kết hợp
	10. Phân cụm và K-means
	11. Cực đại kỳ vọng
- Tài liệu

# 1. Dẫn nhập
## 1.1 Tổng quan
### Bùng nổ dữ liệu
- Từ giao dịch
- Vệ tinh
- Gene
- Kho dữ liệu
- Giám sát
- ...

### Phát hiện tri thức
#### KDD process
https://behavior.lbl.gov/?q=node/11
Why is Tranformation needed in KDD:
1. Loại bỏ nhiễu
2. Chuẩn hoá

```quest
1. Tiền xử lý dữ liệu?
Data cleaning: lấp đầy thiếu hụt dữ liệu, loại bỏ nhiễu, giải quyết sự nhất quán, chiều dữ liệu -> chọn lọc ra những trường dữ liệu quan trọng quyết định

2. Chuyển đổi dữ liệu?
Ảnh -> resize ảnh sao cho phù hợp

3. Vai trò của việc đánh giá?
Accurancy không phản ánh đúng với thực tế, không phù hợp với tất cả bài toán

Bài toán hồi quy và phân lớp:
	Miền: liên tục và rời rạc.
	MSE và accurancy
(miền liên tục không thể sử dụng accurancy)
```
#### Knowledge representations
- table
- linear models: (line in 2D graph)
- trees
- rules
- instance-based representation
- clusters: (each cluster has particalar attributes)
- knowledge model
### Các ứng dụng
Dự đoán, phân tích, phát hiện hành vi, tiếp thị có định hướng
1. Phân tích ảnh vệ tinh: Tìm được mô hình dự báo với dự liệu mới thì nhãn đầu ra là gì?
2. Phát hiện giao dịch bất hợp pháp
3. Dự đoán điện tải
4. Dự đoán tài chính
5. Dự đoán thời tiết
6. Phát hiện hành vi khách hàng
7. Kinh doanh, khám bệnh: xác suất rủi ro, xảy ra
## 1.2 Khai phá
==Khai phá dữ liệu là việc tìm kiếm, khai phá tri thức (hay các mẫu/dạng có nghĩa) trong lượng lớn dữ liệu (bị ẩn).==
1. Dữ liệu nhãn và không nhãn
2. Học có hướng dẫn: phân lớp
3. Học có hướng dẫn: dữ đoán dữ liệu số
4. Học không có hướng dẫn: luật kết hợp. Dựa vào dữ liệu tìm quy luật. (Nếu khách hàng mua chuối và cà chua thì sẽ mua sữa) 
$$\text{banana} ^ \text{tomato} \mapsto \text{milk}$$
6. Học không có hướng dẫn: phân cụm
## 1.3 Định dạng dữ liệu trong dm
### Biểu diễn vật thể trong không gian n chiều
### Các kiểu dữ liệu
1. Nomial categorization
2. binary
3. numeric (interger/real)
4. interval-scaled
5. string
6. date/time
### Sự thiếu hụt
### Chuẩn bị dữ liệu
### Kho dữ liệu UCI
- UCI: uni of california at irvine
## Bài tập
1. Tìm hiểu về Weka
2. Cài đặt Weka
3. Cấu trúc file dữ liệu ARFF trong Weka
4. Các thành phần chính của Weka


# 2. Khai phá luật kết hợp
## Bài toán phân tích giỏ hàng
Dựa trên việc đánh giá các mối quan hệ giữa các mặt hàng mà đặt các hàng hoá nằm cạnh nhau nhằm đạt được lợi ích kinh doanh.
**Bài toán:** 

| TID | Items |
| ---- | ---- |
| 100 | banana, milk, bread |
| 200 | milk, bread, coffee |
| 300 | coffee, milk, shampoo |
**Output:** Liệt kê Các nhóm mặt hàng được mua cùng nhau thường xuyên trong cùng một lần mua hàng. => Thiết kế gian hàng, giảm giá một nhóm các mặt hàng, kế hoạch tiếp thị,...
## Luật kết hợp (association rules)
Cho $I = \{I_1, I_2,..., I_n\}$ là một tập các mục (mặt hàng,...)
Cho $D$ là một tập các giao dịch mà mỗi giao dịch T là một tập các mục
$TID$: mã định danh

Với $A \in I, B \in I, A \cap B = \emptyset$
$$A => B$$

*VD: I = { banana, milk, bread,... }*
*D: bảng dữ liệu trong input [[#Bài toán phân tích giỏ hàng| > Bài toán phân tích giỏ hàng]]*
## Một số độ đo phổ biến
Dựa trên xác suất, độ đo đánh giá sự kết hợp có "tốt" hay không?
1. Độ hỗ trợ: support(A => B) = $P(A\cup B)$: Phần trăm số giao dịch chứa cả mục A và B.
2. Độ tin cậy: confidence(A => B) = $P(B|A)$: Khả năng giao dịch có mục B trên điều kiện giao dịch đã có mục A.
$$P(B|A) = \dfrac{P(A\cup B)}{P(A)}$$
**Luật kết hợp mạnh:** *Những luật kết hợp phải thoả mãn 
min_support <= x && min_confi <= x*
## Tập mục thường xuyên. Frequent itemset
Một tập mục chứa $k$ mục là k-itemset. VD: Đơn hàng {banana, milk} là một 2-itemset.
*Tần suất xuất hiện của một tập mục là số giao dịch chứa tập mục đó: **Frequency, support count, count***
**VD:** D:

| TID | items |
| ---- | ---- |
| 1 | sữa, bánh mì, cam |
| 2 | cam, xoài, nước |
| 3 | táo, lê |
1-itemset: A = {cam}
Frequency of A in D: $\dfrac{2}{3}$

*Tập mục $L_k$ thường xuyên thoả mãn: x >= min_support*

```quest
Từ tập mục thường xuyên, để tìm được lời giải cuối cùng thì các luật kết hợp có chắc chắn là luật mạnh hay không?

```
## Tổng quan về bài toán khai phá luật kết hợp
**Bài toán:** 

| TID | A | B | C | D | E |
| ---- | ---- | ---- | ---- | ---- | ---- |
| 10 | 1 | 1 | 1 | 0 | 0 |
| 20 | 1 | 0 | 1 | 0 | 0 |
| 30 | 1 | 0 | 1 | 1 | 0 |
| 40 | 0 | 1 | 1 | 0 | 1 |
| 50 | 1 | 0 | 1 | 0 | 1 |

| TID | Items |  |
| ---- | ---- | ---- |
| 10 | A, B, C | 11100 |
| 20 | A, C | 10100 |
| 30 | A, C, D | 10110 |
| 40 | B, C, E | 01101 |
| 50 | A, C, E | 10101 |
soát qua bảng trên thấy có tất cả 5 mặt hàng, từ cơ sở dữ liệu biểu diễn vector các mặt hàng được mua trong giao dịch

**Ouput:** Vector các mặt hàng thường được mua cùng nhau (boolean)

==Khai phá luật kết hợp là quá trình gồm 2 bước:==
1. Tìm tất cả các tập mục thường gặp (thường xuyên).
2. Tạo các luật kết hợp mạnh từ các tập mục thường xuyên.

**Kết luận:** *Một tập mục của nó thường xuyên nếu tất cả các tập con trong tập mục đó thường xuyên.*

- Tập mục đóng đóng thường xuyên:
- Tập mục thường xuyên cực đại:
## Thuật Apriori
Tìm ra các tập mục thường xuyên có các luật kết hợp dạng boolean để giải quyết [[#Bài toán phân tích giỏ hàng| > Bài toán phân tích giỏ hàng]]
Chiến lược lặp Apriori: k-itemset => khảo sát (k+1)-itemset

Xây dựng $L_2$ từ các tập thường xuyên $L_1$. (**Kết luận** trong [[#Tổng quan về bài toán khai phá luật kết hợp| > Tổng quan về bài toán khai phá luật kết hợp]]). Sau đó quét qua DB để loại bỏ những $L_2$ không thoả mãn lớn hơn min_support.

**Input:** Tập D các giao dịch, Tần số tối thiểu min_sup_count => Ngưỡng min_support.

| Giao dịch TID | Danh mục |
| ---- | ---- |
| T100 | I1, I2, I5 |
|  |  |
**Ouput:** Tìm các tập mục thường xuyên.

Từ các tập mục thường xuyên kiểm tra điều kiện >= confidence => Luật kết hợp mạnh. VD: từ L_3 = {I1, I2, I5}: 
I1 ^ I2 => I5: $confi(I1 \wedge I2 => T5) = \dfrac{P(I1, I2, I5)}{P(I1, I2)}$,
I1 ^ I5 => I2, ...  


## Sinh luật kết hợp

## Nội dung mở rộng
Một số thuật toán, phiên bản


