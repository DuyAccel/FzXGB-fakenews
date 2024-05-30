# FzXGB-fakenews
## English version:
A muliti-labels fake news detection model base on XGBoost and Fuzzy c-Means Clustering
### Dataset:
This model is trained on LIAR dataset. You can get the dataset on [this folder](liar_dataset/) or [this link](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)<br>
![structure of this model](imgs/diagram-Page-10.png)
### Models:
There are 3 types of models for testing:
- [XGboost multi-class classification](XGBoost_multiclass/)
- [XGBoost binary + XGBoost multi-class classification](XGBoost_binary_multi/)
- [Fuzzy XGBoost: Fuzzifier + XGBoost binary + XGBoost multi-class classification **(best score)**](FuzzyXGBoost/)

## Bản Tiếng Việt:
Đây là một mô hình nhận diện tin giả xây dựng trên cơ sở là hai thuật toán XGBoost và Fuzzy c-Means Clustering
### Bộ dữ liệu thực nghiệm:
Bộ dữ liệu được sử dụng có tên là LIAR, bạn có thể tải nó ở [folder này](liar_dataset/) hoặc từ [link này](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)<br>
![structure of this model](imgs/diagram-Page-5.png)
### Mô hình:
Có 3 loại mô hình được dùng để đánh giá chất lượng:
- [XGBoost phân loại đa lớp](XGBoost_multiclass/)
- [XGBoost nhị phân + XGBoost đa lớp](XGBoost_binary_multi/)
- [Fuzzy XGBoost: Bộ làm mờ + XGBoost nhị phân + XGBoost đa lớp **(kết quả tốt nhất)**](FuzzyXGBoost/)

Trong các folder này, file *base.py* chứa các quy trình chung để đọc và xử lý dữ liệu. Nó cũng là nơi lưu trữ những hàm tính toán dùng cho mô hình [(chi tiết)](base.ipynb).
#### [XGBoost phân loại đa lớp:](XGBoost_multiclass/)
Folder này chứa các mô hình phân loại đa lớp xgboost, bao gồm:
- Chỉ dùng statement: [XGB+TXT.ipynb](XGBoost_multiclass/XGB+TXT.ipynb)
- Dùng statement + metadata chữ: [XGB+TXT+CT.ipynb](XGBoost_multiclass/XGB+TXT+CT.ipynb)
- Dùng statement + metadata chữ + metadat số: [XGB+TXT+CT+CH.ipynb](XGBoost_multiclass/XGB+TXT+CT+CH.ipynb)

#### [XGBoost nhị phân + XGBoost đa lớp](XGBoost_binary_multi/)
Folder này chứa các mô hình kết hợp giữa những bộ phân loại nhị phân với phân loại đa lớp, bao gồm:
- Chỉ dùng statement: [TXT.ipynb](XGBoost_binary_multi/TXT.ipynb)
- Dùng statement + metadata chữ: [TXT+CT.ipynb](XGBoost_binary_multi/TXT+CT.ipynb)
- Dùng statement + metadata chữ + metadat số: [TXT+CT+CH.ipynb](XGBoost_binary_multi/TXT+CT+CH.ipynb)

#### [Fuzzy XGBoost: Bộ làm mờ + XGBoost nhị phân + XGBoost đa lớp **(kết quả tốt nhất)**](FuzzyXGBoost/)
Folder này chứa các mô hình kết hợp giữa những bộ phân loại nhị phân với phân loại đa lớp, bao gồm:
- Chỉ dùng statement: [TXT+FZ.ipynb](FuzzyXGBoost/TXT+FZ.ipynb)
- Dùng statement + metadata chữ: [TXT+CT+FZ.ipynb](FuzzyXGBoost/TXT+CT+FZ.ipynb)
- Dùng statement + metadata chữ + metadat số: [TXT+CT+CH+FZ.ipynb](FuzzyXGBoost/TXT+CT+CH+FZ.ipynb)