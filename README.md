# FzXGB-fakenews
## English version:
A muliti-labels fake news detection model base on XGBoost and Fuzzy c-Means Clustering
### Dataset:
This model is trained on LIAR dataset. You can get the dataset on [this folder](liar_dataset/) or [this link](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)<br>
![structure of this model](imgs/diagram-Page-10.png)
### Models:
There are 3 types of models for testing:
- Single XGboost model for multi-class classification
- Propose method **without** Fuzzifiers
- Propose method **with** Fuzzifiers (best score)

## Bản Tiếng Việt:
Đây là một mô hình nhận diện tin giả xây dựng trên cơ sở là hai thuật toán XGBoost và Fuzzy c-Means Clustering
### Bộ dữ liệu thực nghiệm:
Bộ dữ liệu được sử dụng có tên là LIAR, bạn có thể tải nó ở [folder này](liar_dataset/) hoặc từ [link này](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)<br>
![structure of this model](imgs/diagram-Page-5.png)
### Mô hình:
Có 3 loại mô hình được dùng để đánh giá chất lượng:
- Một lớp XGBoost xử lý bài toán phân loại đa lớp
- Phương án đề xuất **không dùng** bộ làm mờ
- Phương án đề xuất **dùng** bộ làm mờ (kết quả tốt nhất)
