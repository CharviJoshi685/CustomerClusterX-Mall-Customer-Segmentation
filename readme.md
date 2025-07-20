# 🛍️ CustomerClusterX

**CustomerClusterX** is a Streamlit-based interactive web app that segments mall customers into distinct groups based on demographics and purchasing behavior. The app also evaluates different regression models to predict customer spending scores.

---

## 📊 Features

- Upload and analyze your **Mall Customer CSV file**
- Automatically **cluster customers** using KMeans
- Visualize customer segments using **PCA + Plotly**
- Get **cluster-wise summaries** (e.g. average income/spending/age)
- Compare **Linear Regression** vs **Random Forest** for predicting spending score

---

## 🧠 Machine Learning Techniques Used

- **Unsupervised Learning**: KMeans Clustering
- **Dimensionality Reduction**: PCA for 2D Visualization
- **Supervised Learning**:
  - Linear Regression
  - Random Forest Regression

---

## 📁 Dataset Used

We use the [Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial) which includes:

- Age
- Gender
- Annual Income (k\$)
- Spending Score (1-100)

You can also use your own dataset in `.csv` format.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/CustomerClusterX.git
cd CustomerClusterX
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run app.py
```

### 4. Upload Dataset

Upload `Mall_Customers.csv` when prompted.

---

## 📈 Output Example

- Interactive cluster scatter plot
- Cluster-wise metrics table
- R² and MSE for regression models

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙋‍♀️ Author

Built with ❤️ by Charvi Joshi


---

## 🤝 Contribution

Feel free to fork the repo, raise issues or submit pull requests!

> *"Data tells a story — cluster it, visualize it, and make it work for you."*

