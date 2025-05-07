  Books to Scrape – ML Project (Price Prediction & Rating Classification)

  Overview
This project scrapes book data from [Books to Scrape](https://books.toscrape.com/) and performs data analysis and machine learning tasks such as:

-  Predicting book prices using **Linear Regression**
-  Classifying book ratings using **Logistic Regression + NLP (TF-IDF on titles)**
-  Data visualization with **Seaborn** and **Matplotlib**



 🗃 Dataset Details

Data was scraped using `requests` and `BeautifulSoup` from 50 pages, saving 1000+ books into `books_to_scrape.csv`.

| Feature       | Description                      |
|---------------|----------------------------------|
| `Title`       | Title of the book                |
| `Price (£)`   | Price in pounds                  |
| `Rating`      | Star rating (1 to 5)             |
| `Availability`| In stock / Out of stock status   |

---

  Exploratory Data Analysis

  Price Distribution
- Most books are priced between **£20–£50**.
- Few expensive books above £55.

  Rating Distribution
- Most common ratings: **3 and 4 stars**.
- Balanced rating spread, minimal 1-star entries.

![Price Distribution](images/price_distribution.png)  
![Rating Count](images/rating_count.png)



  ML Model 1: Linear Regression

**Objective**: Predict book price using its rating.

- Model: `LinearRegression()`
- Feature: Rating
- Target: Price

  Results:
- **Mean Squared Error**: ~60.4
- **Conclusion**: Rating alone is a weak predictor of price.

![Regression Plot](images/price_vs_rating_regression.png)



  ML Model 2: Logistic Regression (Text Classification)

**Objective**: Predict book rating based on its **title** using **TF-IDF vectorization**.

- Model: `LogisticRegression(max_iter=1000)`
- Input: TF-IDF of title text
- Target: Rating

  Results:
- **Accuracy**: ~54%
- **Best classes**: Ratings 3 & 4
- **Approach**: Title → TF-IDF → Model

![Confusion Matrix](images/confusion_matrix.png)



 🛠 Tools & Libraries

- Python 3.x
- Pandas, NumPy
- BeautifulSoup, Requests
- Scikit-learn
- Seaborn, Matplotlib
- Jupyter Notebook



  Future Improvements

- Add more features (genre, description, author)
- Use deep learning (e.g., BERT for NLP)
- Extend classification to multi-label genres



  How to Run This Project

```bash
git clone [https://github.com/dksh-sood]/book-price-rating-ml.git
cd book-price-rating-ml

# Install required packages
pip install -r requirements.txt

# Run the scraper
python scraper.py

# Open and run the notebook
jupyter notebook analysis.ipynb
