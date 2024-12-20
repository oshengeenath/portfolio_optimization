# Portfolio Optimization Using Genetic Algorithm and Machine Learning  

## Project Overview  
This repository contains the implementation of a Genetic Algorithm (GA) for optimizing financial portfolios. The objective is to allocate portfolio weights that balance competing goals effectively:  

- **Maximize Returns**: Predict and achieve the highest possible returns.  
- **Minimize Risk**: Reduce portfolio volatility while maintaining acceptable returns.  
- **Ensure Diversification**: Prevent over-concentration in a small number of assets.  

The repository includes all necessary scripts and data to reproduce the results.  

---

## How to Run  

**Step 1**: Clone the Repository  

**Step 2**: Install Dependencies  

**Step 3**: Run the ML_GA.py

**Step 6**: Analyze Results  
Visualize portfolio allocation and performance metrics:  

---

## Key Features  

1. **Portfolio Optimization**  
   - **Genetic Algorithm (PyGAD)**: Optimizes portfolio weights to maximize the Sharpe ratio while minimizing risk.  
   - **Constraints**: Ensures portfolio weights sum to 1, are non-negative, and encourage diversification.  

2. **Machine Learning for Return Prediction**  
   - **XGBoost Models**: Predict stock returns based on historical data.  
   - **Scalable Framework**: Handles multiple assets and large datasets efficiently.  

3. **Visualization**  
   - **Portfolio Allocation**: Pie chart showing the weight distribution of assets.  
   - **Fitness Progression**: Line chart displaying Sharpe ratio improvements across generations.  

---

## Sample Results  

### Portfolio Allocation  
| **Stock**   | **Weight** | **Percentage** |  
|-------------|------------|----------------|  
| GC=F        | 0.499461   | 49.95%         |  
| MSFT        | 0.197276   | 19.73%         |  
| NVDA        | 0.169323   | 16.93%         |  
| ^IRX        | 0.157775   | 15.78%         |  
| FB          | 0.008173   | 0.82%          |  

### Portfolio Performance  
- **Expected Return**: 70.58%  
- **Portfolio Volatility**: 15.97%  
- **Sharpe Ratio**: 4.34  

---

## Technologies Used  

- **Python Libraries**:  
  - `pygad`: For implementing the genetic algorithm.  
  - `xgboost`: For training machine learning models.  
  - `pandas` & `numpy`: For data preprocessing and manipulation.  
  - `matplotlib`: For visualizing results.  

- **Data Source**: Historical stock prices loaded from CSV files.  

---

## How It Works  

1. **Data Loading & Preprocessing**:  
   Loads historical stock prices, cleans data, and creates time-series features.  

2. **Machine Learning**:  
   Trains XGBoost models to predict stock returns based on historical data.  

3. **Genetic Algorithm Optimization**:  
   Optimizes portfolio weights to balance returns, risk, and diversification using a fitness function based on the Sharpe ratio.  

4. **Visualization**:  
   Generates visual outputs, including portfolio allocation and fitness progression charts.  
