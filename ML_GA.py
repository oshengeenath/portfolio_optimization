import numpy as np
import pandas as pd
import pygad
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

def load_table(path, dvd=True):
    """
    Load stock price data from CSV files in a directory
    
    Args:
    path (str): Directory path containing stock CSV files
    
    Returns:
    tuple: List of individual stock DataFrames and combined DataFrame
    """
    files = os.listdir(path)
    df_list = list()
    for i in range(len(files)):
        # Read each CSV file, using Date as index
        table = pd.read_csv(path+files[i], index_col='Date', parse_dates=True)[['Close']]
        name = files[i].replace(".csv","")
        table.columns = [name]
        df_list.append(table)
    
    # Combine DataFrames
    df = df_list[0]
    for df_ in df_list[1:]:
        df = pd.concat([df,df_], axis=1)
    
    # Forward fill missing values
    df = df.ffill()
    return df_list, df

def create_time_series_features(df, n_steps=3):
    """
    Create time series features for ML prediction
    
    Args:
    df (pd.DataFrame): Input price DataFrame
    n_steps (int): Number of historical steps to use for prediction
    
    Returns:
    X (np.array): Feature matrix
    y (np.array): Target returns
    """
    X, y = [], []
    
    # Ensure we have price data for the stock
    if df.shape[1] != 1:
        raise ValueError("Input DataFrame should contain data for a single stock")
    
    col_data = df.values.flatten()
    
    for i in range(len(col_data) - n_steps):
        # Ensure we have enough data points
        X.append(col_data[i:i+n_steps])
        # Predict next period return as percentage change
        y.append((col_data[i+n_steps] - col_data[i+n_steps-1]) / col_data[i+n_steps-1] * 100)
    
    return np.array(X), np.array(y)

def train_ml_models(monthly_price):
    """
    Train XGBoost models for each stock to predict returns
    
    Args:
    monthly_price (pd.DataFrame): Monthly stock prices
    
    Returns:
    dict of fitted models and scalers
    """
    models = {}
    
    # Use tqdm for progress tracking during model training
    for stock in tqdm(monthly_price.columns, desc="Training ML Models"):
        # Prepare data for single stock
        X, y = create_time_series_features(monthly_price[[stock]])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        # Reshape and scale features
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        
        # Train XGBoost model
        model = xgb.XGBRegressor(
            n_estimators=100, 
            learning_rate=0.1, 
            random_state=42
        )
        
        model.fit(X_train_scaled, y_train_scaled)
        
        # Store model and scalers
        models[stock] = {
            'model': model,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y
        }
        
        # Optional: Print model performance
        X_test_scaled = scaler_X.transform(X_test)
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        print(f"{stock} Model Performance:")
        print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
        print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}\n")
    
    return models

def predict_returns(models, last_prices, monthly_price):
    """
    Predict returns for each stock using trained ML models
    
    Args:
    models (dict): Trained models for each stock
    last_prices (pd.Series or pd.DataFrame): Last known prices for each stock
    monthly_price (pd.DataFrame): Monthly price data
    
    Returns:
    pd.Series of predicted returns
    """
    # If last_prices is a DataFrame, convert to Series
    if isinstance(last_prices, pd.DataFrame):
        last_prices = last_prices.squeeze()
    
    predicted_returns = {}
    
    for stock, stock_models in models.items():
        # Skip if stock is not in last_prices
        if stock not in last_prices.index:
            predicted_returns[stock] = 0  # Default to 0 return
            continue
        
        model = stock_models['model']
        scaler_X = stock_models['scaler_X']
        scaler_y = stock_models['scaler_y']
        
        try:
            # Ensure using the correct stock prices from monthly_price
            stock_prices = monthly_price[stock].values[-3:].reshape(1, -1)
            input_features_scaled = scaler_X.transform(stock_prices)
            
            # Predict
            return_pred_scaled = model.predict(input_features_scaled)
            return_pred = scaler_y.inverse_transform(return_pred_scaled.reshape(-1, 1)).ravel()[0]
            
            predicted_returns[stock] = return_pred
        except Exception as e:
            predicted_returns[stock] = 0  # Default to 0 return
    
    return pd.Series(predicted_returns)

def create_plots(stock_names, best_weights, fitness_generations):
    """
    Create and save plot for fitness progression
    
    Args:
    stock_names (list): List of stock names
    best_weights (np.array): Optimized portfolio weights
    fitness_generations (list): Fitness progression across generations
    """
    # 1. Portfolio Allocation Pie Chart
    plt.figure(figsize=(10, 8))
    plt.pie(best_weights, labels=stock_names, autopct='%1.1f%%')
    plt.title('Portfolio Allocation', fontsize=15)
    plt.tight_layout()
    plt.savefig('portfolio_allocation_pie.png', dpi=300)
    plt.close()

    # 2. Fitness Progression
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_generations)
    plt.title('Fitness Value Progression', fontsize=15)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.tight_layout()
    plt.savefig('fitness_progression_line.png', dpi=300)
    plt.close()

def main():
    # Folder and date setup
    folder = "Your_Path"
    start_date = '2019-5-29'
    end_date = '2021-7-1'
    stock_names = ['AAPL', 'ADBE', 'AMZN','BTC-USD', 'FB', 'GC=F', '^IRX','MSFT', 'NVDA', 'QCOM','TSLA','TXN']
    risk_free = 1.28

    # Load stock data
    df_list, df = load_table(folder)

    # Create monthly time series
    months = pd.date_range(start=start_date, end=end_date, freq='MS')
    months_df = pd.DataFrame(months, columns=['Date'])
    months_df = months_df.set_index('Date')

    # Resample daily data to monthly, taking the first value of each month
    monthly_price = df.groupby(pd.Grouper(freq='MS')).first()
    monthly_price = monthly_price.reindex(months_df.index)
    monthly_price = monthly_price.ffill()

    # Calculate monthly returns
    monthly_returns = monthly_price.pct_change().dropna() * 100

    # Train ML models for return prediction
    ml_models = train_ml_models(monthly_price)

    # Last known prices for prediction
    last_prices = monthly_price.iloc[-1]

    # Predict returns BEFORE running the genetic algorithm
    predicted_returns = predict_returns(ml_models, last_prices, monthly_price)

    # Initialize fitness_generations as a global list
    global fitness_generations
    fitness_generations = []

    def fitness_func(ga_instance, solution, solution_idx):
        """
        Fitness function with improved constraints
        
        Enforces:
        - Weights sum to 1
        - Non-negative weights
        - Penalizes extreme concentration
        """
        # Normalize weights to sum to 1
        weights = solution / np.sum(solution)
        
        # Constrain weights between 0 and 1
        weights = np.clip(weights, 0, 1)
        
        # Use pre-computed predicted_returns
        portfolio_return = np.sum(weights * predicted_returns)
        
        # Calculate portfolio volatility (historical)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(monthly_returns.cov(), weights)))
        
        # Penalty for concentration (Herfindahl-Hirschman Index inspired)
        concentration_penalty = np.sum(weights**2)
        
        # Calculate Sharpe Ratio (using risk-free rate)
        sharpe_ratio = (portfolio_return - risk_free) / portfolio_volatility
        
        # Combine Sharpe ratio with concentration penalty
        # Multiply by -1 since PyGAD maximizes fitness
        fitness = sharpe_ratio - 0.5 * concentration_penalty
        
        return fitness

    # Genetic Algorithm Parameters
    num_generations = 100
    num_parents_mating = 4
    sol_per_pop = 50
    num_genes = len(stock_names)

    def on_generation(ga_instance):
        best_solution, best_fitness, _ = ga_instance.best_solution()
        fitness_generations.append(best_fitness)
        print(f"Generation {ga_instance.generations_completed}: Best Fitness = {best_fitness}")

    # Create a progress bar with context manager
    with tqdm(total=num_generations, desc="Genetic Algorithm Progress", unit="generation") as pbar:
        ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_func,
        num_genes=num_genes,
        init_range_low=0,
        init_range_high=1,
        mutation_percent_genes=10,
        parent_selection_type="sss",
        crossover_type="single_point",
        mutation_type="random",
        mutation_by_replacement=True,
        on_generation=on_generation,
        sol_per_pop=sol_per_pop,
        # Add elitism parameters
        keep_parents=1,  # Keep the best solution from the previous generation
        keep_elitism=1   
        )

        # Run the genetic algorithm
        ga_instance.run()

    # Get the best solution
    best_solution, best_fitness, _ = ga_instance.best_solution()
    
    # Normalize weights to ensure they sum to 1
    best_weights = best_solution / np.sum(best_solution)
    best_weights = np.clip(best_weights, 0, 1)  # Ensure non-negative weights

    # Create results DataFrame
    results = pd.DataFrame({
        'Stock': stock_names,
        'Weight': best_weights
    })
    results['Percentage'] = results['Weight'] * 100
    results = results.sort_values('Weight', ascending=False)

    # Print results
    print("\nOptimal Portfolio Allocation:")
    print(results)

    # Portfolio Performance
    portfolio_return = np.sum(best_weights * predicted_returns)
    portfolio_volatility = np.sqrt(np.dot(best_weights.T, np.dot(monthly_returns.cov(), best_weights)))
    sharpe_ratio = (portfolio_return - risk_free) / portfolio_volatility

    print(f"\nPortfolio Performance:")
    print(f"Expected Return (ML Prediction): {portfolio_return:.2f}%")
    print(f"Portfolio Volatility: {portfolio_volatility:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    # Create plots
    create_plots(stock_names, best_weights, fitness_generations)

    print("\nPlots have been saved:")
    print("1. portfolio_allocation_pie.png")
    print("2. fitness_progression_line.png")

if __name__ == "__main__":
    main()              