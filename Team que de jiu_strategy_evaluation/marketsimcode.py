"""
An improved version of your marketsim code accepts a “trades” DataFrame (instead of a file). More info on the trades data frame is below. 
It is OK not to submit this file if you have subsumed its functionality into one of your other required code files. 
This file has a different name and a slightly different setup than your previous project. However, that solution can be used with several edits for the new requirements.  
"""

import pandas as pd

def compute_portvals(data: pd.DataFrame, symbol: str, start_val: float = 1_000_000, commission_fee: float = 9.95, market_impact: float = 0.005, use_benchmark: bool = False) -> pd.Series:
    # Initialize the portfolio with cash and no shares
    portfolio = {symbol: 0, 'CASH': start_val}
    
    # Use appropriate orders based on the strategy (benchmark or policy)
    if use_benchmark:
        order_series = data["BenchmarkOrders"].sort_index()
    else:
        order_series = data["Shares"].sort_index()

    # Calculate portfolio values across the entire dataset
    portfolio_values = data.apply(lambda row: calculate_daily_portfolio_value(row, order_series, symbol, portfolio, commission_fee, market_impact), axis=1).dropna()

    return portfolio_values

def calculate_daily_portfolio_value(row: pd.Series, orders: pd.Series, symbol: str, portfolio: dict, commission: float, impact: float) -> float:
    # Apply orders for the day if available
    if row.name in orders.index:
        daily_order = orders.loc[row.name]

        # Update the portfolio with the daily order
        apply_order_to_portfolio(daily_order, row, symbol, portfolio, commission, impact)

    # Calculate total portfolio value for the day (cash + stock holdings)
    total_value = portfolio['CASH']
    for stock in portfolio:
        if stock == 'CASH':
            continue
        total_value += portfolio[stock] * row[stock]

    # Store the portfolio value in the row and return it
    row['PortfolioValue'] = total_value
    return total_value

def apply_order_to_portfolio(order_quantity: int, row: pd.Series, symbol: str, portfolio: dict, commission: float, impact: float):
    # Get the stock price for the current row (day)
    stock_price = row[symbol]

    if order_quantity > 0:
        # If buying shares
        portfolio[symbol] += order_quantity
        stock_price *= (1 + impact)  # Adjust price with market impact (increase for buy)
        portfolio['CASH'] -= order_quantity * stock_price
        portfolio['CASH'] -= commission  # Deduct commission fee

    elif order_quantity < 0:
        # If selling shares
        sell_quantity = -order_quantity  # Convert order to positive for selling
        portfolio[symbol] -= sell_quantity
        stock_price *= (1 - impact)  # Adjust price with market impact (decrease for sell)
        portfolio['CASH'] += sell_quantity * stock_price
        portfolio['CASH'] -= commission  # Deduct commission fee


