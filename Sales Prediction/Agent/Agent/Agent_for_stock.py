import pandas as pd
import numpy as np

# Sample data: Stores, Products, Stock, Sales
data = {
    'StoreID': [1, 1, 2, 2, 3, 3],
    'ProdName': ['Product A', 'Product B', 'Product A', 'Product B', 'Product A', 'Product B'],
    'TotalStockQuantity': [100, 150, 200, 80, 90, 60],
    'TotalSalesQuantity': [110, 140, 190, 70, 100, 50]
}

df = pd.DataFrame(data)

# AI Agent to recommend stock distribution
class StockRecommendationAgent:
    def __init__(self, data):
        self.data = data
        self.recommendations = []

    def analyze_stock(self):
        # Calculate stock shortage
        self.data['StockShortage'] = self.data['TotalStockQuantity'] - self.data['TotalSalesQuantity']
        return self.data
    
    def identify_low_high_stock(self):
        # Define thresholds for low and high stock (example thresholds)
        low_stock_threshold = 0  # Changed to 0 to identify actual shortages
        high_stock_threshold = 150

        low_stock = self.data[self.data['StockShortage'] < low_stock_threshold]
        high_stock = self.data[self.data['StockShortage'] > high_stock_threshold]

        return low_stock, high_stock
    
    def generate_recommendations(self, low_stock, high_stock):
        # For each low stock store, check if another store can fulfill the demand
        for _, low in low_stock.iterrows():
            product = low['ProdName']
            low_store_id = low['StoreID']
            
            # Look for high stock stores that have the same product
            matching_high_stock = high_stock[high_stock['ProdName'] == product]
            
            if not matching_high_stock.empty:
                for _, high in matching_high_stock.iterrows():
                    high_store_id = high['StoreID']
                    self.recommendations.append({
                        'LowStockStoreID': low_store_id,
                        'HighStockStoreID': high_store_id,
                        'Product': product,
                        'Recommendation': f'Transfer stock from Store {high_store_id} to Store {low_store_id} for Product {product}'
                    })
                    
    def get_recommendations(self):
        return self.recommendations


# Instantiate and run the AI agent
agent = StockRecommendationAgent(df)

# Step 1: Analyze Stock
agent.analyze_stock()

# Step 2: Identify low and high stock
low_stock, high_stock = agent.identify_low_high_stock()

# Step 3: Generate recommendations
agent.generate_recommendations(low_stock, high_stock)

# Output recommendations
recommendations = agent.get_recommendations()
for rec in recommendations:
    print(f'LowStockStoreID: {rec["LowStockStoreID"]}, HighStockStoreID: {rec["HighStockStoreID"]}, Product: {rec["Product"]}, Recommendation: {rec["Recommendation"]}')