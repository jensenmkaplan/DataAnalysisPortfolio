import matplotlib.pyplot as plt
import pandas as pd

# ########################################################################
# # Function to format the labels to show actual revenue values
# def absolute_value(val):
#     total = sum(revenue_by_category)
#     return f'${val * total / 100:.2f}'
# ########################################################################


# Load the dataset
df = pd.read_csv('Online Sales Data.csv')


# Print column names for verification
print(df.columns)

# Group by 'Product Category' and sum the 'Total Revenue'
revenue_by_category = df.groupby('Product Category')['Total Revenue'].sum()



# Create the pie chart
plt.figure(figsize=(10, 6))
plt.pie(revenue_by_category, labels=revenue_by_category.index, autopct=absolute_value, startangle=140)

# Adding title
plt.title('Total Revenue by Product Category')

# Equal aspect ratio ensures that pie is drawn as a circle.
plt.axis('equal')

# Display the chart
plt.show()
