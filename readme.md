#### **Simple Linear Regression GUI (Python)**



A Python GUI application that performs Simple Linear Regression on CSV or Excel datasets.

Users can easily select a dataset, choose the independent (X) and dependent (Y) variables, and generate a regression model with visualization.



The program is designed to handle large datasets (millions of rows) using efficient chunk-based processing.



#### **🚀 Features**



* Simple Graphical User Interface (GUI)
* Supports CSV and Excel files (.csv, .xlsx, .xls)
* Handles large datasets efficiently
* Automatic data cleaning (removes missing or non-numeric values)



**Computes regression metrics:**

&#x09;

* &#x09;R² (Coefficient of Determination)
* &#x09;MAE (Mean Absolute Error)
* &#x09;RMSE (Root Mean Squared Error)
* &#x09;Generates scatter plot with regression line
* &#x09;Displays the regression equation on the graph
* &#x09;Shows predicted Y values (ŷ) on the regression line
* &#x09;Automatically saves the plot





#### **🖥️ How to Use**



1. Run the program (linear\_regression\_app.py)
2. Select your dataset
3. Choose a CSV or Excel file.
4. Select variables (X), Variable (Y)
5. Run regression
6. Click Run Regression.





#### **The program will:**

* Fit the regression model
* Compute statistical metrics
* Display the regression plot
* Save the visualization



#### **Regression Model**



The model fitted is:



𝑦=𝛽1𝑥+𝛽0

&#x09;​



Where:



β₁ = slope



β₀ = intercept



Predicted value:



𝑦^=𝛽1𝑥+𝛽0y^

&#x09;

&#x09;​



#### **⚙️ Installation**



* Clone the repository
* Install dependencies
* pip install pandas numpy matplotlib openpyxl (Optional for .xls files)
* pip install xlrd

📦 Dependencies



* Python 3.8+
* pandas
* numpy
* matplotlib
* tkinter (included with Python)
* openpyxl
* xlrd (optional)



#### 

#### **📁 Output**



The program generates:



* linear\_regression\_plot.png
* The plot contains:
* Data points
* Regression line
* Regression equation
* Predicted Y values



#### **🎯 Use Cases**



* This project is useful for:
* Data analysis
* Statistical modeling
* Regression visualization
* Educational purposes
* Exploratory data analysis



#### **📜 License**



No icense is applicable for this project

