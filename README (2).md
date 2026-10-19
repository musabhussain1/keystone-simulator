# Keystone Placement

Supporting files for the Amazon Transcend UAE 2026 submission, Case Study 2, by Syed Musab (American University of Sharjah).

- **Interactive simulation:** https://musabhussain1.github.io/keystone-simulator/ (the file `index.html` in this repository). All products, costs and orders are fictional.
- **Analysis code:** `keystone_analysis.py` produces every number the submission takes from the datasets.
- **Saved outputs:** `report.txt` (the full printed results) and `weekly_results.csv` (results by week and repeat).

## Run the analysis

1. Install Python 3 and the packages: `py -m pip install pandas numpy scipy scikit-learn matplotlib openpyxl`
2. Download the datasets and put the files in one folder with the script. The script searches the folder at any depth.
   - Instacart dataset: https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis (`orders.csv`, `order_products__prior.csv`, `order_products__train.csv`)
   - Olist dataset: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce (`olist_order_items_dataset.csv`, `olist_orders_dataset.csv`, `olist_customers_dataset.csv`)
   - Online Retail dataset: https://doi.org/10.24432/C5BW33 (`Online Retail.xlsx`)
3. Run `py keystone_analysis.py --data-dir .` (on Mac or Linux, use `python3`). The full run takes about 5 minutes. Add `--quick` for a one-minute check with fewer orders; its numbers differ.

The script saves `report.txt`, `weekly_results.csv`, `FigA_bars.png` and `FigB_weekly_build_up.png` in a folder called `keystone_out`.
