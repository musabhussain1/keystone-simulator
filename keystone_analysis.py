"""
keystone_analysis.py

One script that produces every number the report takes from the datasets.

What it tests
    Keystone Placement picks the products for a forward site by the whole orders they fully
    stock together, and not by sales alone. An order is "fully stocked" when every product in
    it is on the shelf. The script replays the three steps of Section 3 week by week, with at
    most 10% of the shelf spaces changed each week, and scores the shelf on orders it never
    saw while choosing.

How the test is set up
    - Customers are split in two halves. One half chooses the products (choose data).
      The other half scores the shelf (test data). No customer is in both halves.
    - The starting shelf and the benchmark are the best sellers of the choose data.
    - The shelf has the same number of spaces before and after every swap.

Parts of the output
    1  The data
    2  Main result: weekly build-up (Instacart dataset)
    3  Shelf methods side by side
    4  Result by order size
    5  Result with fewer orders
    6  Other datasets (Olist, Online Retail)
    7  Sales forecast check (Online Retail)
    8  Numbers for the report

Usage
    python keystone_analysis.py --data-dir data
    python keystone_analysis.py --data-dir data --quick        (about a minute, for a check)

--data-dir is searched for these files at any depth (a dataset is skipped if its files are missing):
    Instacart   orders.csv, order_products__prior.csv, order_products__train.csv
    Olist       olist_order_items_dataset.csv, olist_orders_dataset.csv, olist_customers_dataset.csv
    Online Retail   Online Retail.xlsx   (needs openpyxl; a CSV copy is cached in --out)
"""

import argparse
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import binomtest

warnings.filterwarnings("ignore")

LOG = []
NAMES = {
    "instacart": "Instacart dataset (a US grocery delivery service)",
    "olist": "Olist dataset (a Brazilian online marketplace)",
    "retail": "Online Retail dataset (a UK wholesaler)",
}
SHORT = {"instacart": "Instacart", "olist": "Olist", "retail": "Online Retail"}


# ---------------------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------------------
def say(text=""):
    print(text, flush=True)
    LOG.append(text)


def title(text):
    say()
    say("=" * 78)
    say(text)
    say("=" * 78)


def table(headers, rows, align=None):
    """Fixed-width table that stays readable in a screenshot."""
    n = len(headers)
    align = align or ["<"] + [">"] * (n - 1)
    widths = [max(len(str(headers[i])), *(len(str(r[i])) for r in rows)) for i in range(n)]
    say("  ".join(f"{str(headers[i]):{align[i]}{widths[i]}}" for i in range(n)))
    say("  ".join("-" * w for w in widths))
    for r in rows:
        say("  ".join(f"{str(r[i]):{align[i]}{widths[i]}}" for i in range(n)))


def one(v):
    text = f"{v:.1f}"
    return "0.0" if text == "-0.0" else text


def mean_range(values):
    """Mean with the lowest and highest run in brackets."""
    return f"{one(np.mean(values))} ({one(np.min(values))} to {one(np.max(values))})"


def signed(v):
    return ("+" if v > 0.05 else "") + one(v)


# ---------------------------------------------------------------------------------------
# Loading the three datasets
# ---------------------------------------------------------------------------------------
class Orders:
    """Orders x products matrix (1 = product in the order) and the customer of each order."""

    def __init__(self, name, matrix, customer, lines=None):
        self.name = name
        self.matrix = matrix.tocsr().astype(np.float32)
        self.customer = np.asarray(customer)
        self.lines = lines          # dated order lines, only kept for Online Retail


def find(root, filename):
    hits = sorted(Path(root).rglob(filename))
    return hits[0] if hits else None


def to_matrix(order_codes, product_codes):
    rows, order_index = np.unique(order_codes, return_inverse=True)
    cols, product_index = np.unique(product_codes, return_inverse=True)
    m = sp.csr_matrix((np.ones(len(order_index), np.float32), (order_index, product_index)),
                      shape=(len(rows), len(cols)))
    m.data[:] = 1
    return m, rows


def load_instacart(root):
    orders_file = find(root, "orders.csv")
    prior = find(root, "order_products__prior.csv")
    if not (orders_file and prior):
        return None
    train = find(root, "order_products__train.csv")
    orders = pd.read_csv(orders_file, usecols=["order_id", "user_id"],
                         dtype={"order_id": np.int32, "user_id": np.int32}).set_index("order_id")
    parts = [pd.read_csv(f, usecols=["order_id", "product_id"],
                         dtype={"order_id": np.int32, "product_id": np.int32})
             for f in (prior, train) if f is not None]
    lines = pd.concat(parts, ignore_index=True).drop_duplicates()
    matrix, order_ids = to_matrix(lines["order_id"].to_numpy(), lines["product_id"].to_numpy())
    return Orders("instacart", matrix, orders.loc[order_ids, "user_id"].to_numpy())


def load_olist(root):
    items_file = find(root, "olist_order_items_dataset.csv")
    orders_file = find(root, "olist_orders_dataset.csv")
    customers_file = find(root, "olist_customers_dataset.csv")
    if not (items_file and orders_file and customers_file):
        return None
    items = pd.read_csv(items_file, usecols=["order_id", "product_id"]).drop_duplicates()
    orders = pd.read_csv(orders_file, usecols=["order_id", "customer_id"])
    people = pd.read_csv(customers_file, usecols=["customer_id", "customer_unique_id"])
    orders = orders.merge(people, on="customer_id").set_index("order_id")
    matrix, order_ids = to_matrix(items["order_id"].to_numpy(), items["product_id"].to_numpy())
    person = pd.factorize(orders.loc[order_ids, "customer_unique_id"])[0]
    return Orders("olist", matrix, person)


def load_retail(root, cache_dir):
    xlsx = find(root, "Online Retail.xlsx")
    cache = Path(cache_dir) / "online_retail.csv"
    if cache.exists():
        raw = pd.read_csv(cache, dtype={"InvoiceNo": str, "StockCode": str}, parse_dates=["InvoiceDate"])
    elif xlsx:
        raw = pd.read_excel(xlsx, dtype={"InvoiceNo": str, "StockCode": str})
        cache.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(cache, index=False)
    else:
        return None
    df = raw.copy()
    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    df = df[~df["InvoiceNo"].str.startswith("C")]             # cancellations
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df = df[df["StockCode"].str.fullmatch(r"\d{5}[A-Z]{0,2}")]  # real products only, not postage or fees
    pairs = df.drop_duplicates(["InvoiceNo", "StockCode"])
    matrix, invoices = to_matrix(pairs["InvoiceNo"].to_numpy(), pairs["StockCode"].to_numpy())
    customer = (pairs.groupby("InvoiceNo")["CustomerID"].first().reindex(invoices)
                .fillna(-1).to_numpy().astype(np.int64))
    return Orders("retail", matrix, customer, lines=df)


# ---------------------------------------------------------------------------------------
# Splitting customers into a choose half and a test half
# ---------------------------------------------------------------------------------------
def split_halves(data, target_orders, seed):
    """Sample whole customers up to about target_orders orders, then split them 50/50."""
    rng = np.random.default_rng([seed, 7])
    known = data.customer >= 0
    people, inverse, per_person = np.unique(data.customer, return_inverse=True, return_counts=True)
    order = rng.permutation(len(people))
    if target_orders is not None and known.sum() > target_orders:
        running = np.cumsum(per_person[order])
        order = order[:int(np.searchsorted(running, target_orders)) + 1]
    half = len(order) // 2
    in_choose = np.zeros(len(people), bool)
    in_test = np.zeros(len(people), bool)
    in_choose[order[:half]] = True
    in_test[order[half:]] = True
    skip = ~known
    choose_rows = np.flatnonzero(in_choose[inverse] & ~skip)
    test_rows = np.flatnonzero(in_test[inverse] & ~skip)
    assert not set(data.customer[choose_rows]) & set(data.customer[test_rows])
    return data.matrix[choose_rows], data.matrix[test_rows]


# ---------------------------------------------------------------------------------------
# Shelves and scoring
# ---------------------------------------------------------------------------------------
def fully_stocked_orders(matrix, shelf):
    """True for each order whose products are all on the shelf."""
    return (matrix @ (~shelf).astype(np.float32)) == 0


def lines_covered(matrix, shelf):
    """Per 100 order lines, how many are for a product on the shelf."""
    return 100 * float((matrix @ shelf.astype(np.float32)).sum() / max(matrix.nnz, 1))


def best_sellers(sales, spaces):
    shelf = np.zeros(len(sales), bool)
    shelf[np.argsort(-sales, kind="stable")[:spaces]] = True
    return shelf


def step_by_step_shelf(choose, spaces, rounds=20, max_missing=4):
    """
    The first method I tried: add products in rounds, each time taking the ones that finish the
    most orders, with orders closer to complete counting more.
    """
    by_product = choose.T.tocsr()
    sales = np.asarray(choose.sum(0)).ravel()
    seen = sales > 0
    shelf = np.zeros(choose.shape[1], bool)
    per_round = max(1, spaces // rounds)
    while shelf.sum() < spaces:
        missing = choose @ (~shelf).astype(np.float32)
        weight = np.where((missing >= 1) & (missing <= max_missing), 1.0 / np.maximum(missing, 1) ** 2, 0.0)
        score = np.asarray(by_product @ weight.astype(np.float32), dtype=np.float64)
        score[shelf | ~seen] = -1
        take = min(per_round, spaces - int(shelf.sum()))
        top = np.lexsort((np.arange(len(score)), -score))[:take]
        top = top[score[top] > 0]
        if len(top) == 0:                       # no order is close to complete: fill with best sellers
            rest = np.flatnonzero(seen & ~shelf)
            rest = rest[np.lexsort((rest, -sales[rest]))][:spaces - int(shelf.sum())]
            shelf[rest] = True
            break
        shelf[top] = True
    return shelf


def keystone_weeks(choose, test, start_shelf, weeks, weekly_limit=0.10):
    """
    Run the three steps of Section 3 once a week on the choose data. Score the shelf on the
    test data after each week.

    Step 1  Rank shelf products by the fully stocked orders that need them. The weakest tenth
            are candidates to leave. Rank the other products by how close the orders they
            belong to are to fully stocked (an order missing 2 products counts half for each).
            Pair the best incoming with the weakest outgoing.
    Step 2  Score a group of pairs together: orders added minus orders lost.
    Step 3  Keep swaps while that score rises, up to the weekly limit.

    Pairs are scored as a group because an order missing 3 products is only fully stocked once
    all 3 arrive. A single swap does not show that.
    """
    shelf = start_shelf.copy()
    spaces = int(shelf.sum())
    limit = max(1, int(round(weekly_limit * spaces)))
    group_size = max(1, limit // 5)
    by_product = choose.tocsc()
    sales = np.asarray(choose.sum(0)).ravel()
    missing = (choose @ (~shelf).astype(np.float32)).astype(np.int32)

    def orders_with(product):
        return by_product.indices[by_product.indptr[product]:by_product.indptr[product + 1]]

    def move(pair, direction):
        incoming, outgoing = pair
        missing[orders_with(incoming)] -= direction
        missing[orders_with(outgoing)] += direction
        shelf[incoming], shelf[outgoing] = direction > 0, direction < 0

    history = [dict(week=0, swaps=0, choose=100 * float((missing == 0).mean()),
                    test=100 * float(fully_stocked_orders(test, shelf).mean()))]
    for week in range(1, weeks + 1):
        # Step 1
        needed = np.asarray(choose[missing == 0].sum(0)).ravel()
        closeness = np.where(missing > 0, 1.0 / np.maximum(missing, 1), 0.0).astype(np.float32)
        progress = np.asarray(choose.T @ closeness).ravel() * (~shelf)
        on_shelf = np.flatnonzero(shelf)
        leaving = on_shelf[np.lexsort((sales[on_shelf], needed[on_shelf]))][:limit]
        coming = np.argsort(-progress, kind="stable")[:limit]
        coming = coming[progress[coming] > 0]
        pairs = list(zip(coming, leaving))

        # Steps 2 and 3
        made, base, start = 0, int((missing == 0).sum()), 0
        while start < len(pairs) and made < limit:
            group = pairs[start:start + min(group_size, limit - made)]
            start += len(group)
            best_count, best_len = base, 0
            for n, pair in enumerate(group, 1):
                move(pair, +1)
                count = int((missing == 0).sum())
                if count > best_count:
                    best_count, best_len = count, n
            for pair in reversed(group[best_len:]):
                move(pair, -1)
            base, made = best_count, made + best_len

        history.append(dict(week=week, swaps=made, choose=100 * float((missing == 0).mean()),
                            test=100 * float(fully_stocked_orders(test, shelf).mean())))
    return history, shelf


BANDS = [("1 product", 1, 1), ("2 to 4 products", 2, 4), ("5 to 9 products", 5, 9), ("10 or more", 10, 10 ** 6)]


def one_run(data, share, target_orders, seed, weeks, with_step_by_step=True):
    """One random sample: build the shelves, replay the weeks, score everything on test data."""
    choose, test = split_halves(data, target_orders, seed)
    sales = np.asarray(choose.sum(0)).ravel()
    spaces = max(1, int(round(share / 100 * int((sales > 0).sum()))))
    start = best_sellers(sales, spaces)
    history, final = keystone_weeks(choose, test, start, weeks)

    full_best = fully_stocked_orders(test, start)
    full_key = fully_stocked_orders(test, final)
    gained = int((full_key & ~full_best).sum())
    lost = int((~full_key & full_best).sum())
    chance = binomtest(gained, gained + lost, 0.5).pvalue if gained + lost else 1.0
    sizes = np.diff(test.indptr)
    bands = {}
    for label, lo, hi in BANDS:
        mask = (sizes >= lo) & (sizes <= hi)
        if mask.sum():
            bands[label] = (100 * float(full_best[mask].mean()), 100 * float(full_key[mask].mean()), int(mask.sum()))

    run = dict(seed=seed, spaces=spaces, test_orders=test.shape[0], choose_orders=choose.shape[0],
               history=history, best=history[0]["test"], key=history[-1]["test"],
               lines_best=lines_covered(test, start), lines_key=lines_covered(test, final),
               gained=gained, lost=lost, chance=chance, bands=bands,
               single_share=100 * float((sizes == 1).mean()))
    if with_step_by_step:
        step = step_by_step_shelf(choose, spaces)
        run["step"] = 100 * float(fully_stocked_orders(test, step).mean())
        run["lines_step"] = lines_covered(test, step)
    return run


def run_many(data, share, target_orders, repeats, weeks, label, started, with_step_by_step=True):
    runs = []
    for r in range(repeats):
        run = one_run(data, share, target_orders, 1 + r, weeks, with_step_by_step)
        runs.append(run)
        say(f"  {label}, shelf {share}%, run {r + 1} of {repeats}: best sellers {one(run['best'])}, "
            f"Keystone {one(run['key'])}  ({time.time() - started:.0f}s)")
    return runs


# ---------------------------------------------------------------------------------------
# Sales forecast check (Online Retail)
# ---------------------------------------------------------------------------------------
def forecast_check(lines, horizon=4, lags=8, first_origin=24, step=4, seed=1):
    """
    Forecast each product's sales for the next 4 weeks and compare the forecast with what sold.
    Forecast error = total miss / total sold (lower is better). Only products that sold in the
    8 weeks before the forecast are scored.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    df = lines.copy()
    df["week"] = df["InvoiceDate"].dt.to_period("W-SUN").dt.start_time
    weeks = sorted(df["week"].unique())[1:-1]               # first and last weeks are partial
    df = df[df["week"].isin(weeks)]
    sales = (df.pivot_table(index="StockCode", columns="week", values="Quantity",
                            aggfunc="sum", fill_value=0).reindex(columns=weeks, fill_value=0))
    m = sales.to_numpy(dtype=np.float64)
    n_weeks = m.shape[1]

    def features(s):
        lag = np.stack([m[:, s - i] for i in range(1, lags + 1)], 1)
        return np.column_stack([lag, lag[:, :4].mean(1), lag.mean(1), (lag > 0).sum(1), lag.max(1)])

    def future(s):
        return m[:, s:s + horizon].sum(1)

    def active(s):
        return m[:, s - lags:s].sum(1) > 0

    records = []
    for t in range(first_origin, n_weeks - horizon + 1, step):
        xs, ys = [], []
        for s in range(lags, t - horizon + 1):
            a = active(s)
            xs.append(features(s)[a])
            ys.append(future(s)[a])
        model = HistGradientBoostingRegressor(loss="absolute_error", max_iter=200, random_state=seed)
        model.fit(np.vstack(xs), np.concatenate(ys))
        a = active(t)
        f = features(t)[a]
        records.append(pd.DataFrame(dict(
            actual=future(t)[a], recent=m[a, t - lags:t].sum(1),
            average4=horizon * f[:, lags], last_week=horizon * f[:, 0], model=model.predict(f))))
    r = pd.concat(records, ignore_index=True)
    r["group"] = pd.qcut(r["recent"].rank(method="first"), 3, labels=["slowest third", "middle third", "fastest third"])

    def error(frame, col):
        return 100 * float(np.abs(frame[col] - frame["actual"]).sum() / frame["actual"].sum())

    rows = []
    for label, frame in [("All products", r)] + [(g, r[r["group"] == g]) for g in ["fastest third", "middle third", "slowest third"]]:
        rows.append([label, f"{len(frame):,}", one(error(frame, "last_week")),
                     one(error(frame, "average4")), one(error(frame, "model"))])
    return rows, dict(all_average4=error(r, "average4"), all_model=error(r, "model"),
                      fast_average4=error(r[r["group"] == "fastest third"], "average4"),
                      fast_model=error(r[r["group"] == "fastest third"], "model"),
                      weeks=len(weeks), products=len(sales))


# ---------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Keystone Placement analysis for the report")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="keystone_out")
    ap.add_argument("--orders", type=int, default=600_000, help="Instacart orders sampled per run")
    ap.add_argument("--small", type=int, default=50_000, help="orders in the fewer-orders check")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--weeks", type=int, default=12)
    ap.add_argument("--shares", type=int, nargs="+", default=[10, 20], help="shelf size, %% of products")
    ap.add_argument("--quick", action="store_true", help="small run to check that everything works")
    args = ap.parse_args()
    if args.quick:
        args.orders, args.small, args.repeats, args.weeks = 30_000, 10_000, 2, 6

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    started = time.time()
    headline, other = args.shares[0], {}
    check_week = min(4, args.weeks)

    title("KEYSTONE PLACEMENT: ANALYSIS RUN")
    say("Each customer is in the choose half or the test half, never both.")
    say("A shelf is scored on the test half only: how many of every 100 orders it fully stocks.")
    say(f"Each simulated week changes at most 10% of the shelf spaces. Weeks simulated: {args.weeks}.")
    say("The cost test of Step 2 is not applied: these datasets have no prices or costs.")

    # ---- load
    loaders = [("instacart", lambda: load_instacart(args.data_dir)),
               ("olist", lambda: load_olist(args.data_dir)),
               ("retail", lambda: load_retail(args.data_dir, out / "cache"))]
    datasets = {}
    for key, loader in loaders:
        try:
            d = loader()
        except Exception as error:      # a missing package or a bad file should not stop the other datasets
            say(f"  {SHORT[key]} skipped: {error}")
            d = None
        if d is None:
            say(f"  {SHORT[key]} files not found: skipped.")
        else:
            datasets[key] = d

    title("1  THE DATA")
    rows = []
    for key, d in datasets.items():
        sizes = np.diff(d.matrix.indptr)
        rows.append([NAMES[key], f"{d.matrix.shape[0]:,}", f"{d.matrix.shape[1]:,}",
                     one(sizes.mean()), one(100 * float((sizes == 1).mean()))])
    table(["Dataset", "Orders", "Products", "Products per order", "Single-product orders (%)"], rows,
          ["<", ">", ">", ">", ">"])

    results = {}

    # ---- Instacart
    if "instacart" in datasets:
        for share in args.shares:
            results[("instacart", share)] = run_many(datasets["instacart"], share, args.orders, args.repeats,
                                                     args.weeks, "Instacart", started)
        main_runs = results[("instacart", headline)]

        title("2  MAIN RESULT: HOW THE SHELF IMPROVES WEEK BY WEEK (Instacart dataset)")
        spaces = main_runs[0]["spaces"]
        say(f"Shelf holds {headline}% of products ({spaces:,} spaces). "
            f"Test orders per run: about {int(np.mean([r['test_orders'] for r in main_runs])):,}.")
        say(f"Mean of {args.repeats} runs, lowest to highest in brackets. All figures per 100 test orders.")
        say()
        weeks_arr = np.array([[h["test"] for h in r["history"]] for r in main_runs])
        swaps_arr = np.array([[h["swaps"] for h in r["history"]] for r in main_runs])
        best_arr = weeks_arr[:, [0]]
        table(["Week", "Swaps made", "Orders fully stocked", "Extra over best sellers"],
              [[("start" if w == 0 else w), "-" if w == 0 else f"{int(swaps_arr[:, w].mean()):,}",
                mean_range(weeks_arr[:, w]), mean_range(weeks_arr[:, w] - best_arr[:, 0])]
               for w in range(args.weeks + 1)])
        say()
        say(f"Reading the table: with best sellers, {one(best_arr.mean())} of every 100 test orders are fully stocked.")
        say(f"After {check_week} weekly rounds of swaps that rises to {one(weeks_arr[:, check_week].mean())}, "
            f"and after {args.weeks} weeks to {one(weeks_arr[:, -1].mean())}.")
        if len(args.shares) > 1:
            say()
            say("Same result for the other shelf sizes:")
            table(["Shelf", "Spaces", "Best sellers", f"Keystone, week {check_week}",
                   f"Keystone, week {args.weeks}", f"Extra, week {args.weeks}"],
                  [[f"{s}% of products", f"{results[('instacart', s)][0]['spaces']:,}",
                    mean_range([r["best"] for r in results[("instacart", s)]]),
                    mean_range([r["history"][check_week]["test"] for r in results[("instacart", s)]]),
                    mean_range([r["key"] for r in results[("instacart", s)]]),
                    mean_range([r["key"] - r["best"] for r in results[("instacart", s)]])]
                   for s in args.shares])

        title("3  THREE WAYS TO FILL THE SHELF, SIDE BY SIDE (Instacart dataset)")
        say(f"Orders fully stocked and order lines covered, per 100 test orders / lines (mean of {args.repeats} runs).")
        say()
        rows = []
        for s in args.shares:
            rr = results[("instacart", s)]
            rows += [
                [f"{s}%", "Best sellers", mean_range([r["best"] for r in rr]), one(np.mean([r["lines_best"] for r in rr]))],
                [f"{s}%", "Step-by-step whole orders", mean_range([r["step"] for r in rr]), one(np.mean([r["lines_step"] for r in rr]))],
                [f"{s}%", f"Keystone, week {args.weeks}", mean_range([r["key"] for r in rr]), one(np.mean([r["lines_key"] for r in rr]))],
            ]
        table(["Shelf", "Method", "Orders fully stocked", "Order lines covered"], rows, ["<", "<", ">", ">"])
        say()
        say("Best sellers rank by sales. Step-by-step adds the products that finish the most orders, round by round.")

        title("4  RESULT BY ORDER SIZE (Instacart dataset)")
        say(f"Shelf holds {headline}% of products. Per 100 test orders of each size, mean of {args.repeats} runs.")
        say()
        rows = []
        for label, _, _ in BANDS:
            have = [r["bands"][label] for r in main_runs if label in r["bands"]]
            if have:
                b, k = np.mean([h[0] for h in have]), np.mean([h[1] for h in have])
                rows.append([label, f"{int(np.mean([h[2] for h in have])):,}", one(b), one(k), signed(k - b)])
        table(["Order size", "Test orders", "Best sellers", "Keystone", "Extra"], rows)

        title("5  DOES THE RESULT HOLD WITH FEWER ORDERS? (Instacart dataset)")
        say(f"Same test with about {args.small:,} orders in each run instead of {args.orders:,}.")
        say()
        small = run_many(datasets["instacart"], headline, args.small, args.repeats, args.weeks,
                         "Instacart small", started, with_step_by_step=False)
        say()
        table(["Orders per run", "Best sellers", f"Keystone, week {args.weeks}", "Extra"],
              [[f"{args.orders:,}", mean_range([r["best"] for r in main_runs]),
                mean_range([r["key"] for r in main_runs]), mean_range([r["key"] - r["best"] for r in main_runs])],
               [f"{args.small:,}", mean_range([r["best"] for r in small]),
                mean_range([r["key"] for r in small]), mean_range([r["key"] - r["best"] for r in small])]])
        other["small"] = small

    # ---- other datasets
    title("6  OTHER DATASETS: DOES THE IDEA CARRY OVER?")
    say(f"Shelf holds {headline}% of products. Every customer is in one half only. Per 100 test orders.")
    say()
    rows, p_rows = [], []
    for key in ("instacart", "olist", "retail"):
        if key not in datasets:
            continue
        if key != "instacart":
            results[(key, headline)] = run_many(datasets[key], headline, None, args.repeats, args.weeks,
                                                SHORT[key], started, with_step_by_step=False)
        rr = results[(key, headline)]
        rows.append([NAMES[key], f"{int(np.mean([r['test_orders'] for r in rr])):,}",
                     one(np.mean([r["single_share"] for r in rr])),
                     mean_range([r["best"] for r in rr]), mean_range([r["key"] for r in rr]),
                     mean_range([r["key"] - r["best"] for r in rr])])
        p_rows.append([SHORT[key], f"{int(np.mean([r['gained'] for r in rr])):,}",
                       f"{int(np.mean([r['lost'] for r in rr])):,}",
                       f"{sum(r['chance'] < 0.05 for r in rr)} of {len(rr)}"])
    say()
    table(["Dataset", "Test orders", "Single-product (%)", "Best sellers", f"Keystone, week {args.weeks}", "Extra"],
          rows, ["<", ">", ">", ">", ">", ">"])
    say()
    say("Orders that Keystone fully stocks and best sellers do not (gained), the reverse (lost),")
    say("and in how many runs the difference is larger than chance would give (p below 0.05):")
    say()
    table(["Dataset", "Gained", "Lost", "Runs beyond chance"], p_rows)

    # ---- forecast
    if "retail" in datasets and datasets["retail"].lines is not None:
        title("7  SALES FORECAST CHECK (Online Retail dataset)")
        say("Forecast each product's sales for the next 4 weeks, then compare with what sold.")
        say("Forecast error: total miss divided by total sold, as a percentage. Lower is better.")
        say()
        try:
            rows, fc = forecast_check(datasets["retail"].lines, first_origin=24 if not args.quick else 20)
            table(["Products scored", "Count", "Last week x 4", "4-week average", "Trained model"], rows,
                  ["<", ">", ">", ">", ">"])
            other["forecast"] = fc
        except Exception as error:
            say(f"  Forecast check skipped: {error}")

    # ---- summary
    title("8  NUMBERS FOR THE REPORT")
    rows = []
    if ("instacart", headline) in results:
        for s in args.shares:
            rr = results[("instacart", s)]
            rows += [
                [f"Instacart, {s}% shelf", "Best sellers, orders fully stocked per 100", mean_range([r["best"] for r in rr])],
                [f"Instacart, {s}% shelf", f"Keystone after {check_week} weeks", mean_range([r["history"][check_week]["test"] for r in rr])],
                [f"Instacart, {s}% shelf", f"Keystone after {args.weeks} weeks", mean_range([r["key"] for r in rr])],
                [f"Instacart, {s}% shelf", f"Extra after {args.weeks} weeks", mean_range([r["key"] - r["best"] for r in rr])],
                [f"Instacart, {s}% shelf", "Step-by-step whole orders", mean_range([r["step"] for r in rr])],
                [f"Instacart, {s}% shelf", "Order lines covered: best sellers / Keystone",
                 f"{one(np.mean([r['lines_best'] for r in rr]))} / {one(np.mean([r['lines_key'] for r in rr]))}"],
            ]
    for key in ("olist", "retail"):
        if (key, headline) in results:
            rr = results[(key, headline)]
            rows.append([f"{SHORT[key]}, {headline}% shelf", f"Extra after {args.weeks} weeks",
                         mean_range([r["key"] - r["best"] for r in rr])])
    if "small" in other:
        rows.append([f"Instacart, {args.small:,} orders", f"Extra after {args.weeks} weeks",
                     mean_range([r["key"] - r["best"] for r in other["small"]])])
    if "forecast" in other:
        f = other["forecast"]
        rows += [["Forecast error, all products", "4-week average / trained model",
                  f"{one(f['all_average4'])} / {one(f['all_model'])}"],
                 ["Forecast error, fastest third", "4-week average / trained model",
                  f"{one(f['fast_average4'])} / {one(f['fast_model'])}"]]
    table(["Where", "What", "Value (mean, lowest to highest)"], rows, ["<", "<", ">"])
    say()
    say(f"Finished in {(time.time() - started) / 60:.1f} minutes.")
    say(f"Saved in {out}/: report.txt, weekly_results.csv, FigA_bars.png, FigB_weekly_build_up.png")

    (out / "report.txt").write_text("\n".join(LOG), encoding="utf-8")
    records = []
    for (key, share), rr in results.items():
        for r in rr:
            for h in r["history"]:
                records.append(dict(dataset=key, share=share, seed=r["seed"], spaces=r["spaces"], **h))
    pd.DataFrame(records).to_csv(out / "weekly_results.csv", index=False)
    if ("instacart", headline) in results:
        draw_figures(args, results, out, headline)


def draw_figures(args, results, out, headline):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grey, blue, ink = "#8A8F98", "#1F5AA6", "#222222"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "#BBBBBB",
                         "axes.spines.top": False, "axes.spines.right": False})

    # Figure A: best sellers against Keystone for each shelf size
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=200)
    xs, width = np.arange(len(args.shares)), 0.34
    best = [np.mean([r["best"] for r in results[("instacart", s)]]) for s in args.shares]
    key = [np.mean([r["key"] for r in results[("instacart", s)]]) for s in args.shares]
    ax.bar(xs - width / 2, best, width, color=grey, label="Best sellers")
    ax.bar(xs + width / 2, key, width, color=blue, label=f"Keystone, week {args.weeks}")
    for x, v in zip(xs - width / 2, best):
        ax.text(x, v + 0.8, f"{v:.1f}", ha="center", color=ink)
    for x, v in zip(xs + width / 2, key):
        ax.text(x, v + 0.8, f"{v:.1f}", ha="center", color=ink)
    ax.set_xticks(xs, [f"{s}% of products\non the shelf" for s in args.shares])
    ax.set_ylabel("Orders fully stocked, in every 100 orders")
    ax.set_ylim(0, max(max(best), max(key)) * 1.2 + 1)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Fully stocked orders on test data (Instacart dataset)", loc="left", fontsize=12)
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.16, top=0.9)
    fig.savefig(out / "FigA_bars.png")
    plt.close(fig)

    # Figure B: extra orders fully stocked, week by week
    runs = results[("instacart", headline)]
    extra = np.array([[h["test"] - r["history"][0]["test"] for h in r["history"]] for r in runs])
    x = np.arange(extra.shape[1])
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
    ax.fill_between(x, extra.min(0), extra.max(0), color=blue, alpha=0.15, linewidth=0)
    ax.plot(x, extra.mean(0), color=blue, linewidth=2)
    ax.axhline(3, color=grey, linewidth=1, linestyle="--")
    ax.text(x[-1], 3.1, "pilot target: 3", ha="right", va="bottom", color=ink, fontsize=10)
    ax.set_xlabel("Week")
    ax.set_ylabel("Extra orders fully stocked, in every 100")
    ax.set_title(f"Gain over best sellers, {headline}% shelf, 10% weekly limit", loc="left", fontsize=12)
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.16, top=0.9)
    fig.savefig(out / "FigB_weekly_build_up.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
