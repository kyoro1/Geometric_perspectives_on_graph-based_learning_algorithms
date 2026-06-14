import csv

def write_dict_rows_to_csv(path: str, 
                           rows: list[dict], 
                           *, 
                           encoding="utf-8"):
    if not rows:
        raise ValueError("rows is empty")
    with open(path, "w", newline="", encoding=encoding) as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("saved:", path)
