import json

def main():
    with open('pylint_report.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    total = len(data)
    by_type = {}
    for item in data:
        t = item.get('type')
        by_type[t] = by_type.get(t, 0) + 1
    print(f"total:{total}")
    for k, v in by_type.items():
        print(f"{k}:{v}")

if __name__ == '__main__':
    main()
