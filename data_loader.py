import re
import time
import random
import requests
import pandas as pd

API = "https://content.guardianapis.com/search"
HEADERS = {"User-Agent": "Mozilla/5.0 (Academic Project)"}

KEYWORDS = [
    "israel", "israeli", "palestine", "palestinian",
    "gaza", "west bank", "hamas", "idf",
    "jerusalem", "settlement", "settlers",
    "ceasefire", "hostage", "hostages",
    "two-state", "occupation"
]

QUERY = "(" + " OR ".join([f'"{k}"' for k in KEYWORDS]) + ")"

def contains_keywords(text):
    t = (text or "").lower()
    return any(k in t for k in KEYWORDS)

def fetch_page(section, page, page_size=200):
    params = {
        "api-key": "test",
        "q": QUERY,
        "section": section,
        "page": page,
        "page-size": page_size,
        "show-fields": "body,headline",
        "order-by": "newest",
        "use-date": "published"
    }
    r = requests.get(API, params=params, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        return None
    return r.json()

def clean_html(html):
    if not html:
        return ""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def collect_from_section(section, label, target, min_words=60, sleep_s=0.2, max_pages=400):
    collected = []
    used = set()
    page = 1

    while len(collected) < target and page <= max_pages:
        j = fetch_page(section, page, page_size=200)
        if not j or "response" not in j:
            break

        results = j["response"].get("results", [])
        if not results:
            break

        added = 0
        for it in results:
            url = it.get("webUrl")
            if not url or url in used:
                continue
            used.add(url)

            fields = it.get("fields", {}) or {}
            title = fields.get("headline") or it.get("webTitle") or ""
            body_html = fields.get("body") or ""
            text = clean_html(body_html)

            if len(text.split()) < min_words:
                continue

            # Keep only conflict-related articles
            if not (contains_keywords(title) or contains_keywords(text)):
                continue

            # ✅ ONLY TWO COLUMNS: text + label
            collected.append({
                "text": text,
                "label": label
            })
            added += 1

            if len(collected) >= target:
                break

        print(f"{section} | page={page} | added={added} | total={len(collected)}")
        page += 1
        time.sleep(sleep_s)

    return collected

random.seed(42)

target_per_class = 2500

neutral_sections = ["world", "middle-east", "us-news", "international"]
biased_sections = ["commentisfree"]

neutral = []
for sec in neutral_sections:
    if len(neutral) >= target_per_class:
        break
    neutral += collect_from_section(
        sec,
        "Neutral",
        target_per_class - len(neutral),
        min_words=60,
        sleep_s=0.2,
        max_pages=500
    )

biased = []
for sec in biased_sections:
    if len(biased) >= target_per_class:
        break
    biased += collect_from_section(
        sec,
        "Biased",
        target_per_class - len(biased),
        min_words=60,
        sleep_s=0.2,
        max_pages=700
    )

if len(neutral) < target_per_class or len(biased) < target_per_class:
    raise RuntimeError(
        f"Not enough collected. Neutral={len(neutral)} Biased={len(biased)}. "
        f"Reduce min_words or add more sources/pages."
    )

df = pd.DataFrame(neutral + biased).sample(frac=1, random_state=42).reset_index(drop=True)

out_file = "guardian_text.csv"
df.to_csv(out_file, index=False, encoding="utf-8")

print(df["label"].value_counts())
print("Saved:", out_file)
print(df.head(3))
