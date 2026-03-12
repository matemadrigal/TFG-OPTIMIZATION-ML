import refinitiv.data as rd
import pandas as pd

rd.open_session()

# Test 1: Campos TR de sentimiento para SPY (get_data en vez de get_history)
print("=== Test 1: TR fields de sentimiento ===")
fields_sentiment = [
    "TR.TRNASentiment",
    "TR.TRNASentimentPositive",
    "TR.TRNABuzz",
    "TR.TRNAJoy",
    "TR.TRNAFear",
    "TR.TRNATrust",
    "TR.NewsSentiment",
    "TR.NewsSentimentPositive",
    "TR.NewsSentimentNegative",
]
for f in fields_sentiment:
    try:
        df = rd.get_data(universe=["SPY", "SPY.P", "QQQ.O"], fields=[f])
        if df is not None and not df.empty:
            print(f"  {f}: OK")
            print(df)
    except Exception as e:
        print(f"  {f}: {str(e)[:80]}")

# Test 2: Probar StarMine sentiment model
print("\n=== Test 2: StarMine Models ===")
starmine_fields = [
    "TR.StarMineSentiment",
    "TR.StarMineSmartSentiment",
    "TR.StarMineMMS",
    "TR.StarMineMMSRank",
]
try:
    df = rd.get_data(universe=["SPY.P", "QQQ.O", "IWM.P"], fields=starmine_fields)
    if df is not None and not df.empty:
        print(df)
except Exception as e:
    print(f"  Error: {str(e)[:120]}")

# Test 3: Probar get_history con campos TRNA para índices subyacentes
print("\n=== Test 3: Sentimiento de índices subyacentes ===")
rics_indices = [".SPX", ".IXIC", ".DJI", ".RUT", "GOLD", "US10YT=RR"]
for ric in rics_indices:
    try:
        df = rd.get_history(
            universe=ric,
            fields=["TRNASENT", "TRNASENTP", "TRNABUZZ"],
            start="2024-01-01",
            end="2024-01-31"
        )
        if df is not None and len(df) > 0:
            cols_with_data = [c for c in df.columns if df[c].notna().any()]
            if cols_with_data:
                print(f"  {ric}: OK - {len(df)} filas, datos en: {cols_with_data}")
                print(df[cols_with_data].head(3))
    except Exception as e:
        print(f"  {ric}: {str(e)[:80]}")

# Test 4: News Headlines con sentimiento
print("\n=== Test 4: News con sentimiento ===")
try:
    headlines = rd.news.get_headlines(query="SPY sentiment", count=5)
    print(headlines)
except Exception as e:
    print(f"  News headlines: {str(e)[:120]}")

# Test 5: Buscar más amplio en Discovery
print("\n=== Test 5: Discovery ampliado ===")
queries = [
    "MarketPsych sentiment score",
    "news sentiment daily",
    "TRNA sentiment",
    "StarMine sentiment",
]
for q in queries:
    try:
        result = rd.discovery.search(query=q, top=5)
        if result is not None and not result.empty:
            print(f"\n  Query: '{q}'")
            print(result[["DocumentTitle"]].to_string())
    except:
        pass

rd.close_session()
print("\nExploración completada.")
