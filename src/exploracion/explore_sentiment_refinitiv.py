import refinitiv.data as rd

rd.open_session()

# Buscar qué RICs de MarketPsych / sentimiento existen para nuestros 10 ETFs
etfs = ["SPY", "QQQ", "IWM", "EEM", "EFA", "AGG", "LQD", "GLD", "TIP", "VNQ"]

# Test 1: Probar MarketPsych news sentiment para cada ETF
print("=== Test MarketPsych por ETF ===")
for etf in etfs:
    rics_prueba = [
        f"TRNA/{etf}",
        f"{etf}.O",  # NASDAQ
        f"{etf}.P",  # NYSE Arca
    ]
    for ric in rics_prueba:
        try:
            df = rd.get_history(
                universe=ric,
                fields=["TRNASENT", "TRNASENTP", "TRNABUZZ"],
                start="2024-01-01",
                end="2024-01-31"
            )
            if df is not None and len(df) > 0:
                print(f"  {ric}: OK - {len(df)} filas, columnas: {list(df.columns)}")
                print(df.head(2))
                break
        except:
            pass

# Test 2: Probar índices de sentimiento generales del mercado
print("\n=== Test índices generales ===")
indices = [
    "TRNA/.SPX",
    "TRNA/US",
    "TRNABUZZ/.SPX",
    ".MRPSSENT",
    "MRPSSENTUS"
]
for idx in indices:
    try:
        df = rd.get_history(universe=idx, start="2024-01-01", end="2024-01-31")
        if df is not None and len(df) > 0:
            print(f"  {idx}: OK - {len(df)} filas")
            print(f"  Columnas: {list(df.columns)}")
    except Exception as e:
        print(f"  {idx}: No disponible")

# Test 3: Buscar con Discovery
print("\n=== Búsqueda Discovery ===")
try:
    result = rd.discovery.search(query="MarketPsych sentiment ETF", top=15)
    print(result)
except Exception as e:
    print(f"Discovery: {e}")

rd.close_session()
