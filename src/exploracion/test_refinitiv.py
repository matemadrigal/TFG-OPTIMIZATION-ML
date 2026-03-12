import refinitiv.data as rd

# Abrir sesión Platform (conexión directa por internet, sin Workspace desktop)
rd.open_session()

# Test 1: Verificar conexión descargando precio de SPY
print("Test 1: Conexión básica...")
try:
    df = rd.get_history(universe="SPY", start="2024-01-01", end="2024-01-31")
    print(f"  OK - {len(df)} filas descargadas")
    print(df.head(3))
except Exception as e:
    print(f"  ERROR: {e}")

# Test 2: Probar datos de sentimiento con varios códigos
print("\nTest 2: Buscando datos de sentimiento...")
codigos = [
    "TRNA/SENT_SPY",
    "TRNA/SENT_.SPX",
    "AAPLSENT.TRN",
    ".SPXSENT",
    "TRNA/.SPX",
]
for codigo in codigos:
    try:
        df = rd.get_history(universe=codigo, start="2024-01-01", end="2024-01-31")
        print(f"  {codigo}: OK - {len(df)} filas")
        print(df.head(3))
    except Exception as e:
        print(f"  {codigo}: No disponible")

# Test 3: Buscar instrumentos de sentimiento disponibles
print("\nTest 3: Buscando instrumentos de sentimiento...")
try:
    result = rd.discovery.search(query="MarketPsych sentiment", top=10)
    print(result)
except Exception as e:
    print(f"  Búsqueda no disponible: {e}")

rd.close_session()
print("\nTest completado.")
