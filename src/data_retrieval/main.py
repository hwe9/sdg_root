from retrieval_worker import RetrievalWorker
import time

if __name__ == "__main__":
    # Konfiguration: Container-Pfade verwenden
    sources_file = "/app/quelle.txt"
    data_dir = "/app/raw_data"
    processed_file = "/app/processed_data/processed_data.json"

    worker = RetrievalWorker(
        sources_file=sources_file,
        data_dir=data_dir,
        processed_file=processed_file
    )

    print("Starting RetrievalWorker...")

    while True:
        try:
            worker.run()
            print("Warte 60 Minuten bis zum nächsten Zyklus...")
            time.sleep(3600)  # 60 Minuten
        except Exception as e:
            print(f"Fehler: {e}")
            time.sleep(60)   # 1 Minute bei Fehler

