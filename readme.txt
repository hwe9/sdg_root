# Projekt: Wir sind 17 - der Weg zur Weisheit

## Übersicht

Dieses Projekt ist die technische Plattform für das Buch und die digitale Erlebniswelt "Wir sind 17 - der Weg zur Weisheit". Es handelt sich um ein populärwissenschaftliches Werk über die 17 Ziele für nachhaltige Entwicklung (SDGs) der Vereinten Nationen. Die Plattform ist als ein "lebendes, digitales System" konzipiert, das den Inhalt des Buches mit interaktiven, kollaborativen und multimedialen Features erweitert.

Das System ist als eine Microservices-Architektur aufgebaut, die über Docker und Docker Compose orchestriert wird. Jeder Service ist für eine spezifische Aufgabe innerhalb der Daten-Pipeline verantwortlich, von der Datensammlung bis zur Bereitstellung über eine API.

## Architektur und Services

Die Plattform besteht aus den folgenden Services:

### Data Pipeline Services

*   **Retrieval Service:** Identifiziert und lädt automatisch Daten (z.B. wissenschaftliche Publikationen, Artikel, Videos) aus verschiedenen Quellen.
*   **Processing Service:** Bereitet die heruntergeladenen Rohdaten für die weitere Verarbeitung vor. Dies umfasst Aufgaben wie Audiotranskription, Textextraktion aus PDFs, Übersetzung in eine einheitliche Sprache (Englisch) und Aufteilung der Texte in semantische Einheiten ("Chunks").
*   **Vectorization Service:** Wandelt die aufbereiteten Text-Chunks in numerische Vektoren (Embeddings) um, die für semantische Suchen benötigt werden.
*   **Content Extraction Service:** Extrahiert Inhalte aus verschiedenen Quellen und leitet sie an die entsprechenden Services weiter.

### Core Infrastructure Services

*   **Database Service (PostgreSQL):** Speichert Metadaten zu den Inhalten, wie z.B. Quellen, Autoren, und Nutzerdaten.
*   **Weaviate Service & Weaviate Transformer Service:** Eine Vektor-Datenbank, die eine effiziente Speicherung und Abfrage der Text-Vektoren ermöglicht. Der Transformer Service stellt die notwendigen KI-Modelle bereit.
*   **API Service (FastAPI):** Die zentrale Schnittstelle (Gateway) zwischen dem Frontend und den Backend-Services. Sie nimmt Anfragen entgegen, leitet sie an die zuständigen Services weiter und liefert die Ergebnisse zurück.
*   **Redis:** Dient als Task Queue und für Caching, um die Kommunikation zwischen den Services zu managen.

### Utility Services

*   **PgAdmin Service:** Ein web-basiertes Tool zur Verwaltung der PostgreSQL-Datenbank.
*   **Weaviate Console:** Ein Tool zur Interaktion mit der Weaviate-Datenbank.

## Ausführen des Projekts

Das gesamte Projekt kann mit Docker und Docker Compose gestartet werden.

### Voraussetzungen

*   Docker muss installiert sein.
*   Docker Compose muss installiert sein.

### Konfiguration

1.  Das System wird über Umgebungsvariablen konfiguriert. Eine `.env`-Datei wird für die Konfiguration benötigt. Erstellen Sie eine `.env`-Datei im Hauptverzeichnis des Projekts. Sie können sich an der `docker-compose.yml`-Datei orientieren, um die notwendigen Variablen zu identifizieren. Typische Variablen sind:
    *   `POSTGRES_USER`
    *   `POSTGRES_PASSWORD`
    *   `POSTGRES_DB`
    *   `DATABASE_URL`
    *   `WEAVIATE_URL`
    *   `PGADMIN_DEFAULT_EMAIL`

### Starten der Services

Um alle Services zu starten, führen Sie den folgenden Befehl im Hauptverzeichnis des Projekts aus:

```bash
docker-compose up
```

Um die Services im Hintergrund zu starten, verwenden Sie:

```bash
docker-compose up -d
```

### Zugriff auf die Services

*   **API Service:** `http://localhost:8000`
*   **PgAdmin:** `http://localhost:5050`
*   **Weaviate Console:** `http://localhost:3001`
*   **Weaviate API:** `http://localhost:8080`
