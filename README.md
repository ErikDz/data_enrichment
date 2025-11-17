# Data Enrichment Service

This service provides a web interface to enrich data from an Excel file.

## Setup

1.  **Build and run the Docker container:**
    ```bash
    docker-compose up --build
    ```

2.  The service will be available at `http://localhost:5000`.

## Web Interface

A web interface is available at `http://localhost:5000` to upload files, start/stop the process, and view logs.

## API Endpoints

*   `POST /upload`
    *   Upload an Excel file for processing. The file should have a sheet named "Analyse" with website URLs in the first column.
    *   **Form data:** `file`

*   `GET /download`
    *   Download the enriched Excel file.

*   `POST /start`
    *   Start the data enrichment process.

*   `POST /stop`
    *   Stop the data enrichment process.

*   `GET /status`
    *   Get the current status of the enrichment process.

## Environment Variables

*   `OPENAI_API_KEY`: Your OpenAI API key for using the LLM features. Create a `.env` file in the `data_enrichment` directory to set this variable.
