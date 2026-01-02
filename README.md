# journey-mapper

Turn a list of stops into a globe-spanning journey. (Made this repo to have a visualization of trips made with my friends and family over the years.)

![](./builds/global_journey.gif)

Feed the app a CSV that names each stop and its country. Coordinates are optional: when missing, the script looks them up via the Open-Meteo Geocoding API. It then paints a connected route with numbered markers on a world map and exports an interactive HTML map and an optional camera-tour video.

## Quick start

1. Create and activate a fresh virtual environment (recommended).
   ```bash
   python3 -m venv .venv-journey-mapper          # pick a unique folder name if you keep multiple venvs
   source .venv-journey-mapper/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Generate a journey map:
   ```bash
   python journey_mapper.py data/example_journey.csv --output builds/global_journey.html --title "Global Journey"
   ```
4. Open the resulting HTML file in your browser.
5. (Optional) Capture a video flyover between stops:
   ```bash
   python journey_mapper.py data/example_journey.csv --video builds/global_journey.mp4 --title "Global Journey"
   ```
   Video export uses ffmpeg under the hood; install it or add `imageio-ffmpeg` to your environment if you see codec errors.

If you prefer reusing an existing environment, activate it first and make sure it is isolated from other projects before installing the requirements.

## CSV format

The CSV must include these columns:

- `city`
- `country`

Optional columns:

- `latitude`
- `longitude`
- `date`
- `notes`
- `description`

If `latitude`/`longitude` are missing or blank, Journey Mapper will geocode the city and country via Open-Meteo. Provide coordinates if you want to avoid a network call or override an ambiguous match. Rows render in the order they appear, drawing great-circle style lines from one stop to the next.

An example dataset lives at `data/example_journey.csv`.

## Video tour (optional)

- Pass `--video path/to/file.mp4` to export an MP4 where the camera glides from one stop to the next.
- The camera automatically adjusts its zoom: long-haul hops stay wide, while nearby stops get close-up sweeps so you can see the detail.
- Use `--linger` (seconds) to decide how long the camera pauses at each destination.

You can switch to a different projection with `--projection natural earth` (other supported options: `equirectangular`, `mercator`). Width and height may also be tuned with `--width` and `--height`.

## Example

The bundled example produces this kind of visualization:

```bash
python journey_mapper.py data/example_journey.csv --output builds/global_journey.html
```

Add `--video builds/global_journey.mp4` to also generate a camera flyover. Open the HTML map in your browser and play the MP4 to explore the journey from two complementary angles.

- Hover the markers to see extra context pulled from optional columns.

## Troubleshooting
- For non-Latin characters, save your CSV with UTF-8 encoding.
- If geocoding fails, verify your city/country names or provide explicit latitude/longitude.
- Geocoding needs access to `geocoding-api.open-meteo.com`; offline runs require coordinates in the CSV.
- Coordinates must be decimal degrees if provided; the script raises an error if it cannot parse them.

## Deployment (Cloud Run)

Public URL: https://YOUR-SERVICE-URL (fill in after deploy)

How it works: the Docker container runs `server.py` with Gunicorn on Cloud Run. Each run writes its generated HTML/video to `/tmp` inside the container and serves it back via `/map/<run_id>` and `/video/<run_id>`.

Important: `/tmp` is ephemeral and instance-local on Cloud Run. To avoid “Not Found” errors after generating a run, we deploy with `--max-instances 1` and `--concurrency 1` so requests for a given run stay on the same instance.

Note: this trades off horizontal scaling for correctness with ephemeral storage. If you want the service to scale to multiple instances reliably, store outputs in Google Cloud Storage instead of `/tmp`.

### One-time setup

```bash
# Real values are kept in a local file not committed to git: deployment.env
# set -a; source deployment.env; set +a

export PROJECT_ID="your-project-id"
export REGION="europe-southwest1"   # pick a region close to you
export SERVICE_NAME="journey-mapper" # Cloud Run service name

gcloud auth login
gcloud config set project "$PROJECT_ID"

gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

### Build + deploy (manual)

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-southwest1"
export SERVICE_NAME="journey-mapper"

gcloud config set project "$PROJECT_ID"
gcloud builds submit --tag "gcr.io/$PROJECT_ID/$SERVICE_NAME:latest"
gcloud run deploy "$SERVICE_NAME" \
  --image "gcr.io/$PROJECT_ID/$SERVICE_NAME:latest" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 1 \
  --concurrency 1 \
  --timeout 900 \
  --max-instances 1
```

After deploy, fetch the public URL:
```bash
gcloud run services describe "$SERVICE_NAME" \
  --region "$REGION" \
  --format="value(status.url)"
```

### Auto-deploy on push to `main` (Cloud Build)

This repo includes `cloudbuild.yaml`, which builds the Docker image and deploys it to Cloud Run with the same runtime settings as above.

1. In Google Cloud Console: Cloud Build → Triggers → Create trigger
   - Event: “Push to a branch”
   - Branch: `^main$`
   - Configuration: `cloudbuild.yaml`
2. Ensure the Cloud Build runtime service account can deploy:
   ```bash
   PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format="value(projectNumber)")

   # Some projects run Cloud Build deploys as the Compute Engine default SA.
   gcloud projects add-iam-policy-binding "$PROJECT_ID" \
     --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
     --role="roles/run.admin"

   gcloud projects add-iam-policy-binding "$PROJECT_ID" \
     --member="serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
     --role="roles/run.admin"

   gcloud iam service-accounts add-iam-policy-binding \
     "${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
     --member="serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
     --role="roles/iam.serviceAccountUser"
   ```

If Cloud Build fails to push images or write logs, grant these too:
```bash
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format="value(projectNumber)")

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/logging.logWriter"
```
