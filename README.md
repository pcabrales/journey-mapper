# journey-mapper

Turn a list of stops into a globe-spanning journey visual. 

![](./builds/global_journey.gif)

Feed the app a CSV that names each stop, its country, and its coordinates, and it will paint a connected route with numbered markers on a world map then export an interactive HTML map and an optional camera-tour video.

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
3. (Optional) Provide a Mapbox token if you want the satellite basemap or a video flyover with real imagery (see "Providing a Mapbox token"):
   ```bash
   export MAPBOX_TOKEN="pk.your_token_here"
   ```
4. Generate a journey map:
   ```bash
   python journey_mapper.py data/example_journey.csv --output builds/example_journey.html --title "Around the World in 10 Stops"
   ```
5. Open the resulting HTML file in your browser.
6. (Optional) Capture a video flyover between stops:
   ```bash
   python journey_mapper.py data/example_journey.csv --map-style satellite --video builds/example_journey.mp4 --title "Around the World in 10 Stops"
   ```
   Video export uses ffmpeg under the hood; install it or add `imageio-ffmpeg` to your environment if you see codec errors.

If you prefer reusing an existing environment, activate it first and make sure it is isolated from other projects before installing the requirements.

## Providing a Mapbox token

1. Create a free Mapbox account at https://account.mapbox.com/ if you do not already have one.
2. In your Mapbox dashboard, copy an access token (the default public token works for this project).
3. Supply that token to the app in one of two ways:
   - **Environment variable (ideal for repeated runs):**
     ```bash
     export MAPBOX_TOKEN="pk.your_token_here"
     python journey_mapper.py data/example_journey.csv --map-style satellite --output builds/example_satellite.html
     ```
   - **Command-line flag (handy for ad-hoc runs or multiple tokens):**
     ```bash
     python journey_mapper.py data/example_journey.csv --map-style satellite --mapbox-token "pk.your_token_here" --output builds/example_satellite.html
     ```

If no token is provided, the script automatically falls back to the stylized globe view so you can still generate the HTML and video outputs.

## CSV format

The CSV must include these columns:

- `city`
- `country`
- `latitude`
- `longitude`

Optional columns enrich the tooltip (include any combination of `date`, `notes`, or `description`). Rows render in the order they appear, drawing great-circle style lines from one stop to the next.

An example dataset lives at `data/example_journey.csv`.

## Satellite basemap (optional)

- Set `--map-style satellite` and supply a Mapbox access token (either via `--mapbox-token` or the `MAPBOX_TOKEN` environment variable).
- Without a token the script gracefully falls back to the stylized globe.
- Map markers stay numbered and hover cards keep the optional metadata even on the satellite view.

## Video tour (optional)

- Pass `--video path/to/file.mp4` to export an MP4 where the camera glides from one stop to the next.
- The camera automatically adjusts its zoom: long-haul hops stay wide, while nearby stops get close-up sweeps so you can see the detail.
- Use `--fps` to control playback smoothness and `--linger` (seconds) to decide how long the camera pauses at each destination.
- Pick the recording basemap with `--video-map-style`: `auto` (default), `styled`, or `satellite`. This lets you force the styled globe to avoid Mapbox warnings in network-restricted environments.
- Video rendering requires Kaleido (for Plotly image export) and ImageIO with ffmpeg support. The provided `requirements.txt` includes both `kaleido` and `imageio-ffmpeg`.
- If Mapbox tiles are unreachable while recording (e.g., offline rendering), the script automatically falls back to the styled globe for video frames so the export still succeeds.

If no satellite token is available, the video still renders using the stylized globe, orbiting and zooming appropriately between nearby locations.

## Styling choices

- Projection defaults to an orthographic globe centered on the journey when the styled map is in use.
- Numbered, color-rich markers make the travel order obvious.
- Hover the markers to see extra context pulled from optional columns.
- For the satellite view, the route is rendered with neon lines and warm markers to pop against real imagery.

You can switch to a different projection with `--projection natural earth` (other supported options: `equirectangular`, `mercator`). Width and height may also be tuned with `--width` and `--height`.

## Example

The bundled example produces this kind of visualization:

```bash
python journey_mapper.py data/example_journey.csv --output builds/example_journey.html
```

Add `--video builds/example_journey.mp4 --map-style satellite` to also generate a camera flyover. Open the HTML map in your browser and play the MP4 to explore the journey from two complementary angles.

## Troubleshooting

- If you see an error about missing modules, ensure `pandas`, `plotly`, `kaleido`, and `imageio` are installed (reinstall with `pip install -r requirements.txt`).
- For non-Latin characters, save your CSV with UTF-8 encoding.
- Coordinates must be decimal degrees; the script raises an error if it cannot parse them.
- Codec complaints when exporting video usually mean ffmpeg is not available. Installing `imageio-ffmpeg` (already listed in `requirements.txt`) or system ffmpeg solves it.
