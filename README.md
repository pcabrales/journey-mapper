# journey-mapper

Turn a list of stops into a globe-spanning journey. (Made this repo to have a visualization of trips made with my friends and family over the years.)

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
- `latitude`
- `longitude`

Optional columns enrich the tooltip (include any combination of `date`, `notes`, or `description`). Rows render in the order they appear, drawing great-circle style lines from one stop to the next.

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
- Coordinates must be decimal degrees; the script raises an error if it cannot parse them.
