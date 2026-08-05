# NeuralForge (Model Factory)

NeuralForge is a lightweight, cloud-assisted backend engine designed to generate and customize specialized AI LoRA adapters and model configurations on demand. It bridges user project specifications with offline mobile environments (such as AEGES-CORE-POCKET) without requiring expensive local hardware or paid cloud compute fees.

## Key Features
* **CPU-Optimized LoRA Training:** Fine-tunes micro-models (like TinyLlama-1.1B) safely within lightweight container limits.
* **Live Web Knowledge Injection:** Integrated DuckDuckGo search scraping to feed real-time topic information directly into the model customization workflow.
* **Automatic Data Hygiene:** Stateless micro-jobs with built-in auto-purge rules to clear logs and protect resources.
* **Integrity Validation:** Generates file checksum hashes to ensure zero corruption during mobile download handshakes.

## API Endpoints
* `POST /api/build-model`: Accepts project parameters and triggers asynchronous adaptation.
* `GET /api/status/{project_id}`: Checks build progress, retrieves download routes, or purges expired sessions past the retention window.
