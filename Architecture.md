# System Architecture

## Current Prototype Architecture
Because this is a 24-hour assignment prototype, we prioritize functionality, visual excellence, and rapid workflow using a localized architecture.

```mermaid
graph TD;
    Client[Browser UI / Vanilla CSS Glassmorphism] -->|Multipart-form Data| Flask[Flask Backend];
    Flask --> Vision[utils/vision.py - OpenCV pipeline];
    Flask --> Estimator[utils/estimation.py - Math Engine];
    Vision -->|Mask pixel count & Processed Image| Estimator;
    Estimator -->|Cost Breakdown & Metrics| Flask;
    Flask -->|Jinja Render| ResultView[UI Result Page];
```

### Components
1. **Frontend**: A highly responsive, modern "Glassmorphism" UI built with Vanilla CSS (`index.css`) and HTML. Prints directly as a report.
2. **Backend**: Python/Flask acting as an API gateway and orchestrator.
3. **Computer Vision Layer**: 
    - **cv2.GrabCut**: Isolates the foreground object (the house) from the background (sky, objects).
    - **Luminance Thresholding**: Prevents painting over windows and deep shadows.
    - **Multiply/Overlay Blending**: Retains original object geometry (bricks, shadows) to make the added Paint or Texture highly realistic.
4. **Estimation & Costing Engine**: Scales pixel count to square footage using reference metrics, automatically appending 10% wastage and multiplying against a predefined local flat-rate material/labor database.

## Ideal Scaled Architecture (Production)
For a future scaling of this project beyond the prototype, the architecture would shift to a microservice-based model leveraging Deep Learning:

```mermaid
graph TD;
    NextJS[React/Next.js Frontend] --> FastAPI[FastAPI Backend];
    FastAPI --> Celery[Celery Job Queue];
    Celery --> SegModel[Pytorch Mask2Former Segmentation];
    FastAPI --> AWS[S3 Bucket - Image Storage];
    FastAPI --> Postgres[PostgreSQL DB - User Accounts, Dynamic Material Rates];
```

- **Segmentation**: Upgrading from OpenCV's GrabCut to an ML model like `Mask2Former` or `DeepLabV3` trained on housing exteriors for precise window/door/wall semantic segmentation.
- **Frontend**: A Next.js PWA providing interactive canvas elements so users can individually paint different walls with different materials dynamically.
- **Database**: A PostgreSQL database where contractors can upload live material rates.
