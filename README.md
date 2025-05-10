## 🧭 OVERVIEW: LEARNING PLAN STRUCTURE

| Phase | Focus Area                  | Goal                                       | Time     |
| ----- | --------------------------- | ------------------------------------------ | -------- |
| 1️⃣   | System Architecture         | Understand how the system fits together    | 1 day    |
| 2️⃣   | Resume Parsing & Gemini LLM | Master the pipeline from upload → JSON     | 2.5 days |
| 3️⃣   | NLP Classifier & Matching   | Understand role classification + matching  | 2 days   |
| 4️⃣   | Backend & API               | Learn how FastAPI endpoints work           | 1.5 days |
| 5️⃣   | Frontend & UI               | Learn what the user sees + how UI connects | 1.5 days |
| 🔁    | Revision & Mock Review      | Practice file walkthroughs + Q\&A sessions | 2.5 days |

*(Time estimates adjusted for depth)*

---

You are absolutely right! My apologies. I got carried away with the content of Module 2 and didn't adhere to the new, more detailed structure you requested for *each* module.

Let's restart with **Module 1: System Architecture** and apply the full, detailed template you've outlined.

---

Okay, I understand. You want an even more exhaustive, "triple-length, double-depth" version of Module 1, with more emojis and highlighting, and we'll remove references to Elasticsearch and RabbitMQ/Celery as they weren't in the final implemented version, focusing purely on what *is* in the project according to the plan and file structure.

Let's rebuild Module 1 with that extreme level of detail.

---

## 💎 Module 1: System Architecture (Definitive Deep Dive & Orchestration)

### 🎯 **Core Objectives for Eva:**

*   🧠 **Develop an Unshakeable Mental Model:** Visualize the entire system as a cohesive unit, understanding precisely how each microservice, database, and the frontend interact.
*   🗣️ **Articulate with Precision:** Explain the *purpose (the "Why")* and *functionality (the "How")* of every major component, justifying design choices with clear, technical reasoning.
*   🗺️ **Navigate the Codebase Confidently:** Mentally (and physically, if asked) connect high-level architectural concepts to specific directories and key files within the project structure.
*   💡 **Understand Data Flow Dynamics:** Trace the journey of data for primary user actions (e.g., resume upload, match request) through all involved services and databases.
*   ⚙️ **Grasp Orchestration & Configuration:** Explain how Docker Compose launches, networks, and configures the services, and how environment variables control behavior.
*   🛡️ **Identify Key Strengths & Trade-offs:** Discuss the advantages of the microservice and hybrid database approach, while also acknowledging potential complexities.

---

### 🚀 Technologies Involved (System-Wide Keystone Technologies)

| Category                     | Technology                                                                  | 🌟 Purpose & Significance in Overall Architecture                                                                                                                                                                                                                                                                                             |
| :--------------------------- | :-------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **🌊 Frontend Framework**      | React (via Next.js)                                                         | **Interactive User Interface**: Builds dynamic, component-based UIs. Next.js adds crucial features like file-system routing (`pages/`), Server-Side Rendering (SSR) for better initial load performance & SEO, and a robust development environment.                                                                                                     |
| **💅 Frontend Styling**        | Tailwind CSS                                                                | **Utility-First CSS**: Enables rapid development of custom, responsive designs directly in the markup, ensuring visual consistency with minimal custom CSS.                                                                                                                                                                                       |
| **📦 Frontend State**        | Zustand                                                                     | **Global State Management**: A lightweight, unopinionated state manager for React. Used to hold application-wide data like lists of resumes, jobs, current match results, and API loading/error states, making them accessible across components.                                                                                                       |
| **🔗 Frontend API Client**   | Axios                                                                       | **HTTP Requests**: A promise-based HTTP client used in `frontend/src/services/api.ts` to make calls to the Backend API Gateway. Handles request/response interception (e.g., for adding auth tokens).                                                                                                                                               |
| **⚙️ Backend Framework**     | Python & FastAPI                                                             | **API Development**: All backend microservices are built with Python. FastAPI is a modern, high-performance web framework chosen for its native `async` support (crucial for I/O-bound tasks like calling LLMs or DBs), automatic data validation with Pydantic, and auto-generated API docs.                                                               |
| **🔄 ASGI Server**           | Uvicorn                                                                     | **Running FastAPI**: An ASGI (Asynchronous Server Gateway Interface) server required to run FastAPI applications, handling concurrent requests efficiently.                                                                                                                                                                                              |
| **📄 PDF Processing**        | PyMuPDF (fitz)                                                              | **PDF Text & Layout Extraction**: Used by the Document Processor to extract detailed information from PDFs, including text blocks, their bounding box coordinates, font attributes (size, bold), and page numbers. This layout data is essential for accurate section identification.                                                              |
| **📝 DOCX Processing**       | `python-docx`                                                               | **DOCX Text Extraction**: Used by the Document Processor to extract textual content from `.docx` files, including paragraphs and table content.                                                                                                                                                                                                        |
| **🤖 LLM API Integration**  | Google Gemini API (via `google-generativeai` Python SDK) & `httpx` (async) | **Structured Data Extraction**: The Document Processor uses Gemini (specifically `gemini-1.5-flash-latest`) to convert unstructured text sections from resumes into structured JSON. `httpx` is used by the SDK (or directly if custom calls were made) for asynchronous API communication.                                                                |
| **📜 LLM Prompting**        | XML Templates                                                               | **Guiding the LLM**: Prompts are defined in XML format (`services/document_processor/prompts/`) to provide clear, structured instructions and examples to Gemini, ensuring more consistent and accurate JSON output for different resume sections.                                                                                                     |
| **🧠 NLP Embeddings**        | `sentence-transformers` (e.g., `all-MiniLM-L6-v2`)                          | **Semantic Representation**: Used by the NLP Classifier service to convert resume text (summaries, job titles, skills) into dense numerical vectors (embeddings) that capture semantic meaning. These embeddings are a key input to the role classification model.                                                                                            |
| **🤖 NLP Classification**   | PyTorch, Scikit-learn, NumPy                                                | **Role Prediction**: The NLP Classifier uses a custom PyTorch neural network (`ResumeMultiHeadClassifier`) that combines text embeddings and structured features. Scikit-learn is used for utilities like `LabelEncoder` and evaluation metrics. NumPy handles numerical array operations.                                                            |
| **🧩 Matching Algorithms**   | Scikit-learn (`TfidfVectorizer`, `cosine_similarity`), NumPy                | **Similarity Calculation**: The Matching Service uses TF-IDF to vectorize text (resume summary, job description) and cosine similarity to measure keyword-based similarity. Jaccard similarity (custom Python) for skill set overlap. Numerical comparisons for experience/education.                                                               |
| **🗃️ Relational Database** | PostgreSQL (via SQLAlchemy & `psycopg2-binary`)                             | **Structured Data Storage**: Stores user accounts, job descriptions (metadata, key fields), skills, resume metadata (linking to MongoDB), and match records. SQLAlchemy provides an ORM for Python interaction. Alembic manages schema migrations.                                                                                                       |
| **📄 Document Database**     | MongoDB (via `motor`)                                                       | **Flexible JSON Storage**: Stores the full, potentially deeply nested and variably structured JSON output from the resume parser (Document Processor). `motor` is the asynchronous Python driver used by the FastAPI backend.                                                                                                                       |
| **🔒 Authentication**       | JWT (JSON Web Tokens) via `python-jose` & `passlib`                         | **User Security**: The Backend API Gateway uses JWTs for stateless user authentication. `passlib` (with `bcrypt`) is used for securely hashing passwords. `python-jose` handles JWT encoding/decoding.                                                                                                                                               |
| **✅ Data Validation**       | Pydantic (Backend API), `jsonschema` (Document Processor)                 | **Ensuring Data Integrity**: Pydantic models validate API request/response bodies in FastAPI services. `jsonschema` validates the Document Processor's final JSON output against `base_output_schema.json`.                                                                                                                                               |
| **🐳 Containerization**     | Docker                                                                      | **Packaging Services**: Each microservice (frontend, backend API, document processor, NLP, matching) and database is packaged into a lightweight, portable Docker container with all its dependencies, ensuring consistency across environments.                                                                                                    |
| **🚢 Orchestration**       | Docker Compose                                                              | **Managing Multi-Container Application**: `docker-compose.yml` defines how all the Docker containers are built, networked, configured (ports, volumes, environment variables), and run together as a single application. It simplifies development and deployment.                                                                         |

---

### 🚶 Step-by-Step High-Level System Flow (Elaborated Example: Full Resume Ingestion & Match Display)

This flow details the journey from a user uploading a resume to seeing job matches, touching all key services.

1.  **👤 User Initiates Resume Upload (Frontend Service - Port 3000)**
    *   Eva accesses the application, likely landing on `frontend/src/pages/index.tsx`.
    *   She navigates to the "Upload Resume" page (`frontend/src/pages/upload.tsx`).
    *   An interactive UI (possibly using `react-dropzone`) allows her to select or drag-and-drop a resume file (e.g., `MyCV.pdf`).
    *   Client-side JavaScript performs initial validation (file type, size).
    *   Eva clicks "Submit/Upload".

2.  **📡 API Call to Backend Gateway (Frontend → Backend API Service - Port 8000)**
    *   The frontend's `uploadResume` function (likely in `frontend/src/store/useAppStore.ts` which calls `frontend/src/services/api.ts`) constructs a `FormData` object containing the file.
    *   An Axios `POST` request is sent to the Backend API Gateway's `/api/v1/resumes` endpoint.
    *   Crucially, an **Authorization header** with a **JWT (Bearer token)**, previously obtained during login and stored in `localStorage`, is automatically attached by an Axios interceptor in `api.ts`.

3.  **🔐 Authentication & Initial Handling (Backend API Gateway - `backend/`)**
    *   The `/api/v1/resumes` endpoint in `backend/app/api/v1/endpoints/resumes.py` receives the request.
    *   The `Depends(get_current_active_user)` dependency (`backend/app/core/security.py`) validates the JWT. If invalid/expired, a 401 Unauthorized error is returned. If valid, the user's identity is confirmed.
    *   The uploaded file is temporarily saved to the server's filesystem (e.g., into a `/temp_uploads` directory).

4.  **📞 Orchestration: Calling Document Processor (Backend API Gateway → Document Processor Service - Port 8001)**
    *   The Backend API Gateway, using its `httpx.AsyncClient` (from `backend/app/core/http_client.py`), makes an asynchronous `POST` request to the **Document Processor** service's `/process-resume` endpoint.
    *   The URL for the Document Processor (e.g., `http://document_processor:8001`) is retrieved from environment variables (via `backend/app/core/config.py`).
    *   The resume file is forwarded in this request.

5.  **📄 Document Parsing & Structuring (Document Processor Service - `services/document_processor/`)**
    *   The `/process-resume` endpoint in `src/api/main.py` receives the file.
    *   **Text/Layout Extraction (`src/core/extractor.py`):**
        *   If PDF, `extractors/pdf.py` (`extract_pdf_data_pymupdf`) uses PyMuPDF to get text blocks, bounding boxes, fonts.
        *   If DOCX, `extractors/docx.py` (`extract_text_from_docx`) uses `python-docx` for paragraph text.
    *   **Layout Analysis (`src/layout/layout_analyzer.py` - for PDF):** `LayoutProcessor` uses regex (`layout/patterns.py`) and heuristics (font size, boldness) on PDF blocks to identify sections (e.g., "Experience", "Education").
    *   **Sectional LLM Processing (`src/core/parser.py` orchestrates `src/text_processing/` modules):**
        *   For each identified section (or full text for fallbacks), the corresponding `LLMProcessor` in `text_processing/base_processor.py` is invoked.
        *   `llm/prompt.py` loads the specific XML prompt template (e.g., `prompts/experience.xml`).
        *   The section text and a schema example are injected into the prompt.
        *   `llm/client.py` (`generate_structured_json_async`) sends the formatted prompt to the **Google Gemini API**. It handles retries with exponential backoff for API errors or malformed JSON responses.
    *   **Aggregation & Validation:** Results from all LLM calls are merged into a single JSON object. `utils/data.py` functions might deduplicate entries or calculate derived fields (like total experience). `utils/schema_validator.py` validates this final JSON against `schemas/base_output_schema.json`.
    *   The Document Processor returns the complete, structured resume JSON to the Backend API Gateway.

6.  **💾 Storing Parsed Resume & Orchestration: Calling NLP Classifier (Backend API Gateway)**
    *   The Backend API Gateway receives the structured JSON.
    *   It stores this full JSON document in the **MongoDB** `parsed_resumes` collection (via `core/mongo.py` using `motor`). MongoDB generates a unique `_id` (ObjectID).
    *   The Gateway then makes an asynchronous `POST` request (via `httpx` and `services/nlp_client.py`) to the **NLP Classifier** service's `/classify-role` endpoint, sending the same structured resume JSON.

7.  **🧠 Role Classification (NLP Classifier Service - `services/nlp_service/` - Port 8002)**
    *   The `/classify-role` endpoint in `src/api/main.py` receives the resume JSON.
    *   `src/inference.py` (`ResumeRolePredictor`):
        *   Extracts key text for embedding (`utils.py -> extract_text_from_canonical_resume`).
        *   Generates structured features (`feature_extraction_structured.py -> StructuredResumeFeatureExtractor`).
        *   Uses its loaded `ResumeRoleClassifierTransformer` (MiniLM + MLP heads from `model.py`) to predict probabilities for fine-grained and coarse-grained roles.
    *   Returns a JSON response containing the predicted roles and confidence scores.

8.  **📝 Finalizing Resume Record & Responding to Frontend (Backend API Gateway)**
    *   The Backend API Gateway receives the classification results.
    *   It creates/updates a record in the **PostgreSQL** `resumes` table (`models/models.py`). This record includes the `user_id`, `mongo_doc_id` (linking to the MongoDB document), and the `predicted_role` and `role_confidence` from the NLP service.
    *   A success response (e.g., 201 Created, with the new resume's SQL ID, Mongo ID, and predicted role) is sent back to the frontend.

9.  **🖥️ Frontend Displays Confirmation/Updates (Frontend Service)**
    *   The frontend (`upload.tsx`) receives the success response.
    *   The `useAppStore` is updated with the new resume's metadata.
    *   The UI shows a success message and may navigate to the dashboard (`pages/dashboard.tsx`) or the list of resumes (`pages/resumes.tsx`).

10. **✨ User Requests Job Matches (Frontend → Backend API → Matching Service → Backend API → Frontend)**
    *   User navigates to `pages/matches.tsx` and selects a processed resume from a dropdown.
    *   Frontend calls Backend API's `/api/v1/resumes/{resume_id}/matches` endpoint.
    *   Backend API (`endpoints/matching.py`):
        *   Fetches the selected resume's full JSON from MongoDB and its NLP role from PostgreSQL.
        *   Fetches a list of candidate jobs from PostgreSQL (job metadata, skills).
        *   Sends the resume data and list of job data to the **Matching Service**'s `/match/jobs` endpoint (Port 8003).
    *   **Matching Service (`services/matching_service/`):**
        *   `core/matcher.py` orchestrates the matching.
        *   `core/scoring.py` calculates individual scores for skills, experience, education, role, keywords.
        *   Scores are weighted and aggregated.
        *   Returns a ranked list of top job matches (job IDs and scores) to the Backend API.
    *   **Backend API Gateway:**
        *   (Optionally) Stores the match results in the PostgreSQL `matches` table.
        *   Forwards the ranked list of matches to the frontend.
    *   **Frontend (`pages/matches.tsx`):**
        *   Updates its state (via `useAppStore`) with the received matches.
        *   Renders the list of matched jobs, often using a `MatchCard.tsx`-like component and `MatchScoreIndicator.tsx` to display scores.

---

### 🧠 Deep Dive: Key Architectural Decisions & Their Implications

*   **Microservices vs. Monolith:**
    *   **Decision:** Microservices.
    *   **Rationale:** As outlined in the dissertation (Sec 4.1), this promotes modularity (independent development & deployment of parsing, NLP, matching), scalability (scale specific services like Document Processor under load), and technology flexibility (NLP service could use a different stack if needed, though currently Python).
    *   **Implications:**
        *   👍 Easier to manage complexity of individual components.
        *   👍 Teams can work on different services in parallel.
        *   👍 Fault isolation (e.g., if matching service is down, parsing can still work).
        *   👎 Increased operational complexity (managing multiple deployed services).
        *   👎 Network latency for inter-service calls (mitigated by `asyncio` and efficient local Docker networking).
        *   👎 Requires careful API contract design and versioning between services.

*   **API Gateway Pattern (`backend/` service):**
    *   **Decision:** Use a dedicated backend service as an API Gateway.
    *   **Rationale:** Provides a single entry point for the frontend, simplifying frontend logic. Centralizes concerns like authentication (JWT), rate limiting (planned), and request routing. Abstracts the internal microservice structure from the client.
    *   **Implications:**
        *   👍 Simplified frontend: only talks to one API.
        *   👍 Centralized security and policy enforcement.
        *   👎 The gateway itself can become a bottleneck if not designed to be highly performant and scalable (FastAPI's async nature helps here).
        *   👎 Adds an extra network hop for requests destined for other microservices.

*   **Hybrid Database (PostgreSQL + MongoDB - Dissertation Sec 4.1.2, Fig 4.12):**
    *   **Decision:** Use PostgreSQL for relational/metadata and MongoDB for flexible resume JSON.
    *   **Rationale:**
        *   **PostgreSQL:** Ideal for structured data like user accounts (`users` table), job postings (`jobs` table with defined fields), skills (`skills` table), and the relationships between them (e.g., `matches` table linking resumes and jobs). Offers ACID compliance, strong indexing, and powerful SQL querying. The dissertation (Figure 4.12) shows these relational links.
        *   **MongoDB:** Perfect for storing the full parsed resume JSONs (`parsed_resumes` collection). Resumes are highly variable in structure and content, making a fixed relational schema impractical. MongoDB's document model accommodates this flexibility, allowing for easy storage and retrieval of complex, nested JSON.
    *   **Implications:**
        *   👍 Best tool for each job: relational power where needed, flexibility where needed.
        *   👎 Data consistency: Requires application-level logic to maintain consistency between the `mongo_doc_id` in PostgreSQL and the actual document in MongoDB. No native distributed transactions.
        *   👎 Increased operational overhead: Managing two different database systems.

*   **Sectional LLM Parsing (Document Processor - Dissertation Sec 4.2.4, Fig 4.6):**
    *   **Decision:** Break resume into sections (layout analysis) and send each section to LLM with a specialized prompt, rather than sending the whole resume.
    *   **Rationale:**
        *   **Token Optimization:** Reduces total token count sent to/received from Gemini, lowering costs and latency (Dissertation Sec 5.1.4 states 40-60% token reduction).
        *   **Improved Accuracy/Focus:** LLM can focus on extracting specific information from a smaller, more relevant text chunk (e.g., extracting only experience details from the "Experience" section).
        *   **Manages Context Window:** Handles long resumes that might exceed whole-document LLM context limits.
        *   **Resilience:** Failure to parse one section doesn't necessarily mean failure for the whole document; other sections can still be processed.
    *   **Implications:**
        *   👍 More efficient and often more accurate than whole-document prompting.
        *   👎 Relies on the accuracy of the initial layout analysis step to correctly segment sections. If layout analysis fails or misidentifies sections, the LLM might get incorrect input.
        *   👎 Requires managing multiple prompts (one per section type).

*   **Stateless Authentication (JWT):**
    *   **Decision:** Use JSON Web Tokens for user authentication.
    *   **Rationale:** JWTs are stateless, meaning the server (Backend API Gateway) doesn't need to store session information for each user. The token itself contains all necessary information for authentication, signed by the server. This makes scaling easier as any instance of the Backend API can validate a token.
    *   **Implications:**
        *   👍 Scalability: No need for shared session stores.
        *   👎 Token Management: Tokens need to be securely stored on the client (e.g., `localStorage`) and can be larger than session cookies.
        *   👎 Revocation: Revoking a JWT before its expiry can be more complex than invalidating a server-side session (often requires a blacklist).

---

### 📄 Key Files for Eva (Module 1 - System Architecture)

*(This list remains consistent with the previous detailed response, focusing on high-level definition files and entry points for each service.)*

| File Path                                  | Role in Architecture                                                                                                                                  |
| :----------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `docker-compose.yml` (Root)                | **Master Orchestrator**: Defines all services (frontend, backend, document_processor, nlp_service, matching_service, postgres, mongo), their build instructions, ports, networks, volumes, and environment variables. Key to understanding how services are linked and run together. |
| `README.md` (Root)                         | **Project Entry Point**: Provides high-level overview, tech stack summary, and setup instructions. Confirms the microservice nature.                  |
| `docs/architecture/system_design.md`       | **Architectural Blueprint**: Should contain detailed diagrams (like dissertation Fig 4.1), explanations of design choices, data flow, and components. |
| `backend/app/main.py`                      | **API Gateway Core**: Initializes the FastAPI app for the main backend, includes routers from `endpoints/`, sets up middleware (CORS), and lifespan events for managing shared resources like HTTP clients and DB connections. |
| `backend/app/core/config.py`               | **Backend Configuration Hub**: Loads essential configurations like database URLs (PostgreSQL, MongoDB), URLs for other microservices (Document Processor, NLP, Matcher), and JWT secrets, typically from environment variables (`.env`). |
| `services/document_processor/src/api/main.py` | **Document Processor API Definition**: Defines the FastAPI app and endpoints (e.g., `/process-resume`) for the document parsing service.           |
| `services/nlp_service/src/api/main.py`     | **NLP Classifier API Definition**: Defines the FastAPI app and endpoint (e.g., `/classify-role`) for the role classification service.             |
| `services/matching_service/src/api/main.py`  | **Matching Service API Definition**: Defines the FastAPI app and endpoint (e.g., `/match/jobs`) for the job matching service.                   |
| `backend/app/models/models.py`             | **PostgreSQL Schema Definition**: Contains SQLAlchemy ORM models that define the structure of tables like `users`, `jobs`, `resumes` (metadata), `matches`, `skills`. |
| `schemas/base_output_schema.json` (in `document_processor`) | **Resume JSON Contract**: Defines the expected structure for the JSON output by the Document Processor.                               |

---

### "Eva must understand" (Module 1 - Re-verified & Elaborated)

*   ✅ **The 4 main operational backend microservices (+ Frontend + Databases):**
    *   **Backend API Gateway (`backend/`):** 🌟 **Central Coordinator**. FastAPI. Handles ALL frontend requests. Manages 🛡️ user authentication (JWTs). Routes tasks to other specialized microservices. Directly interacts with 🐘 PostgreSQL for user profiles, job metadata, match scores, and resume metadata. Interacts with 🍃 MongoDB for full parsed resume JSONs.
    *   **Document Processor (`services/document_processor/`):** 📄➡️📊 **Parser**. FastAPI. Receives resume/job files. Uses 🔎 PyMuPDF/python-docx for initial text/layout extraction. Then, for each identified section, it sends the text with a specialized 🤖 **XML prompt** to the **Google Gemini LLM** to get structured JSON output.
    *   **NLP Classifier (`services/nlp_service/`):** 🧠🏷️ **Role Tagger**. FastAPI. Takes the structured resume JSON from the Document Processor (via Backend API). Uses a pre-trained 🗣️ `all-MiniLM-L6-v2` model for text embeddings and combines these with ~39 extracted structured features. A custom 🤖 PyTorch model (`ResumeMultiHeadClassifier`) then predicts fine-grained (13 specific tech roles) and coarse-grained (e.g., "Engineering" vs. "Data") job roles.
    *   **Matching Service (`services/matching_service/`):** 🎯⚖️ **Scorer**. FastAPI. Compares a structured resume JSON against a list of structured job JSONs. Calculates a comprehensive match score based on multiple weighted dimensions: 🛠️ skills (Jaccard/embeddings), 📅 experience (years, seniority level), 🎓 education, 🏷️ role alignment, and 🔑 keywords (TF-IDF).
    *   *Dissertation Abstract Confirmation*: "The system was implemented as a microservice architecture with three components: a Document Processor handling extraction, a Backend API Gateway orchestrating requests, and a React-based frontend." (NLP and Matcher are often considered sub-components or downstream processors of the core parsing flow).

*   ✅ **Databases: MongoDB for full resumes; PostgreSQL for relational data.**
    *   🍃 **MongoDB (`mongo` service):** Stores the complete, rich, and potentially variably structured JSON output from the Document Processor for each resume in a collection like `parsed_resumes`. This is ideal because resume structures can differ significantly, and MongoDB's document model handles this flexibility well. The Backend API's `resumes.py` endpoint saves here.
    *   🐘 **PostgreSQL (`postgres` service):** Stores well-defined, relational data.
        *   `users`: User accounts, credentials.
        *   `jobs`: Job descriptions (metadata like title, company, location, and potentially structured data if jobs are also parsed), required skills.
        *   `resumes`: *Metadata* about each resume, including a `mongo_doc_id` that links to the full JSON in MongoDB, the `user_id`, `predicted_role` from NLP, etc.
        *   `matches`: Records of resume-job matches with their calculated scores.
        *   `skills`: A table of unique skills, linked to both resumes and jobs via many-to-many join tables (`resume_skills`, `job_skills`).
    *   *Dissertation Confirmation (Fig 4.12)*: Shows this clear separation and link.

*   ✅ **React frontend calls Backend API Gateway via REST API.**
    *   Correct. The `frontend/` service, built with Next.js (which uses React), makes HTTP requests (GET, POST, etc.) to the `/api/v1/...` endpoints exposed by the `backend/` service (Backend API Gateway). `frontend/src/services/api.ts` likely centralizes these Axios calls.

*   ✅ **Containerization with Docker and Orchestration with Docker Compose.**
    *   The entire system is designed to run as a set of interconnected Docker containers. Each service (`frontend`, `backend`, `document_processor`, `nlp_service`, `matching_service`, `postgres`, `mongo`) has its own `Dockerfile` to build its image.
    *   The root `docker-compose.yml` file defines how to build/pull these images, sets up a common Docker network for them to communicate (using service names like `http://document_processor:8001`), maps necessary ports to the host machine, mounts volumes for persistent data (like PostgreSQL data in `postgres_data` volume) and code (for development), and injects environment variables from `.env` files. This ensures a consistent and reproducible multi-service environment.

---

### ❓ Advanced Viva Questions Eva Must Be Ready For (Module 1 - Extended & Refined)

*(These questions are designed to test deep understanding, linking to dissertation content where applicable)*

1.  **"Your dissertation's Figure 4.1 ('High-Level System Architecture') shows several components. Can you trace the primary data flow when a new resume is uploaded, detailing the role of each microservice and database as per your implementation?"**
    *   **Answer:** "Certainly.
        1.  **Frontend (React/Next.js on Port 3000):** The user uploads a PDF/DOCX file via the UI (`pages/upload.tsx`). `services/api.ts` sends this file via a POST request to the Backend API Gateway.
        2.  **Backend API Gateway (FastAPI on Port 8000):** This service first authenticates the user. It then forwards the file to the Document Processor.
        3.  **Document Processor (FastAPI on Port 8001):** This service is the core parser. It uses PyMuPDF (for PDFs) or python-docx (for DOCX) to extract text and layout. It then performs layout analysis to identify sections (as described in dissertation Sec 4.2.2). For each section, it sends the text to the Google Gemini LLM using specific XML prompts (dissertation Sec 4.2.3, Fig 4.7) to get structured JSON. These sectional JSONs are merged and validated against `base_output_schema.json`.
        4.  **Backend API Gateway (again):** It receives the full structured JSON from the Document Processor.
            *   **MongoDB (Port 27017):** The Gateway stores this complete JSON document in the `parsed_resumes` collection in MongoDB (dissertation Sec 4.1.2).
            *   Then, it sends this structured JSON to the NLP Classifier.
        5.  **NLP Classifier (FastAPI on Port 8002):** This service takes the structured JSON, extracts text for embeddings (using `all-MiniLM-L6-v2`) and structured features. Its PyTorch model (`ResumeMultiHeadClassifier`) predicts the job role.
        6.  **Backend API Gateway (final steps for upload):** It receives the role classification.
            *   **PostgreSQL (Port 5432):** The Gateway stores metadata for the resume in the `resumes` table – this includes the `user_id`, the `mongo_doc_id` (linking to MongoDB), the `predicted_role`, and `role_confidence`. User and Job data are also in PostgreSQL.
            *   Finally, it returns a success response to the Frontend."

2.  **"The Abstract mentions a 'Sectional LLM approach' and 'heuristic layout analysis'. Why was this hybrid approach chosen for the Document Processor over, say, a purely visual model like LayoutLM or sending the whole document to Gemini?"**
    *   **Answer:** "We chose the Sectional LLM approach with heuristic layout analysis (dissertation Sections 3.6, 4.2.2, 4.2.4) as a balance between accuracy, cost, and implementation complexity.
        *   **Purely Visual Models (LayoutLM):** As mentioned in the dissertation (Sec 3.4.2, 4.9.1), models like LayoutLM are powerful for understanding document structure from images but require significant computational resources (GPUs for training and often for inference) and large, annotated training datasets, which were outside the project's scope and resource constraints.
        *   **Whole-Document LLM Processing:** Sending the entire resume text to Gemini in one go (dissertation Sec 4.8.2, 4.9.1) faced challenges with LLM token limits for input/output, especially for longer resumes. It also made it harder for the LLM to maintain context and accurately structure diverse information from across the entire document into a complex nested JSON.
        *   **Our Hybrid (Sectional) Approach:**
            1.  **Heuristic Layout Analysis (PyMuPDF data):** For PDFs, we first use font size, boldness, and keyword patterns to identify logical sections (Experience, Education, etc.). This is computationally cheaper than full visual models and gives us focused text chunks. This is detailed in Section 4.2.2.
            2.  **Sectional LLM Prompts:** Each text chunk is then sent to Gemini with a prompt *specifically designed* for that section type (e.g., an 'experience' prompt for the experience section). This makes the LLM's task easier, improves accuracy for that section, and significantly reduces token usage per call and overall (as noted in dissertation Sec 5.1.4, a 40-60% reduction).
        This approach leverages the LLM's semantic understanding for complex free-text within sections, while using efficient heuristics for initial structural segmentation."

3.  **"Docker Compose (`docker-compose.yml`) is used for orchestration. How does it manage service dependencies (e.g., ensuring databases are ready before applications that use them) and networking between services?"**
    *   **Answer:**
        *   **Service Dependencies (`depends_on`):** In `docker-compose.yml`, we can use the `depends_on` key to specify startup order. For instance, the `backend` service definition would include `depends_on: [postgres, mongo]`. This tells Docker Compose to *start* the `postgres` and `mongo` containers before it starts the `backend` container. However, `depends_on` only waits for the container to be started, not for the service *inside* the container (like the PostgreSQL server) to be fully initialized and ready to accept connections. For true readiness, applications often implement retry logic for their database connections, or more advanced `docker-compose` setups use healthchecks with `condition: service_healthy`.
        *   **Networking:** Docker Compose automatically creates a default bridge network for all services defined in the `docker-compose.yml` file. Services on the same network can discover and communicate with each other using their service names as hostnames. For example, if the Document Processor service is named `document_processor` in the YAML and exposes port 8001, the Backend API Gateway can make an HTTP request to `http://document_processor:8001`. The actual IP addresses are managed by Docker's internal DNS. Environment variables (like `DOCUMENT_PROCESSOR_URL=http://document_processor:8001`) are passed to services to tell them how to reach each other."

4.  **"What data is stored in PostgreSQL versus MongoDB, and what's the rationale for this split (dissertation Section 4.1.2, Figure 4.12)?"**
    *   **Answer:**
        *   🐘 **PostgreSQL** is our relational database and stores:
            *   `users`: User authentication details (email, hashed password), profile information.
            *   `jobs`: Core job information (title, company, location, description snippet, required skills - potentially linked via a join table). This data is structured and benefits from relational queries.
            *   `resumes` (metadata): Stores metadata for each processed resume, like its SQL ID, the `user_id` it belongs to, the `mongo_doc_id` (which is the foreign key to the full document in MongoDB), the `predicted_role` from the NLP service, and timestamps.
            *   `skills`: A table of unique skills, potentially linked to both jobs and resumes via many-to-many join tables (`job_skills`, `resume_skills`).
            *   `matches`: Records of successful matches, storing `resume_id`, `job_id`, the calculated `score`, and a timestamp.
            *   *Rationale for PostgreSQL*: Used for data that has clear relationships, requires strong consistency (ACID compliance), benefits from structured querying (SQL for filtering users, jobs, matches), and needs referential integrity (e.g., ensuring a `resume.user_id` points to an existing user).
        *   🍃 **MongoDB** is our NoSQL document database and stores:
            *   `parsed_resumes` (collection): The full, detailed, and often deeply nested JSON object that results from parsing a resume by the Document Processor. This JSON contains all extracted sections like personal info, experience, education, skills, projects, etc.
            *   `parsed_jobs` (collection, optional): Similarly, if job descriptions are parsed into complex JSON by the Document Processor, they could also be stored here.
            *   *Rationale for MongoDB*: Resumes are highly variable in structure. A flexible document model like MongoDB's is ideal for storing this diverse JSON data without the constraints of a rigid relational schema. It excels at storing and retrieving entire complex documents quickly.
        *   The dissertation's Figure 4.12 illustrates this, showing `mongo_id` in PostgreSQL tables linking to the `_id` in MongoDB collections."

5.  **"How are environment variables managed across the different services, and what are some key examples of variables used for inter-service communication or external API keys?"**
    *   **Answer:**
        1.  **`.env.template` & `.env` Files:** Each service (and the root project) has a `.env.template` file listing required environment variables with placeholders. Developers copy this to `.env` and fill in actual values. The `.env` files are gitignored.
        2.  **`docker-compose.yml`:** This file is crucial for injecting these variables into the respective Docker containers at runtime. For each service, under the `environment:` key, we can either directly set variables or, more commonly, reference variables from a `.env` file (e.g., `env_file: ./.env` for root, or specific `.env` files for services if they have their own). Example:
            ```yaml
            services:
              backend:
                env_file: ./backend/.env
                environment:
                  - DATABASE_URL=${POSTGRES_URI_BACKEND} # From .env
                  - DOCUMENT_PROCESSOR_URL=http://document_processor:8001
              document_processor:
                env_file: ./services/document_processor/.env
                environment:
                  - GOOGLE_API_KEY=${GEMINI_API_KEY} # From its .env
            ```
        3.  **Application Code (`config.py`):** Within each Python service (e.g., `backend/app/core/config.py`, `services/document_processor/config/settings.py`), Pydantic's `BaseSettings` or `os.getenv()` is used to load these environment variables into application configuration objects.
        *   **Key Examples:**
            *   `DATABASE_URL` (or `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_HOST`): For PostgreSQL connection in the `backend` service.
            *   `MONGO_URI` (or `MONGO_HOST`, `MONGO_PORT`, `MONGO_DB_NAME`): For MongoDB connection in the `backend` service.
            *   `GEMINI_API_KEY`: For the `document_processor` service to authenticate with Google Gemini.
            *   `DOCUMENT_PROCESSOR_URL`, `NLP_SERVICE_URL`, `MATCHING_SERVICE_URL`: Used by the `backend` service to know the network addresses of the other microservices.
            *   `JWT_SECRET_KEY`, `ALGORITHM`, `ACCESS_TOKEN_EXPIRE_MINUTES`: For JWT authentication in the `backend` service.
            *   `LOG_LEVEL`: To control logging verbosity in services.

*(The remaining questions from the previous response's Module 1 Advanced section (6-20) are still highly relevant and cover aspects like API versioning, data in `matches` table, API key security, token usage rationale, schema evolution, `data_ingestion_service` role, FastAPI concurrency, resource utilization differences, Dockerfile strategy, http_client resilience, Pydantic's role, and Alembic usage. Eva should prepare for those as well.)*

---

### ✅ Final Review Checklist for Eva (Module 1 - Consistent & Emphasized)

*   [ ] **Master the System Diagram:** Be able to draw it from memory, labeling all services (Frontend, Backend API, Document Processor, NLP, Matcher), databases (Postgres, Mongo), external APIs (Gemini), ports, and primary data flow arrows for both resume upload and matching.
*   [ ] **`docker-compose.yml` Deep Dive:** Explain how services are defined, linked (`depends_on`, network), configured (ports, volumes, `env_file`, `environment`), and built (`build` context).
*   [ ] **Inter-Service Calls:** Clearly articulate how, for example, the `backend` service calls `document_processor` using its service name and port defined in Docker Compose and passed via an environment variable.
*   [ ] **Data Storage Strategy:** Confidently explain *why* MongoDB for full resume JSON (flexibility for varied structures) and *why* PostgreSQL for metadata, users, jobs, and matches (relational integrity, querying). Know which service writes to/reads from which DB for key entities. (Refer to dissertation Figure 4.12).
*   [ ] **Configuration Flow:** Trace an environment variable (e.g., `GEMINI_API_KEY`) from its definition in an `.env` file, through `docker-compose.yml`, to its usage in a service's `config.py` or `settings.py`.
*   [ ] **Role of Each Service:** Give a concise "elevator pitch" for the primary responsibility of the Frontend, Backend API Gateway, Document Processor, NLP Classifier, and Matching Service.
*   [ ] **Asynchronous Nature:** Explain that FastAPI services are asynchronous and how this benefits I/O-bound operations like calling external APIs (Gemini) or other internal microservices (`httpx.AsyncClient`).
*   [ ] **Key Files Walkthrough Prep:** Be ready to open and briefly explain the purpose of `docker-compose.yml`, the main `README.md`, `backend/app/main.py`, and the `src/api/main.py` of one of the processing microservices (e.g., Document Processor).


---

## 💎 Module 2: Resume & Job Description Parsing (Gemini LLM & Regex) - Definitive Deep Dive & Orchestration

### 🎯 **Core Objectives for Eva:**

*   🧠 **Master Dual Parsing Pipelines:** Attain an exceptionally detailed, step-by-step understanding of how the `services/document_processor/` ingests **both resume files AND job description files** (PDF/DOCX), and transforms them into structured JSON. This includes:
    *   The **primary resume parsing pipeline** leveraging layout analysis and the Gemini LLM for deep semantic extraction.
    *   The **alternative job description parsing pipeline**, which might utilize a simpler regex/heuristic approach (`src/core/job_parser_regex.py`) for efficiency if LLM is not strictly needed or as a distinct pathway.
*   🗣️ **Articulate with Unmatched Precision:** Fluently explain every stage: initial file handling, robust text/layout extraction (PyMuPDF for PDFs, `python-docx` for DOCX), sophisticated section identification heuristics for resumes, intricate XML prompt engineering for Gemini (for resumes), asynchronous LLM API interactions, comprehensive JSON response validation, intelligent data merging strategies, and critical post-processing steps. Be ready to discuss the nuances of why the job parsing might differ.
*   🗺️ **Navigate the Document Processor Codebase Like a Pro:** Confidently pinpoint where specific functionalities reside within `services/document_processor/src/` and its subdirectories (e.g., `core/parser.py` as the main resume orchestrator, `core/job_parser_regex.py` for jobs, `layout/layout_analyzer.py`, `llm/client.py`, `text_processing/base_processor.py`, individual section processors, and `prompts/`).
*   💡 **Justify All Design Choices and Trade-offs:** Defend decisions such as the sectional LLM approach for resumes vs. potentially simpler regex for jobs, the selection of XML for prompts, the use of PyMuPDF for its rich layout data, and the comprehensive error handling/retry mechanisms. Refer to and elaborate on dissertation sections (e.g., Ch 4.2, Figs 4.3-4.8, Code Snippets 4.1-4.4).
*   ⚙️ **Understand Configuration, Schemas, and Outputs:** Explain how prompts are dynamically loaded and formatted, how `base_output_schema.json` (for resumes) and any job-specific schemas guide the process, and how detailed parsing metrics and validation results are captured in the final JSON output.
*   🛡️ **Address Edge Cases & Limitations:** Discuss how the system handles unconventional resume layouts, corrupted files, LLM API failures, and the current scope regarding non-English content or image-only PDFs.

---

### 🚀 Technologies Involved (Crucial for `services/document_processor/`)

| Category                           | Technology                                                                  | 🌟 Purpose & Significance in Document Processor Service                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Relevant Files (Examples)                                                                                                                                                                                                                                                                                                                                                     |
| :--------------------------------- | :-------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **📦 Core Service & API**          | Python 3.9+, FastAPI, Uvicorn                                               | Foundation of the microservice. FastAPI exposes the API endpoints (`/process-resume` for LLM-based resume parsing, `/process-job` potentially for regex/heuristic job parsing). Uvicorn serves the FastAPI application asynchronously.                                                                                                                                                                                                                                                                                                                                                                                      | `src/api/main.py`, `Dockerfile` for the service.                                                                                                                                                                                                                                                                                                                                  |
| **📄 PDF Extraction & Analysis**   | PyMuPDF (`fitz`)                                                              | **👑 King of PDF Data**: This is paramount for resume parsing. It doesn't just get text; it extracts text *blocks* with their precise bounding box coordinates (`bbox`), page numbers, and often font attributes (name, size, bold flags if encoded in PDF). This rich spatial and stylistic data is the bedrock for `LayoutProcessor` to heuristically identify sections in visually complex resumes. (Dissertation Sec 4.2.1)                                                                                                                                                                                                                            | `src/extractors/pdf.py` (function `extract_pdf_data_pymupdf`).                                                                                                                                                                                                                                                                                                                    |
| **📝 DOCX Extraction**             | `python-docx`                                                                 | **Word Document Specialist**: Extracts textual content from `.docx` files, preserving paragraph structure and accessing text within tables. It's simpler than PDF extraction as DOCX is an inherently more structured format, but it doesn't provide the same granular positional/font metadata, thus typically bypassing the advanced layout analysis used for PDFs. (Dissertation Sec 4.2.1)                                                                                                                                                                                                                                       | `src/extractors/docx.py` (function `extract_text_from_docx`).                                                                                                                                                                                                                                                                                                                 |
| **🔍 Layout Analysis (Resumes)**   | Custom Heuristics & Python `re` (Regular Expressions)                       | **Resume Structure Detective (PDFs)**: The `LayoutProcessor` class in `src/layout/layout_analyzer.py` is a custom-built engine. It intelligently analyzes the `blocks` (text + coordinates + font info) from PyMuPDF. It uses a combination of: <br> 1. Font size/style changes (headers are often larger/bolder). <br> 2. Vertical/horizontal spacing between blocks. <br> 3. Keyword spotting using regex patterns from `src/layout/patterns.py` (e.g., "EXPERIENCE", "EDUCATION", "SKILLS"). <br> This allows it to segment a PDF resume into logical sections even without explicit XML-like tags. (Dissertation Sec 4.2.2, Fig 4.5, Code Snippet 4.1) | `src/layout/layout_analyzer.py`, `src/layout/patterns.py`, `config/mappings.py` (for `FALLBACK_SECTIONS_MAP` which standardizes detected header names).                                                                                                                                                                                                                                        |
| **🤖 LLM API Integration (Resumes)**| Google Gemini API (specifically `gemini-1.5-flash-latest` via SDK)          | **Semantic Understanding & JSON Generation**: For resumes, this is the core intelligence for converting unstructured/semi-structured text *within each identified section* into a well-defined JSON structure. The `google-generativeai` Python SDK (which uses `httpx` for async calls) handles the communication. (Dissertation Sec 4.2.3)                                                                                                                                                                                                                               | Interacted with via `src/llm/client.py` (function `generate_structured_json_async`). Configuration in `config/settings.py` (model name, API key from env).                                                                                                                                                                                                                         |
| **🗣️ LLM Prompt Engineering (Resumes)** | XML Templates & Python String Formatting                                  | **Instructing the LLM Precisely**: Prompts are meticulously crafted as XML files (`prompts/*.xml`). XML is chosen for its ability to clearly delineate: <br> 1. System role/persona. <br> 2. User input (the resume section text). <br> 3. Explicit instructions on what to extract. <br> 4. The desired JSON output format, often including a schema snippet or example. <br> Placeholders (e.g., `{{SECTION_TEXT}}`, `{{schema_example}}`) are filled dynamically. (Dissertation Fig 4.7, Code Snippet 4.3)                                                            | XML files in the `prompts/` directory (e.g., `experience.xml`, `education.xml`, `skills.xml`). Loading logic in `src/llm/prompt.py` (`load_prompt`). Formatting in `src/text_processing/base_processor.py`.                                                                                                                                                                  |
| **⚡ Asynchronous Operations**    | Python `asyncio`, `httpx`                                                     | **Concurrent Processing Power**: Essential for performance. `async def` and `await` are used extensively. <br> 1. FastAPI endpoints are async. <br> 2. Calls to the Gemini LLM via `httpx` (through the SDK) are async. <br> 3. `asyncio.gather` in `src/core/parser.py` allows multiple LLM calls for different resume sections to be made concurrently, significantly speeding up the processing of a single resume. (Dissertation Fig 4.3 "Asynchronous Document Processing Flow", Code Snippet 4.2).                                                                 | `src/api/main.py`, `src/core/parser.py`, `src/llm/client.py`, `src/text_processing/base_processor.py`.                                                                                                                                                                                                                                                                       |
| **🔄 Retry Mechanism for LLM**   | Custom Loop with `asyncio.sleep`                                              | **Ensuring LLM Call Resilience**: The `generate_structured_json_async` function in `src/llm/client.py` implements a retry loop (default 3 attempts) with exponential backoff (`asyncio.sleep(delay)` where delay doubles). This handles transient network issues, LLM API rate limits, or occasional malformed JSON responses, improving overall robustness. (Dissertation Fig 4.8)                                                                                                                                                                       | `src/llm/client.py`.                                                                                                                                                                                                                                                                                                                                                  |
| **🧩 Section-Specific Processors (Resumes)** | `LLMProcessor` base class & derived logic                               | **Modular & Focused Parsing**: The `src/text_processing/base_processor.py` defines `LLMProcessor`. Each semantic resume section (Experience, Education, Skills, etc.) has a corresponding module in `src/text_processing/` (e.g., `experience.py`). These modules typically instantiate an `LLMProcessor` configured with a specific prompt name, a section-specific validation function, and a default result. This modularity makes it easy to manage and refine parsing for each section. (Dissertation Sec 4.2.4, Code Snippet 4.6)                   | `src/text_processing/base_processor.py`, and all files like `src/text_processing/experience.py`, `education.py`, etc.                                                                                                                                                                                                                                                      |
| **📜 Schema Definition (Resumes)**| `base_output_schema.json` (JSON Schema Draft 7)                             | **The Blueprint for Resume Output**: This file rigorously defines the target structure, fields, data types, and constraints for the final JSON output of a parsed *resume*. It's used to: <br> 1. Provide an example structure within LLM prompts to guide Gemini. <br> 2. Perform final validation on the aggregated JSON output. (Dissertation Fig 4.9)                                                                                                                                                                                           | `schemas/base_output_schema.json`. Loaded by `src/utils/data.py`.                                                                                                                                                                                                                                                                                                     |
| **✅ Output Validation**         | `jsonschema` library (Resumes), Pydantic (API layer)                      | **Ensuring Quality & Consistency**: <br> 1. **For Resumes**: `src/utils/schema_validator.py` (`validate_result`) uses `jsonschema` to validate the final aggregated resume JSON against `base_output_schema.json`. (Dissertation Sec 4.7.4). <br> 2. **API Layer**: FastAPI uses Pydantic models implicitly to validate incoming request bodies and format outgoing responses for both `/process-resume` and `/process-job`.                                                                                                                               | `src/utils/schema_validator.py`, `src/api/main.py` (implicit Pydantic use).                                                                                                                                                                                                                                                                                          |
| **🛠️ Data Utilities**            | Custom Python functions (`data.py`, `date_utils.py`, `validation.py`)       | Helper functions for: <br> - Merging nested dictionaries (`set_nested_value` in `parser.py`). <br> - Deduplicating list entries (e.g., skills, experiences) using similarity checks. <br> - Calculating total work experience duration. <br> - Parsing and normalizing various date formats. <br> - Validating individual fields from LLM outputs (string, array, object). (Dissertation Code Snippet 4.4 mentions `assign_result`)                                                                                                    | `src/utils/data.py`, `src/utils/date_utils.py`, `src/utils/validation.py`.                                                                                                                                                                                                                                                                                                   |
| **⚙️ Configuration Management**  | `.env` files, `config/settings.py`, `config/mappings.py`                | **Controlling Behavior**: <br> - `config/settings.py`: Loads LLM parameters (model name, temperature, retries) and paths (prompts, schemas, logs) from environment variables (via `.env`). <br> - `config/mappings.py`: Defines crucial dictionaries: `SECTION_PROCESSING_MAP` (layout section name → processor function), `FALLBACK_SECTIONS_MAP` (alias → canonical section name), `PROCESSOR_TO_SCHEMA_PATH_MAP` (processor → schema path for examples/merging). | `config/settings.py`, `config/mappings.py`, service-level `.env` file.                                                                                                                                                                                                                                                                                                               |
| **📝 Logging**                   | Python `logging` module, `loguru` (mentioned in extended plan)              | **Tracking & Debugging**: Provides detailed logs for every step of the parsing process, including LLM interactions, errors encountered, section identification, validation results, and performance timings. `src/utils/logging.py` sets up file and console handlers.                                                                                                   | Used extensively throughout all modules in `src/`. Log files typically in `logs/`.                                                                                                                                                                                                                                                                                                     |
| **👔 Job Description Parsing** | Python `re` (Regex), Heuristics (Potentially no LLM for simple jobs)      | **Separate Path for Jobs**: The `/process-job` endpoint in `src/api/main.py` uses `src/core/job_parser_regex.py`. This module likely contains regex patterns and heuristics tailored to extract common fields from job descriptions (title, company, location, skills, salary, experience years) *without necessarily using an LLM*. This can be faster and cheaper for well-structured job posts. (Dissertation Sec 4.3 implies a simpler approach initially). | `src/api/main.py` (routes to `parse_job_regex`), `src/core/job_parser_regex.py` (contains regex and heuristic logic for job fields), `src/utils/regex_patterns.py` might hold some JDs patterns. |

---

### 🚶 Step-by-Step In-Depth Parsing Pipeline (within `services/document_processor/`)

This details the journey of a single resume file *after* it's received by the Document Processor's API endpoint `/process-resume`. The `/process-job` endpoint follows a simpler, likely regex-based path using `src/core/job_parser_regex.py`.

1.  **➡️ API Endpoint Reception & Temp File (`src/api/main.py` -> `/process-resume`)**
    *   A `POST` request with an `UploadFile` (resume) and a boolean `use_layout` flag hits the endpoint.
    *   File type is checked (PDF/DOCX). If unsupported, 🏳️ 415 HTTP error.
    *   A unique `request_id` (UUID) is generated for this parsing task.
    *   The uploaded file is saved to a temporary directory (e.g., `temp_uploads/`) using `shutil.copyfileobj`. The temp filename includes the `request_id` (e.g., `uuid_request_id.pdf`). 💾
    *   This temporary file path is passed to the core parsing logic.

2.  **⚙️ Core Resume Parsing Orchestration (`src/core/parser.py` -> `parse_document` which calls `parse_document_logic_async`)**
    *   **🌟 Initialization (`initialize_result` helper called by `parse_document_logic_async`):**
        *   A `result` dictionary is created by deep copying the structure from `schemas/base_output_schema.json` (loaded via `utils/data.py -> load_schema()`). This ensures all expected top-level keys and nested structures are present from the start.
        *   Essential `metadata` is populated: `filename`, `parsing_date` (current timestamp), `layout_analysis_used` (based on input flag), `llm_model_used` (from `config/settings.py`), and initializes `parsing_metrics` (for timings, LLM calls, errors) and `validation` sub-dictionaries.
    *   **📄 Text & Layout Extraction (`src/core/extractor.py` -> `DocumentExtractor`):**
        *   `start_time = time.time()`.
        *   `extractor.extract(temp_file_path)` is called.
        *   **If PDF:** `extractors/pdf.py -> extract_pdf_data_pymupdf()` uses `fitz.open()`.
            *   Extracts `full_text` (concatenation of all page text).
            *   Extracts `blocks`: A list of `dict`s, each with `text`, `bbox` (coordinates `x0, y0, x1, y1`), `page`. Font info (size, bold flags) is also extracted if `page.get_text("dict")` is used, which `LayoutProcessor` relies on.
            *   Extracts PDF `metadata` (title, author, etc. from PDF properties).
        *   **If DOCX:** `extractors/docx.py -> extract_text_from_docx()` uses `python-docx`.
            *   Extracts `full_text`. `blocks` list will be empty/None as DOCX doesn't provide this granular layout info through `python-docx`.
        *   The extracted `full_text`, `blocks` (if PDF), and `doc_metadata` are stored within `result['extracted_content']` and `result['metadata']['file_info']`.
        *   `parsing_metrics['extraction_time_ms']` is recorded.
        *   If extraction fails (e.g., corrupted file), `extracted_content` is `None`, and the parser returns the initialized `result` (indicating failure).
    *   **📊 Layout Analysis (for PDF & if `use_layout` is True - `src/layout/layout_analyzer.py` -> `LayoutProcessor`):**
        *   `start_time = time.time()`.
        *   An instance of `LayoutProcessor` is created.
        *   `LayoutProcessor.process_layout(pdf_blocks)` is called. (Described in dissertation Sec 4.2.2, Fig 4.5, Code Snippet 4.1).
            *   `_find_section_headers()`: Iterates `pdf_blocks`. Identifies header candidates based on:
                *   **Font Size Heuristic:** Larger than `most_common_font_size * 1.2`.
                *   **Boldness:** If `is_bold` flag (from PyMuPDF block/span data) is true.
                *   **All Caps:** If text `isupper()` and `len > 4`.
                *   **Keyword Match:** If line matches regex patterns from `layout/patterns.py` (`COMMON_SECTION_PATTERNS`).
            *   `_get_canonical_section()`: Maps matched header text (e.g., "WORK HISTORY") to a canonical key (e.g., "experience") using `FALLBACK_SECTIONS_MAP` from `config/mappings.py`.
            *   Segments text blocks under these canonical section names, producing `layout_sections = {"experience": "text_for_experience_section...", "education": "..."}`.
        *   `layout_sections` is stored (e.g., in `result['extracted_content']['section_text']`).
        *   `parsing_metrics['layout_analysis_time_ms']` is recorded.
        *   `result['metadata']['parsing_metrics']['layout_analysis_used']` is set.
    *   **🤖 Preparing Asynchronous LLM Tasks for Resume Sections (within `parse_document_logic_async`):**
        *   A list, `tasks_to_run`, is created to hold tuples of `(target_path_in_final_json, asyncio_coroutine_for_llm_call)`.
        *   **Primary Pass (Layout-Based):** If `layout_sections` were successfully generated:
            *   It iterates through `layout_sections.items()`. For each `(section_key_layout, section_text)`:
                *   The `section_key_layout` is normalized (lowercase, strip) and then mapped to a canonical `mapped_key` using `FALLBACK_SECTIONS_MAP` if needed.
                *   The system looks up the `processing_func_name` (e.g., `extract_experience`) for this `mapped_key` from `SECTION_PROCESSING_MAP` (`config/mappings.py`).
                *   It also gets the `schema_path_str` (e.g., `professional_experience.positions`) from `PROCESSOR_TO_SCHEMA_PATH_MAP`. This path dictates where in the final `result` JSON the output of this LLM call will be merged.
                *   The actual asynchronous processor function (e.g., `tp.extract_experience`) is retrieved from `TEXT_PROCESSING_FUNCTIONS` (populated from `src/text_processing/__init__.py`).
                *   A coroutine is created: `coro = proc_func(section_text)`.
                *   The tuple `(tuple(schema_path_str.split('.')), coro)` is added to `tasks_to_run`.
                *   A set `processed_target_paths_layout` keeps track of schema paths handled by this layout-driven pass.
        *   **Fallback Pass:** The system then iterates through all standard sections defined in `SECTION_PROCESSING_MAP` (e.g., "experience", "education", "skills").
            *   For each standard section, it checks if its `target_path_tuple` was already added to `tasks_to_run` during the layout-based pass.
            *   If not (meaning layout analysis didn't find this section, or the section was empty), a new task is created. This task will run the section's_specific processor function (e.g., `tp.extract_skills`) on the *entire `full_text`* of the resume. This gives the LLM a chance to find skills even if they weren't in a clearly labeled "Skills" section.
        *   *Note:* The `extract_personal_info` task is often deferred to a separate, later pass because it might benefit from context from other already-parsed sections (like inferring location from the latest job).
    *   **⚡ Executing LLM Tasks Concurrently (`asyncio.gather`):**
        *   `llm_task_start_time = time.time()`.
        *   If `tasks_to_run` is not empty, `coroutines_only = [coro for path, coro in tasks_to_run]` are extracted.
        *   `results_from_gather = await asyncio.gather(*coroutines_only, return_exceptions=True)` is called. This is a **critical performance optimization**: all LLM calls for different sections are made in parallel, and the code waits here until all ofthem have completed or timed out/errored.
        *   `parsing_metrics['llm_parsing_time_ms']` and `parsing_metrics['llm_calls']` are recorded.
    *   **🧩 Aggregating Concurrent LLM Results (within `_aggregate_results` or similar logic):**
        *   The code iterates through `results_from_gather` (the outputs from each concurrent LLM task).
        *   For each `llm_output` and its corresponding `result_path_tuple`:
            *   If `llm_output` is an exception or an error dictionary from `llm/client.py`, it's logged in `parsing_metrics['llm_errors']` and `parsing_metrics['llm_error_count']` is incremented.
            *   If successful, `set_nested_value(result, result_path_tuple, llm_output)` is called. This utility function (defined in `parser.py` or `utils/data.py`, see Dissertation Code Snippet 4.4 for a similar `assign_result`) carefully merges the `llm_output` (which is a JSON snippet for one section, already validated by its `LLMProcessor`) into the main `result` dictionary at the specified nested path. It handles creating parent dictionaries if they don't exist and extending lists (e.g., adding multiple experience entries to `result['professional_experience']['positions']`).
            *   The time taken for each section's LLM call (if `LLMProcessor` returns it) is stored in `parsing_metrics['llm_section_times_ms']`.
    *   **👤 Final Personal Info Pass (Asynchronous, after other sections):**
        *   The `text_processing.personal_info.extract_personal_info` coroutine is `await`ed.
        *   It's given the text from the "header" section (if found by layout) or the first ~1000 chars of `full_text`.
        *   Crucially, it can also be passed the list of `experience_entries` that were just aggregated from other LLM calls. This allows it to infer the candidate's current location from their most recent job if not explicitly stated in the contact info.
        *   Its output (a dictionary for `personal_information`) is merged into `result['personal_information']` using `set_nested_value`.
    *   **🛠️ Post-Processing Steps (`src/utils/data.py`):**
        *   `calculate_total_experience(result['professional_experience']['positions'])` is called. This utility iterates through the extracted experience entries, parses their start/end dates (using `utils/date_utils.py`), handles "Present", and calculates the total years and months of work experience, accounting for overlaps. The result is stored in `result['professional_experience']['total_experience']`.
        *   `deduplicate_sections(result)` is called. This function iterates through list-based sections (like skills, experience, education) and removes entries that are highly similar to each other (using `are_items_similar` which might use string similarity like `SequenceMatcher`), preventing redundant information.
        *   Timings for these are recorded.
    *   **✅ Final Schema Validation (`src/utils/schema_validator.py`):**
        *   `validate_result(result, BASE_SCHEMA_NAME)` is called. This uses the `jsonschema` library (e.g., `Draft7Validator`) to rigorously check if the entire aggregated `result` dictionary conforms to the structure and data types defined in `schemas/base_output_schema.json`.
        *   The validation outcome (`is_valid`, `error_count`, `errors_list`) is stored in `result['metadata']['validation']`. If not valid, errors are logged. (Dissertation Sec 4.7.4)
    *   **🧹 Cleanup & Return:**
        *   Temporary intermediate data like `result['extracted_content']` (which held raw blocks) is deleted.
        *   `parsing_metrics['total_processing_time_ms']` is calculated.
        *   The final, rich, structured, and validated `result` dictionary is returned.

3.  **➡️ API Endpoint Response (Return to Backend API Gateway - `src/api/main.py`)**
    *   The `/process-resume` endpoint receives the `result` dictionary from `parse_document`.
    *   If `result` is `None` or indicates a critical failure, an `HTTPException` (e.g., 500 Internal Server Error) is raised, which gets returned to the Backend API Gateway.
    *   Otherwise, the `result` JSON is returned in the HTTP response (status 200 OK).
    *   The temporary uploaded file (e.g., `uuid_request_id.pdf`) is deleted from `temp_uploads/`.

---

### 🧠 Deep Dive: Key Internal Logic & Justifications

*   **Layout Analysis (`LayoutProcessor`): Why Heuristics?**
    *   **Rationale:** As discussed in dissertation (Sec 4.2.2, 4.9.1), full visual models like LayoutLM are computationally expensive and require significant labeled data for training. A heuristic approach using PDF text block coordinates and font attributes (font size, boldness) from PyMuPDF, combined with regex for common headers, provides a good balance of speed and reasonable accuracy for segmenting *most* standard resume formats. It's a pragmatic choice for this project's scope.
    *   **How it Works (Recap):** Identifies lines with larger fonts, bold text, or all-caps text that also match known section keywords (e.g., "EXPERIENCE", "EDUCATION") from `layout/patterns.py`.
    *   **Limitations:** Struggles with highly unconventional layouts, resumes using graphics as separators, or where headers are not visually distinct. (Dissertation Sec 5.6.1 "Layout Complexity").

*   **Sectional LLM Processing vs. Whole-Document LLM:**
    *   **Rationale (Dissertation Sec 4.8.2, 4.9, 5.1.4):**
        *   **Token Limits:** Sending an entire resume (especially multi-page ones) to an LLM can easily exceed input/output token limits of models like Gemini Flash, leading to truncated or failed responses.
        *   **Cost Efficiency:** LLM APIs charge per token. Processing smaller, targeted sections is significantly cheaper (40-60% token reduction reported).
        *   **Accuracy & Focus:** Prompts can be highly specialized for each section type (e.g., specific instructions for extracting "responsibilities" in an experience prompt vs. "degree level" in an education prompt). This focus generally leads to more accurate and relevant extraction for that specific piece of information, as the LLM isn't trying to parse everything at once.
        *   **Error Isolation:** If the LLM fails to parse one section (e.g., a poorly formatted "Projects" section), other sections can still be processed successfully. With a whole-document approach, one problematic part could derail the entire extraction.
    *   **Orchestration:** `core/parser.py` manages this by first trying layout-defined sections, then falling back to processing predefined sections against the full text if a section was missed.

*   *   **XML Prompts (`prompts/*.xml`): Why XML?**
    *   **Rationale (Dissertation Sec 4.8.2, Fig 4.7):**
        1.  🌟 **Structure for the LLM:** XML's tagged nature allows for a very clear and explicit structure for the prompt itself. We can define distinct parts like `<system_instructions>`, `<user_input_section text="{{SECTION_TEXT}}">`, `<desired_output_format_example json_schema="{{schema_example}}">`, and `<specific_extraction_rules>`. This structured input helps the LLM (Gemini) better understand the context and the precise nature of the task for each section.
        2.  📚 **Embedding Examples:** It's easier to embed well-formatted examples of the desired JSON output for a particular section within XML tags, which is crucial for guiding the LLM to produce output that conforms to our target schema (`base_output_schema.json`).
        3.  🔧 **Maintainability & Versioning:** Storing prompts as separate XML files makes them easier to manage, version control (using Git), and iterate upon compared to having large, complex multi-line string prompts embedded directly in Python code. Different team members could potentially work on different prompt files.
        4.  🎯 **Reduced Ambiguity:** For complex extraction tasks like parsing job experiences (with multiple fields like company, title, dates, responsibilities, achievements), XML provides a less ambiguous way to specify all the requirements to the LLM than a purely natural language paragraph of instructions.
        5.  💡 **Clarity of Intent:** The tags make the prompt's intent clearer both for human developers and potentially for the LLM's internal parsing of the request.
    *   **Loading & Formatting:** `src/llm/prompt.py` (`load_prompt`) reads the raw XML string. Then, `src/text_processing/base_processor.py` (`LLMProcessor`) dynamically injects the actual resume `section_text` and a `schema_example` (a JSON string snippet of the expected output for that section) into placeholders like `{{SECTION_TEXT}}` and `{{schema_example}}` within the XML string.

*   **Asynchronous LLM Calls (`asyncio.gather` in `core/parser.py`):**
    *   **Rationale (Dissertation Fig 4.3):** Parsing a resume involves multiple calls to the Gemini LLM (one for each section like experience, education, skills, etc.). If these calls were made sequentially (synchronously), the total parsing time would be the sum of all individual LLM call latencies, which could be very long (e.g., 5 sections * 3 seconds/call = 15 seconds).
    *   By using `async def` for the LLM client functions (`generate_structured_json_async`) and the section processing methods, and then using `await asyncio.gather(*coroutines_for_each_section)`, the `parser.py` can launch all these LLM requests *concurrently*.
    *   **Benefit:** The Python event loop can switch between tasks while waiting for the LLM API to respond to each request. This means the total time spent waiting for LLM responses is closer to the time of the *longest single LLM call* rather than the sum of all calls, leading to a significant speed-up in overall resume processing time (as noted in dissertation Table 5.2, avg total processing 5.2s, avg LLM API response 3.5s - implies concurrency).

*   **Error Handling & Retries in LLM Client (`llm/client.py`):**
    *   **Rationale:** External API calls (like to Gemini) can fail due to transient network issues, temporary service overload (rate limits), or even occasional inconsistencies in the LLM's output format. A robust system needs to handle these gracefully.
    *   **Implementation:**
        1.  **Retry Loop:** `generate_structured_json_async` uses a `while attempt <= MAX_RETRIES:` loop (default `MAX_RETRIES = 2`, so 3 attempts).
        2.  **Exception Catching:** Inside the loop, the call to `model.generate_content_async(...)` and subsequent JSON parsing (`json.loads()`) are within a `try...except Exception as e:`.
        3.  **Specific Error Checks:**
            *   `response.candidates[0].finish_reason != 1 (STOP)`: Checks if Gemini stopped for reasons other than successful completion (e.g., `SAFETY`, `MAX_TOKENS`).
            *   `ValueError` when accessing `response.text`: Can happen if content is blocked due to safety filters.
            *   `json.JSONDecodeError`: If Gemini returns a string that isn't valid JSON even after stripping markdown.
        4.  **Exponential Backoff:** If an error occurs and retries are remaining, `await asyncio.sleep(delay)` is called. The `delay` starts at `RETRY_DELAY` (e.g., 1 second) and typically doubles with each subsequent retry.
        5.  **Logging:** All errors and retry attempts are logged with details.
        6.  **Final Error Return:** If all retries are exhausted, an error dictionary like `{"error": "LLM call failed after X attempts: [original_error]"}` is returned.

*   **Job Description Parsing (`src/core/job_parser_regex.py` via `/process-job` API):**
    *   **Rationale:** While resumes demand sophisticated LLM parsing due to their extreme variability, job descriptions from some sources (or if users input them manually with some structure) might be parseable with a simpler, faster, and cheaper regex/heuristic-based approach. This provides an alternative pathway.
    *   **How it Works:** The `/process-job` endpoint in `src/api/main.py` calls `parse_job_regex` (aliased from `extract_fields_from_text` in `job_parser_regex.py`).
        *   This function takes the full text of the job description and the PDF path (if applicable, though its primary use of PDF path isn't fully clear in the snippet, it might be for source identification).
        *   It uses a series of regular expressions (some defined in `src/utils/regex_patterns.py`, others could be inline) and string manipulation heuristics to find common job fields:
            *   `title`: Looks for "Job Title:" or uses the first few lines.
            *   `company_name`: Looks for "Company:" or patterns like "...is hiring".
            *   `location_display`: Looks for "Location:" or common city/country patterns.
            *   `salary_min`, `salary_max`, `currency`, `salary_period`: Uses `extract_salary_information` which has multiple regexes for different salary formats (e.g., "£30k - £40k", "$50,000 per year").
            *   `contract_type`, `contract_time`: Uses `extract_job_type` for terms like "Permanent", "Contract", "Full-time".
            *   `skills`: Uses `extract_skills_from_text` with a predefined `COMMON_SKILLS` list and `SKILL_NORMALIZATION_MAP`.
            *   `experience_level`: Uses `normalize_experience_level`.
            *   `required_experience_years`: Uses `extract_experience_years`.
        *   It's a best-effort extraction. If a field isn't found or a pattern doesn't match, the field in the output JSON might be `None` or empty.
    *   **No LLM Integration by Default for this Path:** This path provides a non-LLM alternative for job descriptions, making it potentially faster and free of API costs for that specific task if the JDs are reasonably structured or if high-precision semantic parsing isn't the primary goal for them.

---

### 📄 Key Files for Eva (Module 2 - Document Processor Service - with focus)

*(This list remains consistent with the previous detailed response, but Eva's understanding of each file's specific contribution to parsing *both* resumes (LLM) and jobs (potentially Regex) should be deeper.)*

| File Path                                                | 🌟 Role in Parsing Resumes (LLM) & Jobs (Regex/LLM)                                                                                                                                                                                                                                                                                                                                                                                                                      |
| :------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/api/main.py`                                        | 🚪 **Service Entrypoint**: Defines FastAPI app. Has `/process-resume` endpoint (calls `core.parser.parse_document` for LLM-based resume parsing) AND `/process-job` endpoint (calls `core.job_parser_regex.extract_fields_from_text` for regex-based job parsing). Handles file uploads.                                                                                                                                                                                |
| `src/core/parser.py`                                     | **🌟 Resume Orchestrator (LLM)**: `parse_document` & `parse_document_logic_async`. The heart of the *resume* parsing pipeline. Manages PDF/DOCX extraction, layout analysis (PDFs), concurrent sectional LLM calls via `LLMProcessor`, result aggregation, post-processing (experience calc, deduplication), and final schema validation.                                                                                                                             |
| `src/core/job_parser_regex.py`                           | **🛠️ Job Description Parser (Regex/Heuristic)**: `extract_fields_from_text`. Contains specific regex patterns and logic to extract fields like title, company, skills, salary, experience directly from the text of a *job description*, typically without LLM calls. This is the primary parser for the `/process-job` endpoint.                                                                                                                                     |
| `src/core/extractor.py` (`DocumentExtractor`)            | **📄 Raw Content Getter**: Used by *both* `parser.py` (for resumes) and potentially by `job_parser_regex.py` (if it needs to get text from a file path rather than direct text). Delegates to `pdf.py` or `docx.py`.                                                                                                                                                                                                                                                  |
| `src/extractors/pdf.py`                                  | **📄 PDF Specialist**: `extract_pdf_data_pymupdf`. Uses PyMuPDF to get `full_text`, and importantly, `blocks` with `bbox` coordinates and font info from PDFs. This layout data is crucial for `LayoutProcessor` in resume parsing.                                                                                                                                                                                                                                   |
| `src/extractors/docx.py`                                 | **📝 DOCX Specialist**: `extract_text_from_docx`. Uses `python-docx` for text from DOCX. Provides `full_text`.                                                                                                                                                                                                                                                                                                                                                             |
| `src/layout/layout_analyzer.py`                          | **📐 PDF Section Detector (Resumes)**: `LayoutProcessor` class. Uses heuristics (font size, boldness, keywords from `patterns.py`) on PDF `blocks` to identify semantic sections in resumes.                                                                                                                                                                                                                                                                               |
| `src/layout/patterns.py`                                 | **🔍 Header Keywords (Resumes)**: Contains regex patterns (e.g., `COMMON_SECTION_PATTERNS`) used by `LayoutProcessor` to spot section headers in resumes.                                                                                                                                                                                                                                                                                                                    |
| `src/llm/client.py`                                      | **🤖 Gemini API Client (Resumes)**: `generate_structured_json_async`. Primary function for sending formatted prompts to Gemini and receiving structured JSON responses. Includes crucial error handling, retry logic with exponential backoff, and JSON parsing.                                                                                                                                                                                                         |
| `src/llm/prompt.py`                                      | **📜 Prompt Loader (Resumes)**: `load_prompt`. Fetches the content of XML prompt templates from the `prompts/` directory based on `prompt_name`.                                                                                                                                                                                                                                                                                                                             |
| `prompts/*.xml` (e.g., `experience.xml`)                 | **🗣️ LLM Instructions (Resumes)**: Individual XML files defining the task, structure, context, and output examples for Gemini for *each resume section* (e.g., experience, education, skills).                                                                                                                                                                                                                                                                         |
| `src/text_processing/base_processor.py`                  | **🛠️ Base LLM Logic (Resumes)**: `LLMProcessor` class. A reusable component that takes a `prompt_name`, `text_content`, `validator_func`, and `schema_section_key`. It formats the prompt (injecting text and schema example), calls `llm/client.py`, and then invokes the specific validator for the section's output.                                                                                                                                                   |
| `src/text_processing/experience.py` (and other `*.py`)   | **🎯 Section Specialists (Resumes)**: Each file (e.g., `experience.py`, `education.py`) instantiates an `LLMProcessor` for its specific resume section and defines a `_validate_...` function to clean and ensure the structure of Gemini's JSON output for that section. The main `async def extract_...()` function in each just calls its `LLMProcessor` instance.                                                                                                   |
| `src/utils/schema_validator.py`                          | **✅ Final Output Validator (Resumes)**: `validate_result`. Uses `jsonschema` library to validate the *entire aggregated resume JSON* against `schemas/base_output_schema.json`.                                                                                                                                                                                                                                                                                           |
| `schemas/base_output_schema.json`                        | **🗺️ Resume Output Blueprint**: The target JSON schema that the *resume parsing pipeline* (via `parser.py`) aims to produce. Defines all fields and their types for a parsed resume.                                                                                                                                                                                                                                                                                       |
| `config/mappings.py`                                     | **🔗 Configuration Links (Mainly Resumes)**: `SECTION_PROCESSING_MAP` (layout section → processor func), `FALLBACK_SECTIONS_MAP` (header alias → canonical section), `PROCESSOR_TO_SCHEMA_PATH_MAP` (processor → schema path for examples/merging). Used heavily by `parser.py` and `LayoutProcessor`.                                                                                                                                                                           |
| `src/utils/regex_patterns.py`                            | **🔍 Entity Patterns (Jobs & Resume Fallbacks)**: Contains pre-compiled regex for extracting specific entities like emails, phone numbers, complex dates, and potentially some patterns used by `job_parser_regex.py`. `personal_info.py` might use these as fallbacks if LLM fails for contacts.                                                                                                                                                                     |

---


### 🔐 Important Concepts to Explain Simply (Module 3 Focus)

Eva needs to be able to break down complex NLP and ML concepts into understandable explanations. Here’s how she can explain key terms relevant to this module:

| Concept                                 | 💬 Eva’s Simple Explanation (Short Version)                                                                                                | 💡 Eva’s Elaborated Explanation (for Deeper Dives)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Where it's Used                                                                                                                                                                                                                                                                                                                                                           |
| :-------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **🔡 Text Embeddings**                  | "Turning words and sentences into meaningful number lists (vectors) so the computer can understand their relationships."                     | "Text embeddings are like creating a rich numerical fingerprint for pieces of text—be it a single word, a sentence, or even a whole paragraph. We use a pre-trained model, specifically **`all-MiniLM-L6-v2`** from `sentence-transformers`, which has learned from vast amounts of text. When we feed it a piece of resume text (like a summary or a job description), it outputs a list of numbers, typically a 384-dimensional vector in our case. The magic is that texts with similar meanings will have vectors that are 'close' to each other in this high-dimensional space, even if they don't use the exact same words. This allows the NLP model to grasp semantic similarities." | `services/nlp_service/src/model.py` (within `ResumeMultiHeadClassifier`, `self.model_.transformer.encode()` generates these for resume text). Mentioned in dissertation (Sec 5.2.3) for generating document, section, and skill-level embeddings.                                                                                                                 |
| **📐 Cosine Similarity**                 | "A math trick to see how alike two of those number lists (vectors) are. A score of 1 means they're identical in direction, 0 means unrelated." | "Once we have text represented as embedding vectors (those lists of numbers), cosine similarity is a way to measure how similar their 'directions' are in that multi-dimensional space. Imagine two arrows; if they point in almost the same direction, their cosine similarity is close to 1. If they are perpendicular (totally unrelated in meaning), it's close to 0. If they point in opposite directions, it's close to -1 (though for text similarity, we usually care about 0 to 1). It's better than just measuring distance because it's not affected by the length (magnitude) of the vectors, only their orientation. So, a short, concise job description can still be highly similar to a longer, more verbose resume summary if they cover the same core concepts." | `services/matching_service/src/core/scoring.py` (within `keyword_score`, `sklearn.metrics.pairwise.cosine_similarity` is used on TF-IDF vectors). If skill embeddings were used for matching (as per dissertation Sec 4.4.3 "Semantic Skill Matching"), cosine similarity would be the go-to metric

| **🧠 Semantic Search**                  | "Finding documents or sections that are similar in meaning, not just matching words. It’s like Google for understanding!"                       | "Semantic search is a powerful technique that goes beyond traditional keyword matching. Instead of just looking for exact words or phrases, it uses the embeddings we talked about earlier to find documents or sections that are semantically similar. For example, if a job description mentions 'software development' and a resume has 'building software applications', a semantic search would recognize these as related even though they don't share the same words. This is crucial for matching candidates to jobs based on their actual skills and experiences rather than just surface-level keywords." | `services/matching_service/src/core/scoring.py` (within `keyword_score`, `sklearn.metrics.pairwise.cosine_similarity` is used on TF-IDF vectors). If skill embeddings were used for matching (as per dissertation Sec 4.4.3 "Semantic Skill Matching"), cosine similarity would be the go-to metric.                                                                                     

| **🗂️ TF-IDF (Term Frequency-Inverse Document Frequency)** | "A way to figure out how important a word is in a document compared to all documents. It helps find unique words." | "TF-IDF is a statistical measure that evaluates how important a word is to a document in a collection or corpus. The 'Term Frequency' (TF) part counts how often a word appears in a document, while the 'Inverse Document Frequency' (IDF) part measures how common or rare that word is across all documents. If a word appears frequently in one document but rarely in others, it gets a high TF-IDF score, indicating it's likely important for that document's content. This helps us filter out common words (like 'the', 'is', etc.) and focus on the more meaningful terms that can help differentiate documents." | `services/matching_service/src/core/scoring.py` (within `keyword_score`, `sklearn.metrics.pairwise.cosine_similarity` is used on TF-IDF vectors). If skill embeddings were used for matching (as per dissertation Sec 4.4.3 "Semantic Skill Matching"), cosine similarity would be the go-to metric.                                                                                      |

| **🧩 Sectional LLM Processing**        | "Breaking down a big task (like parsing a resume) into smaller, focused tasks for better accuracy and efficiency." | "Sectional LLM processing is a strategy where we divide a larger task (like parsing an entire resume) into smaller, more manageable sections (like 'Experience', 'Education', etc.). Each section is processed separately, often using specialized prompts tailored to that section's content. This allows the LLM to focus on one specific part at a time, improving accuracy and reducing the risk of overwhelming the model with too much information at once. It also helps in managing costs since we can optimize the number of tokens sent to the LLM for each section." | `services/document_processor/src/core/parser.py` (within `parse_document()`, orchestrates sectional processing).                                                                                                           |

### "Eva must be able to answer" (from original plan - with significantly more detailed and confident answers for Module 2):

*   **"What does `parse_document()` in `services/document_processor/src/core/parser.py` do, and how is it different from the job parsing logic?"**
    *   **Answer:** "`parse_document()` is the primary orchestrator specifically for **resume parsing** using our sectional LLM approach. It's an asynchronous function that manages the entire pipeline:
        1.  It starts by initializing an empty JSON structure based on our `base_output_schema.json`.
        2.  It calls `DocumentExtractor` to get raw text and, critically for PDFs, detailed layout blocks including bounding box coordinates and font information.
        3.  If it's a PDF and layout analysis is enabled, it uses `LayoutProcessor` to identify semantic sections like 'Experience', 'Education', 'Skills' by analyzing font styles, positions, and keywords.
        4.  Then, for each identified section (or for predefined sections using the full text as a fallback if layout analysis is skipped or doesn't find a section), it creates an asynchronous task. Each task involves an `LLMProcessor` instance tailored for that section type (e.g., `_experience_processor` from `text_processing/experience.py`).
        5.  This `LLMProcessor` loads its specific XML prompt (e.g., from `prompts/experience.xml`), injects the section's text and a relevant schema example into it, and then calls `generate_structured_json_async` in `llm/client.py` to query the Google Gemini API.
        6.  `asyncio.gather` is used to execute all these LLM calls for different resume sections concurrently, waiting for all to complete.
        7.  The JSON results from each section are then validated by their respective `_validate_...` functions within each `LLMProcessor` and carefully merged into the main result JSON using `set_nested_value`, respecting the structure of `base_output_schema.json`.
        8.  Post-processing includes calculating total work experience and deduplicating entries.
        9.  Finally, the entire aggregated resume JSON is validated against `base_output_schema.json` using `jsonschema`.
        This is quite different from the **job parsing logic**. While job descriptions *can* be processed by a similar LLM pipeline if needed, the project also includes `src/core/job_parser_regex.py` which is called by the `/process-job` API endpoint. This module uses **regular expressions and heuristic rules** to extract fields like job title, company, skills, and salary directly from the job description text. This regex approach is generally faster and doesn't incur LLM API costs, making it suitable for job descriptions that might have more predictable structures or where a slightly less nuanced extraction is acceptable."

*   **"How does the Document Processor extract text, and what's the significance of layout information from PDFs?"**
    *   **Answer:** "Text extraction is managed by the `DocumentExtractor` in `src/core/extractor.py`:
        *   **For PDFs (`extractors/pdf.py` using PyMuPDF/`fitz`):** It opens the PDF and iterates through pages. The key method used is `page.get_text("blocks")`. This doesn't just give plain text; it returns a list of text blocks, where each block is a dictionary containing the `text` itself, its `bbox` (bounding box coordinates: x0, y0, x1, y1), the `page` number, and often, through deeper PyMuPDF capabilities, font information like size and boldness can be inferred for text within these blocks.
            *   🌟 **Significance of PDF Layout Information:** This `bbox` and font data is **absolutely critical** for our `LayoutProcessor` (`src/layout/layout_analyzer.py`). The layout analyzer uses these coordinates and font characteristics (e.g., larger font size, bold style for headers) along with keyword patterns to heuristically identify the start and end of different semantic sections in the resume (like "Experience", "Education"). Without this, we'd have to send the entire resume text to the LLM for every piece of information, which is inefficient and less accurate.
        *   **For DOCX files (`extractors/docx.py` using `python-docx`):** It opens the DOCX and iterates through paragraphs and tables to concatenate their text content. While it preserves basic structure like paragraph breaks, `python-docx` doesn't readily provide the detailed positional (bounding box) or rich font style information that PyMuPDF does. Therefore, for DOCX files, our advanced layout analysis step is typically skipped, and the system usually relies on processing the full text for each targeted section during the LLM phase, or the `/process-job` endpoint might use regex on this full text."

*   **"Why use Google Gemini for resumes instead of just more advanced regular expressions? And why are prompts structured with XML?"**
    *   **Answer:**
        *   **Gemini vs. Regex for Resumes:** "While our `job_parser_regex.py` shows regex can be effective for more structured documents or simpler fields, resumes are incredibly diverse in format, language, and how information is presented. Trying to capture all these variations with regex would lead to an unmanageable and extremely brittle system (as noted in dissertation Sec 3.2.2). **Google Gemini LLM excels at contextual understanding**. It can interpret varied phrasing, infer relationships, and extract information even when it's not in a perfectly predictable spot. For example, it can understand that "Led a team of 5 engineers on a project resulting in 20% revenue growth" belongs to an achievement within an experience entry. Regex would struggle immensely with this level of semantic parsing. The 'Sectional LLM' approach (dissertation Sec 3.6) was chosen to leverage this power while managing cost and complexity.
        *   **XML Prompts (Dissertation Fig 4.7, Code Snippet 4.3):** "We structure our prompts for Gemini using XML (e.g., `prompts/experience.xml`) for several key reasons:
            1.  ✨ **Clarity and Structure for the LLM:** XML tags like `<system_instructions>`, `<section_text_to_parse>`, `<desired_output_format_example>` provide a very clear, hierarchical structure to the prompt. This helps the LLM better distinguish between instructions, the input text, and the format it needs to produce.
            2.  📋 **Enforcing JSON Output:** A critical part of our XML prompts is an example of the exact JSON structure we want Gemini to return for that specific section, often including data types or constraints. This significantly increases the reliability of getting well-formed, valid JSON from the LLM. Dissertation Section 3.5.3 highlights JSON output generation as a key LLM capability.
            3.  ⚙️ **Maintainability:** Separating prompts into individual XML files in the `prompts/` directory makes them much easier to manage, version control, and iteratively refine than embedding long, complex string prompts in Python code.
            4.  🎯 **Specificity:** Each section (experience, education, skills) has its own tailored XML prompt with instructions and examples relevant only to that section, making the LLM's task more focused."

*   **"Describe the multi-stage JSON validation process in the Document Processor."**
    *   **Answer:** "JSON validation is critical and happens at several points:
        1.  💧 **Initial LLM Response Parsing (`llm/client.py`):** After `generate_structured_json_async` gets a text response from Gemini, it first strips any markdown fences (like ```json). Then, it attempts `json.loads()` on this cleaned string. If this fails (meaning Gemini didn't return valid JSON), it's a `JSONDecodeError`. This error is caught, and the retry mechanism (with exponential backoff) is triggered for that LLM call. If all retries fail, an error dictionary is returned.
        2.  🔬 **Section-Specific Validation (within each `LLMProcessor` in `text_processing/`):** If `json.loads()` succeeds, the resulting Python dictionary or list is passed to the `validator_func` specific to that section (e.g., `_validate_experience` in `text_processing/experience.py`). This function performs more detailed semantic and structural checks:
            *   Is the top-level structure correct (e.g., a list of dictionaries for 'experience')?
            *   Are required fields present in each item (e.g., 'job_title' and 'company_name' for an experience entry)?
            *   Are data types as expected (e.g., `technologies_used` should be a list of strings)?
            *   It also performs light data cleaning or normalization here (e.g., `utils/date_utils.py` is used to parse date strings and extract years).
            *   If these validations fail significantly, the validator returns the `default_result` for that section (e.g., an empty list).
        3.  🏆 **Final Aggregated Schema Validation (`core/parser.py` calling `utils/schema_validator.py`):** After all sections have been processed by their respective LLMs and their outputs merged by `set_nested_value` into the main `result` dictionary, the `validate_result()` function is called. This function uses the `jsonschema` library (specifically `Draft7Validator`) to validate the *entire* `result` object against our master `schemas/base_output_schema.json`. This ensures the overall structure, all required top-level and nested fields, and their data types are correct according to our defined contract. The outcome of this final validation (is_valid, error_count, list of errors) is stored in `result['metadata']['validation']`. This rigorous process ensures the JSON sent to the Backend API is reliable."

---

### ❓ Advanced Viva Questions Eva Must Be Ready For (Module 2 - Exhaustive & Extended)

*(Includes previous questions, plus new ones focusing on the dual parsing paths and deeper dissertation connections)*

1.  **"The Document Processor API (`src/api/main.py`) has `/process-resume` and `/process-job`. How does the internal processing logic differ between these two endpoints, especially regarding LLM usage?"**
    *   **Answer:**
        *   "The `/process-resume` endpoint is designed for the complex task of parsing resumes. It invokes `core.parser.parse_document`, which uses our **sectional LLM approach**. This involves layout analysis (for PDFs), splitting the resume into semantic sections, and then making concurrent, specialized Gemini LLM calls for each section using tailored XML prompts to extract detailed structured JSON.
        *   The `/process-job` endpoint, on the other hand, is intended for parsing job descriptions. As per its implementation in `src/api/main.py`, it calls `core.job_parser_regex.extract_fields_from_text`. This function relies on **regular expressions and heuristic string processing** to extract common job fields like title, company, location, skills, and salary directly from the job description text. It generally **does not use the Gemini LLM**. This choice is made because job descriptions, especially if sourced from specific platforms or entered manually, can sometimes have a more predictable structure where regex can be efficient and cost-effective. However, for highly unstructured job descriptions from diverse sources, an LLM-based approach similar to resume parsing could also be applied if higher accuracy is needed, but the current implementation for `/process-job` primarily uses the regex/heuristic path."

2.  **"Dissertation Code Snippet 4.1 shows `identify_section_headers` logic. If a PDF resume has 'WORK HISTORY' in a large, bold font, how would this function and `config/mappings.py` lead to it being processed by the 'experience' LLM prompt?"**
    *   **Answer:**
        1.  "In `src/layout/layout_analyzer.py`, `identify_section_headers` would process the text blocks. The block containing 'WORK HISTORY' would likely score high as a header due to its large font and potentially boldness.
        2.  The `_get_canonical_section` method (or similar logic within `identify_section_headers`) would then take the detected header text 'WORK HISTORY'.
        3.  It consults `FALLBACK_SECTIONS_MAP` from `config/mappings.py`. This map contains entries like `"work_history": "experience"`. So, 'WORK HISTORY' is mapped to the canonical section name 'experience'.
        4.  Later, in `src/core/parser.py`, when iterating through identified layout sections, it sees the section 'experience' with its associated text.
        5.  It uses `SECTION_PROCESSING_MAP` from `config/mappings.py` which maps `"experience": "extract_experience"`.
        6.  This tells the parser to use the `extract_experience` function (from `text_processing/experience.py`), which in turn uses an `LLMProcessor` configured with the `prompt_name="experience"` (loading `prompts/experience.xml`).
        So, the layout analysis, fallback mapping, and section-to-processor mapping work together to route the 'WORK HISTORY' text to the correct LLM prompt and processing logic for work experience."

3.  **"Explain the `LLMProcessor`'s role in `src/text_processing/base_processor.py`. How does it use the `prompt_name`, `validator_func`, and `schema_section_key` passed during its initialization by, say, `src/text_processing/skills.py`?"**
    *   **Answer:** "The `LLMProcessor` acts as a reusable engine for handling LLM-based extraction for any given resume section. When an instance like `_skills_processor` is created in `skills.py`:
        1.  **`prompt_name` (e.g., "skills"):** This tells the `LLMProcessor` to load `prompts/skills.xml` using `src/llm/prompt.py -> load_prompt()`. This XML template is stored internally.
        2.  **`validator_func` (e.g., `_validate_skills` from `skills.py`):** This function is stored. After the LLM returns a JSON response and it's parsed by `llm/client.py`, the `LLMProcessor.process()` method will call this `_validate_skills` function, passing it the LLM's output. `_validate_skills` then checks if the JSON structure is correct for skills (e.g., has `technical_skills` list, `soft_skills` list), cleans the data, and returns the validated skills dictionary or a default empty skills structure if validation fails.
        3.  **`schema_section_key` (e.g., "skills_assessment"):** This dot-separated path is used by `LLMProcessor.__init__` to look up an example snippet from the main `schemas/base_output_schema.json` (loaded via `utils/data.py`). This JSON snippet, representing the desired output structure for skills, is then injected into the `{schema_example}` placeholder in the `prompts/skills.xml` template before it's sent to Gemini. This helps guide Gemini to produce correctly formatted JSON.
        The `LLMProcessor.process(section_text)` method then orchestrates these: formats the loaded prompt with the input `section_text` and the schema example, calls the `llm/client.py` to get the JSON from Gemini, and finally passes that JSON through the stored `validator_func`."

4.  **"The dissertation's Figure 4.8 shows a 'Rule-Based Emergency Fallback' (1.7% success). Where in your `document_processor` code might such a rule-based fallback be implemented if an LLM section processor consistently fails for a critical section like 'personal_info'?"**
    *   **Answer:** "While Figure 4.8 shows it, the core `LLMProcessor.process()` method currently returns `self.default_result` if the LLM call (after retries in `llm/client.py`) fails or if its `validator_func` indicates a failure. There isn't an explicit secondary "rule-based emergency fallback" *within the `LLMProcessor` itself* for most sections in the provided code.
        However, for **`personal_info`**, the `src/text_processing/personal_info.py` module *already* implements a hybrid approach:
        *   It first attempts to extract contacts (email, phone, LinkedIn, website) using **reliable regex patterns** from `src/utils/regex_patterns.py`. This is a rule-based extraction.
        *   Then, it calls the LLM (via its `LLMProcessor`) primarily for `full_name` and `professional_summary`.
        *   If the LLM fails to provide a `full_name`, there's a `_extract_name_heuristic` function that uses simpler regex as a fallback.
        *   Similarly, location can be inferred from experience data or extracted by regex if the LLM fails for that specific field.
        So, for `personal_info`, the "rule-based emergency fallback" is partially integrated. For other sections like 'experience' or 'education', if their `LLMProcessor` returns the `default_result` (e.g., an empty list), the `core/parser.py` currently just accepts that and moves on. To fully implement the strategy in Figure 4.8 for *all* sections, `core/parser.py` or the `LLMProcessor` would need to be enhanced. After `LLMProcessor` returns a default/failure, `parser.py` could then call a *separate set of regex-based extraction functions* designed for that section as a last resort. For example, `extract_experience_regex(full_text)` could be called if `extract_experience` (LLM) yielded nothing."

5.  **"Dissertation Section 5.1.3 (Example Parsing Results) shows JSON output (Fig 5.2, 5.4). Which specific files and functions in the `document_processor` are responsible for generating the 'skills_assessment' -> 'technical_skills' array of objects, each with 'name' and 'level'?"**
    *   **Answer:**
        1.  **Input Text:** The text for the "Skills" section of the resume is first identified by `src/layout/layout_analyzer.py` (if PDF) or provided as part of `full_text`.
        2.  **Processor Invocation:** `src/core/parser.py` calls `extract_skills` from `src/text_processing/skills.py` with this text.
        3.  **LLM Call Orchestration:** `extract_skills` uses `_skills_processor` (an instance of `LLMProcessor` from `base_processor.py`).
        4.  **Prompting:** The `_skills_processor` formats `prompts/skills.xml`. This prompt specifically instructs Gemini to identify technical skills and, if possible, their proficiency levels, and to structure them as a list of objects, each with "name" and "level" keys. It also includes a schema example for `skills_assessment.technical_skills`.
        5.  **Gemini Interaction:** `src/llm/client.py` (`generate_structured_json_async`) sends this prompt to Gemini.
        6.  **Response & Validation:** Gemini returns a JSON string. `llm/client.py` parses it. Then, `_validate_skills` in `text_processing/skills.py` is called. This validator checks if `technical_skills` is a list of dictionaries and if each dictionary has a `name` (string) and an optional `level` (string). It cleans these values.
        7.  **Merging:** The `parser.py` merges this validated list of skill objects into `result['skills_assessment']['technical_skills']`. The output in Figure 5.2, like `{"name": "Python", "level": "Expert"}`, is directly a result of this structured prompting and validation."

*(Eva should also be prepared for questions 1, 2, 6-20 from the previous "Advanced Viva Questions for Module 2" as they cover crucial aspects like layout analysis details, handling corrupted files, XML prompt specifics, retry mechanisms, multilingual support, testing, scalability, local LLM alternatives, debugging, versioning, and the use of `asyncio.gather`.)*

---

### ✅ Final Review Checklist for Eva (Module 2 - Exhaustive & Consistent)

*   [ ] **Deep Walkthrough of `core/parser.py` (`parse_document_logic_async`):** Explain the full orchestration for *resumes*: init, extraction (`DocumentExtractor`), layout (`LayoutProcessor`), concurrent sectional LLM task creation (`LLMProcessor` from `text_processing/`, XML prompts from `prompts/`), `asyncio.gather` execution, result aggregation (`set_nested_value`), specific `extract_personal_info` call, post-processing (`calculate_total_experience`, `deduplicate_sections`), and final `jsonschema` validation.
*   [ ] **Contrast with `/process-job`:** Explain how the API endpoint in `src/api/main.py` for jobs calls `core/job_parser_regex.py` which uses primarily regex/heuristics, not the sectional LLM pipeline.
*   [ ] **XML Prompt Analysis:** Open `prompts/experience.xml` and `prompts/skills.xml`. For each:
    *   Identify the `<system>` instructions.
    *   Point out the `{{SECTION_TEXT}}` placeholder.
    *   Explain the specific extraction instructions for that section.
    *   Show the `<output_format>` example and how it guides Gemini to produce the desired JSON structure (e.g., a list of objects for experience, a dictionary with lists for skills).
*   [ ] **`llm/client.py` (`generate_structured_json_async`):** Detail the Gemini API call: model name used (`gemini-1.5-flash-latest`), temperature, `response_mime_type="application/json"`, safety settings. Explain the **retry loop**: conditions for retry (API error, `JSONDecodeError`, non-STOP finish reason), use of `asyncio.sleep`, and exponential backoff.
*   [ ] **`text_processing/base_processor.py` (`LLMProcessor`):** Explain its `__init__` (how it stores `prompt_name`, `validator_func`, `default_result`, `schema_section_key`, and pre-loads the prompt template and schema example) and its `async process()` method (how it formats the prompt with runtime text and the pre-loaded schema example, calls the LLM client, and then invokes its specific validator).
*   [ ] **Layout Analysis In-Depth (`layout_analyzer.py`):** Explain how `LayoutProcessor` uses PDF block coordinates and font attributes (size, boldness from PyMuPDF) combined with regex patterns from `layout/patterns.py` (like `COMMON_SECTION_PATTERNS`) and `FALLBACK_SECTIONS_MAP` from `config/mappings.py` to identify and canonically name sections. Refer to Code Snippet 4.1 from dissertation.
*   [ ] **Error Scenario - Gemini returns malformed JSON for "Education":**
    1.  `llm/client.py`: `json.loads()` fails, `JSONDecodeError` caught.
    2.  Retry loop in `llm/client.py` attempts again (up to `MAX_RETRIES`).
    3.  If all retries fail, `generate_structured_json_async` returns `{"error": "Invalid JSON..."}`.
    4.  `LLMProcessor` instance for education (in `text_processing/education.py`) receives this error dict from `llm_client`. Its `process()` method returns its `default_result` (which is `[]` for education).
    5.  `core/parser.py`: The `result['education']['degrees']` will be an empty list. The error details are logged in `result['metadata']['parsing_metrics']['llm_errors']`. Parsing continues for other sections.
*   [ ] **Data Flow for "Skills" Section (Resume):** `parser.py` gets skills section text (from layout or full text) → calls `tp.extract_skills()` → `_skills_processor.process(skills_text)` → `LLMProcessor` formats `prompts/skills.xml` (with text & schema example) → `llm.client.generate_structured_json_async` calls Gemini → Gemini returns JSON string for skills → `llm.client` parses to dict → `_validate_skills` (in `skills.py`) validates/cleans it (e.g., ensuring `technical_skills` is a list of objects) → validated skills dict returned to `parser.py` → `set_nested_value` merges it into `result['skills_assessment']`.
*   **Configuration Files' Roles:**
    *   `config/settings.py`: Provides runtime parameters like `GEMINI_MODEL_NAME`, `LLM_TEMPERATURE`.
    *   `config/mappings.py`: Crucial for `parser.py` and `LayoutProcessor`. `SECTION_PROCESSING_MAP` links canonical section names to their `text_processing` functions. `FALLBACK_SECTIONS_MAP` standardizes detected header text. `PROCESSOR_TO_SCHEMA_PATH_MAP` guides where results are merged and where schema examples for prompts are sourced.
    *   `schemas/base_output_schema.json`: The definitive target structure for parsed resumes.

Understood! Let's proceed with **Module 4: Backend API (FastAPI, PostgreSQL, MongoDB)** using the exhaustive "Definitive Deep Dive & Orchestration" structure, incorporating all the detailed explanations, emojis, and specific file references.

---

## 💎 Module 4: Backend API Gateway (FastAPI, PostgreSQL, MongoDB) - Definitive Deep Dive & Orchestration

### 🎯 **Core Objectives for Eva:**

*   🧠 **Master the Backend's Central Role:** Attain an exceptionally detailed understanding of how the `backend/` service functions as the API Gateway, orchestrating all communication between the frontend and the various specialized microservices (Document Processor, NLP Classifier, Matching Engine).
*   🗣️ **Articulate API & Data Flows with Unmatched Precision:** Fluently explain the entire lifecycle of key user requests (e.g., user registration, login, resume upload, job upload, match retrieval), detailing how data is received, validated (Pydantic), processed through inter-service calls (`httpx`), and persisted in or retrieved from both PostgreSQL (SQLAlchemy) and MongoDB (`motor`).
*   🗺️ **Navigate Key Backend Files & Structures Confidently:** Be able to pinpoint where specific functionalities reside within `backend/app/` – including `main.py` (app setup), `api/v1/endpoints/*.py` (route handlers), `core/*.py` (config, security, DB clients, HTTP client), `models/models.py` (SQLAlchemy ORM), `schemas/schemas.py` (Pydantic models), and `db/repositories/` (CRUD logic).
*   💡 **Justify Architectural & Design Choices:** Defend the use of FastAPI, the hybrid database approach, JWT for authentication, and the patterns for error handling and dependency injection, referencing the dissertation's rationale where applicable.
*   ⚙️ **Understand Database Schemas & Migrations:** Clearly explain the PostgreSQL schema defined by SQLAlchemy models, the role of MongoDB for resume JSONs, and how Alembic is used to manage PostgreSQL schema evolution.
*   🛡️ **Address Security, Configuration, and Error Handling:** Detail how user authentication works, how configurations (like database URLs and service endpoints) are managed via environment variables, and how the API handles various error conditions gracefully.

---

### 🚀 Technologies Involved (Central to the `backend/` Service)

| Category                               | Technology                                                                  | 🌟 Purpose & Significance in Backend API Gateway                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Relevant Files (Examples)                                                                                                                                                                                                                                                                                                                                                                          |
| :------------------------------------- | :-------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **📦 Core Framework & Server**         | Python 3.11+, FastAPI, Uvicorn                                              | **Foundation of the API Gateway**: Python is the language. FastAPI provides the high-performance, asynchronous web framework for defining API routes, handling requests, and managing dependencies. Uvicorn is the ASGI server that runs the FastAPI application, enabling concurrent request handling.                                                                                                                                                                                                                                                                                                                                                       | `app/main.py` (FastAPI app instance), `app/api/v1/endpoints/*.py` (route definitions). `Dockerfile` and `docker-compose.yml` specify Uvicorn for running.                                                                                                                                                                                                                                  |
| **✅ Data Validation & Serialization** | Pydantic                                                                    | **Ensuring Data Integrity & Structure**: Pydantic models (schemas) are used extensively to: <br> 1. Define the expected structure and data types of incoming API request bodies (e.g., for user registration, login). <br> 2. Automatically validate this incoming data; FastAPI returns detailed 422 errors if validation fails. <br> 3. Define the structure of outgoing API responses, ensuring consistency. FastAPI uses these models to serialize Python objects (like SQLAlchemy models or custom dicts) into JSON. <br> Also used in `app/core/config.py` for loading and validating environment settings. | `app/schemas/schemas.py` (e.g., `UserCreate`, `User`, `ResumeCreateResponse`, `Job`, `MatchRequest`, `MatchResponse`). Used as type hints in endpoint function parameters and `response_model`.                                                                                                                                                                                          |
| **🐘 Relational Database ORM**        | SQLAlchemy (with `psycopg2-binary` adapter)                                 | **Interacting with PostgreSQL**: SQLAlchemy is the ORM used to define the PostgreSQL database schema as Python classes (in `app/models/models.py`). It provides an object-oriented way to query and manipulate data (CRUD operations) in tables like `users`, `resumes` (metadata), `jobs`, `skills`, `matches`. `psycopg2-binary` is the low-level Python driver for PostgreSQL.                                                                                                                                                                                                                              | `app/models/models.py` (table definitions), `app/db/session.py` (engine & `SessionLocal` setup, `get_db` dependency), `app/db/repositories/` (e.g., `user_repository.py` containing CRUD logic), `app/utils/crud.py` (e.g., `get_or_create_skills`).                                                                                                                             |
| **📜 Database Migrations**           | Alembic                                                                     | **Managing PostgreSQL Schema Evolution**: As the SQLAlchemy models in `app/models/models.py` change (e.g., new tables, columns), Alembic is used to generate and apply migration scripts that update the live PostgreSQL database schema accordingly, ensuring schema consistency without manual SQL.                                                                                                                                                                                                                                        | `alembic/` directory (contains version scripts), `alembic.ini` (configuration file). Commands like `alembic revision` and `alembic upgrade head` are used.                                                                                                                                                                                                                  |
| **🍃 Document Database Driver**      | `motor` (Asynchronous MongoDB Driver)                                       | **Interacting with MongoDB Asynchronously**: `motor` is the official asyncio-compatible driver for MongoDB. The Backend API Gateway uses it to perform non-blocking operations (primarily `insert_one`) to store the full parsed resume JSON (received from the Document Processor) into the `parsed_resumes` collection in MongoDB.                                                                                                                                                                                                         | `app/core/mongo.py` (sets up the `AsyncIOMotorClient` and `get_mongo_db` dependency), `app/api/v1/endpoints/resumes.py` (uses `mongo_db.insert_one()`).                                                                                                                                                                                                                           |
| **📡 Asynchronous HTTP Client**      | `httpx`                                                                     | **Calling Other Microservices**: An `httpx.AsyncClient` is used to make asynchronous (non-blocking) HTTP requests from the Backend API Gateway to the other internal microservices (Document Processor, NLP Classifier, Matching Service). This is crucial for maintaining the gateway's responsiveness while waiting for downstream services.                                                                                                                                                                                                         | `app/core/http_client.py` (initializes and provides the shared `AsyncClient` via `get_async_client` dependency). Used in `app/api/v1/endpoints/resumes.py`, `jobs.py`, `matching.py`, and `app/services/nlp_client.py`.                                                                                                                                                    |
| **🔑 Authentication (JWT)**          | `python-jose[cryptography]`, `passlib[bcrypt]`                              | **Securing User Access**: <br> - `python-jose` (with `cryptography` backend) is used for encoding (creating) and decoding (validating) JSON Web Tokens (JWTs). <br> - `passlib` (with `bcrypt` scheme) is used for securely hashing user passwords before storing them in the `users` table and for verifying passwords during login.                                                                                                                                                                                                          | `app/core/security.py` (contains `create_access_token`, `verify_password`, `get_password_hash`, and the `get_current_active_user` dependency which decodes and validates JWTs).                                                                                                                                                                                                   |
| **🔧 Configuration Management**    | `python-dotenv`, Pydantic's `BaseSettings`                                  | **Loading Settings**: `python-dotenv` loads variables from a `.env` file. Pydantic's `BaseSettings` class in `app/core/config.py` then reads these environment variables (e.g., `DATABASE_URL`, `MONGO_URI`, `SECRET_KEY`, URLs of other services) into a typed configuration object (`settings`), making them accessible throughout the backend application.                                                                                                                                                                            | `app/core/config.py` (`Settings` class), `.env.template` and `.env` file at the `backend/` root.                                                                                                                                                                                                                                                                                       |
| **📝 Logging**                       | Python `logging` module                                                       | **Tracking & Debugging**: Standard Python logging is used to record application events, API requests, errors, and debug information, aiding in monitoring and troubleshooting.                                                                                                                                                                                                         | Used throughout `backend/app/` modules. May be configured via `app/utils/setup_logger()` or a similar utility, or basic configuration in `app/main.py`.                                                                                                                                                                                                                              |
| **🤝 CORS Middleware**             | FastAPI's `CORSMiddleware`                                                  | **Enabling Frontend Communication**: Configured in `app/main.py` to allow Cross-Origin Resource Sharing. This is essential for the React frontend (running on a different port, e.g., 3000) to make API requests to the Backend API Gateway (running on port 8000).                                                                                                                       | `app/main.py`.                                                                                                                                                                                                                                                                                                                                                                |

---

### 🚶 Step-by-Step In-Depth Backend API Gateway Operations

This section details the internal workings of the `backend/` service for key flows.

#### 1. 🚀 Application Startup & Initialization (`app/main.py`)

*   When Uvicorn starts the FastAPI application defined in `app/main.py`:
    *   ✨ An instance of `FastAPI()` is created, with `title`, `description`, `version`, and crucially, a `lifespan` manager.
    *   **🕊️ Lifespan Manager (`@asynccontextmanager lifespan`):**
        *   **On Startup:**
            *   `await start_async_client()` (from `app/core/http_client.py`): This initializes a single, shared `httpx.AsyncClient` instance. This client has pre-configured timeouts and potentially connection limits. It's stored globally (e.g., `_client`) within `http_client.py` to be reused for all outgoing asynchronous HTTP calls to other microservices. This reuse is vital for performance (connection pooling).
            *   `await start_mongo_client()` (from `app/core/mongo.py`): This initializes a single, shared `motor.AsyncIOMotorClient` instance, connects to the MongoDB server (URI from `settings`), and gets a handle to the specific database (`settings.MONGO_DB_NAME`). This client and database handle are stored globally (e.g., `mongo_client`, `mongo_db`) for reuse.
            *   Logging confirms initialization of these critical shared resources.
        *   **On Shutdown (when Uvicorn stops the app):**
            *   `await stop_async_client()`: Gracefully closes the shared `httpx.AsyncClient` and its connection pool.
            *   `await stop_mongo_client()`: Closes the MongoDB client connection.
    *   **🌐 CORS Middleware (`app.add_middleware(CORSMiddleware, ...)`):**
        *   Configured to allow requests from specific origins (like `http://localhost:3000` for the frontend during development, or production frontend URLs).
        *   Allows specified HTTP methods (`*` for all), headers (`*` for all), and credentials (`allow_credentials=True` if JWTs/cookies are sent cross-origin, though JWTs in headers are more common). This is essential for the browser to permit the frontend to call the backend API.
    *   **🧩 API Routers (`app.include_router(...)`):**
        *   The application includes various `APIRouter` instances defined in `app/api/v1/endpoints/`. For example:
            *   `app.include_router(resumes_router.router, prefix="/api/v1", tags=["Resumes"])`
            *   `app.include_router(jobs_router.router, prefix="/api/v1", tags=["Jobs"])`
            *   `app.include_router(matching_router.router, prefix="/api/v1", tags=["Matching"])`
            *   And routers for user authentication (`/api/v1/login`, `/api/v1/register`).
        *   This modularizes the API, keeping related endpoints grouped in separate files. The `prefix="/api/v1"` ensures all these routes are versioned. `tags` group them in the auto-generated Swagger/OpenAPI documentation.
    *   **⚕️ Health Check Endpoints (`@app.get("/")`, `@app.get("/api/v1/health")`):**
        *   Simple endpoints that return a success status, used for basic liveness checks by monitoring systems or load balancers.

#### 2. 👤 User Registration Flow (e.g., `POST /api/v1/register` in `app/main.py` or `app/api/v1/endpoints/users.py`)

*   **📥 Request:** Frontend sends a JSON payload with `name`, `email`, and `password`.
*   **✅ Pydantic Validation:** FastAPI automatically validates the incoming JSON against the `schemas.UserCreate` Pydantic model. If fields are missing, have wrong types, or violate constraints (e.g., invalid email format, password too short), FastAPI returns a 422 Unprocessable Entity response with detailed error messages *before* the endpoint code even runs.
*   **🔑 Business Logic (Endpoint Handler):**
    *   `db: Session = Depends(get_db)`: A SQLAlchemy session for PostgreSQL is injected.
    *   **Check for Existing User:** `existing_user = user_repo.get_by_email(db, email=user_in.email)` (from `app/db/repositories/user_repository.py`). This queries the `users` table. If a user with that email exists, an `HTTPException` (409 Conflict) is raised.
    *   **🔒 Password Hashing:** `hashed_password = get_password_hash(user_in.password)` (from `app/core/security.py`). This uses `passlib` with `bcrypt` to securely hash the plain-text password. **Only the hash is stored.**
    *   **Create User Record:** A `models.User` SQLAlchemy object is created with the email, name, and `hashed_password`.
    *   **💾 Database Commit:** `user_repo.create(db, obj_in=user_data_with_hashed_password)` adds the new user to the session and commits it to PostgreSQL. `db.refresh(user)` ensures the `user` object has the new ID from the database.
*   **📤 Response:** Returns the newly created user data (Pydantic `schemas.User` model, excluding the password hash) with a 201 Created HTTP status.

#### 3. 🔑 User Login Flow (e.g., `POST /api/v1/login` in `app/main.py` or `app/api/v1/endpoints/users.py`)

*   **📥 Request:** Frontend sends JSON with `email` and `password`. (Note: `PROJECT_DOCUMENTATION.md` shows these as `Body(...)` parameters, which is fine for FastAPI to parse from `application/x-www-form-urlencoded` or JSON depending on `Content-Type`).
*   **✅ Business Logic:**
    *   `user = user_repo.get_by_email(db, email=email)`: Fetches user from PostgreSQL.
    *   **Password Verification:** If no user or if `verify_password(password, user.hashed_password)` (from `app/core/security.py`) returns `False`, an `HTTPException` (401 Unauthorized) is raised.
    *   **📜 JWT Creation:** If credentials are valid, `create_access_token(data={"sub": user.email}, user=user)` is called.
        *   The `data` payload includes `sub: user.email` (standard subject claim).
        *   The `user` object itself is passed to include `user.name` and `user.email` directly in the token payload, which can be useful for the frontend to display user info without another API call.
        *   An expiration time (`ACCESS_TOKEN_EXPIRE_MINUTES` from `config.py`) is added.
        *   The payload is signed with `SECRET_KEY` and `ALGORITHM` (e.g., HS256) using `python-jose`.
*   **📤 Response:** Returns a JSON `{"access_token": "THE_JWT_STRING", "token_type": "bearer"}`.

#### 4. 📄 Resume Upload & Processing Orchestration (POST `/api/v1/resumes` in `app/api/v1/endpoints/resumes.py` -> `upload_resume`)

*   **🛡️ Authentication:** The `current_user: models.User = Depends(get_current_active_user)` dependency is resolved first.
    *   `get_current_active_user` itself depends on `get_current_user`, which uses FastAPI's `OAuth2PasswordBearer(tokenUrl="/api/token")` to extract the Bearer token from the `Authorization` header.
    *   `python-jose.jwt.decode()` validates the token's signature and expiry against `settings.SECRET_KEY` and `settings.ALGORITHM`.
    *   The user's email (from `sub` claim) is used to fetch the `models.User` object from PostgreSQL via `user_repo`. If any step fails, 401 Unauthorized is raised.
*   **📁 File Reception:** `file: UploadFile = File(...)` receives the uploaded resume. `original_filename` is extracted.
*   **📞 Call Document Processor Service (Asynchronously):**
    *   `http_client: httpx.AsyncClient = Depends(get_async_client)` injects the shared async HTTP client.
    *   A `files` dictionary is prepared for multipart form upload: `{'file': (original_filename, await file.read(), file.content_type)}`.
    *   `response = await http_client.post(DOCUMENT_PROCESSOR_URL, files=files_to_send)`. `DOCUMENT_PROCESSOR_URL` (e.g., `http://document_processor:8001/process-resume`) comes from `settings`.
    *   **Error Handling for DP Call:**
        *   If `response.status_code != 200`, logs error, extracts detail from DP's JSON error response, and raises 502 Bad Gateway.
        *   If `httpx.TimeoutException` or `httpx.RequestError`, raises 504 Gateway Timeout or 503 Service Unavailable.
        *   If `response.json()` fails (DP returned non-JSON), raises 502 Bad Gateway.
*   **📄 Process Document Processor's JSON Response:**
    *   `dp_output_data = response.json()`.
    *   **(Optional) Transformation:** `canonical_data = transform_dp_output_to_canonical(dp_output_data, original_filename)` (from `app/core/data_transformer.py`) could be used here if the DP's output schema (`base_output_schema.json`) isn't directly what the backend wants to store or send to NLP. The current code seems to assume they are compatible or uses `jsonschema` directly on `dp_output_data`.
    *   **📜 Validation against Canonical Schema:** The `resumes.py` endpoint loads `app/schemas/resume_canonical.json` (or a similar path) and uses `jsonschema.validate(instance=canonical_data, schema=RESUME_SCHEMA)`. If `ValidationError`, raises 500.
*   **💾 Store Full JSON in MongoDB:**
    *   `mongo_db: AsyncIOMotorDatabase = Depends(get_mongo_db)` injects the async Mongo client.
    *   Metadata like `_user_id_bk` (backend user ID), `_original_filename_bk`, `_created_at_bk` (ISO format) are added to the `canonical_data`.
    *   `insert_result = await mongo_db[MONGO_COLLECTION_NAME].insert_one(canonical_data)`. `MONGO_COLLECTION_NAME` is `parsed_resumes`.
    *   `mongo_id_str = str(insert_result.inserted_id)` is obtained.
    *   Handles `PyMongoError` by raising 500.
*   **🧠 Call NLP Classifier Service (Asynchronously):**
    *   `nlp_result = await call_nlp_classify(canonical_data)` (from `app/services/nlp_client.py`). This function POSTs `canonical_data` to the NLP service's `/classify-role` endpoint.
    *   `nlp_result` will contain `fine_role`, `fine_confidence`, `role_probabilities`, etc. If NLP call fails, `nlp_result` might be `None` or contain minimal error info; the endpoint proceeds but stores nulls for NLP fields.
*   **🐘 Store Metadata in PostgreSQL:**
    *   A `models.Resume` SQLAlchemy object is created.
    *   Populated with `user_id=current_user.id`, `title` (derived from `canonical_data` or filename), `original_file_path=original_filename`, `mongo_doc_id=mongo_id_str`.
    *   Crucially, `predicted_role`, `role_confidence`, and `nlp_role_probabilities` from `nlp_result` are also set.
    *   `db.add(db_resume); db.commit(); db.refresh(db_resume)`. Handles `IntegrityError` (e.g., duplicate `mongo_doc_id`) with 409, other `SQLAlchemyError` with 500.
*   **📤 Response to Frontend:** Returns 201 Created with `schemas.ResumeCreateResponse` containing `resume_id` (PostgreSQL ID), `mongo_doc_id`, `predicted_role`, and `role_confidence`.

*(The flows for **Job Upload (`/api/v1/jobs/upload`)** and **Job Matching (`/api/v1/resumes/{resume_id}/matches`)** follow similar patterns of authentication, calling downstream services (`Document Processor` for jobs, `Matching Service` for matches), interacting with databases, and returning structured responses, as detailed in the previous comprehensive Module 1 and Module 4 outlines.)*

---

### 📂 Key Files Eva Should Know (Backend API Gateway - with detailed focus)

| File Path                                      | 🌟 Role & Key Functionality in Backend API                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| :--------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `app/main.py`                                  | 🚀 **Application Heart**: Initializes the main `FastAPI` application instance. Sets up global configurations like API title, version. Crucially, it defines the **`lifespan` manager** which handles startup (initializing shared `httpx.AsyncClient` for inter-service calls via `app/core/http_client.py`, and the `motor.AsyncIOMotorClient` for MongoDB via `app/core/mongo.py`) and shutdown events. It also configures **`CORSMiddleware`** to allow requests from the frontend. Most importantly, it uses `app.include_router()` to mount all the API endpoint modules from `app/api/v1/endpoints/`. |
| `app/api/v1/endpoints/resumes.py`              | 📄 **Resume Operations Hub**: Defines API routes for resume-related actions. Key endpoint: `POST /resumes` (`upload_resume` function) which orchestrates the entire resume ingestion pipeline: authenticates user, calls Document Processor, stores full JSON in MongoDB, calls NLP Classifier, stores metadata & NLP results in PostgreSQL. Also likely has `GET /resumes` (list user's resumes) and `GET /resumes/{resume_id}/full` (get full merged data for one resume).                                                                                                                                     |
| `app/api/v1/endpoints/jobs.py`                 | 💼 **Job Operations Hub**: Defines API routes for job postings. Key endpoint: `POST /jobs/upload` (`upload_job_description`) which handles uploading a job description file. It calls the Document Processor (potentially its `/process-job` endpoint which might use regex via `job_parser_regex.py`), optionally calls NLP for role classification of the job, and stores the structured/raw job data in PostgreSQL and/or MongoDB. Also likely includes GET endpoints for listing/searching jobs.                                                                                                     |
| `app/api/v1/endpoints/matching.py`             | ✨ **Matching Orchestration Point**: Defines API routes for triggering job matching. Key endpoint: `GET /resumes/{resume_id}/matches` (`get_job_matches_for_resume`). This endpoint: 1. Fetches the target resume's full data (Mongo JSON) and NLP classification (Postgres). 2. Fetches a list of candidate jobs (from Postgres). 3. Sends this data to the Matching Service. 4. Receives ranked match results and returns them to the frontend. May also store match results in PostgreSQL `matches` table.                                                                                               |
| `app/schemas/schemas.py`                       | 📦 **Data Contracts (Pydantic)**: Defines all Pydantic models used for: <br> 1. Validating the structure and types of incoming API request bodies (e.g., `UserCreate`, `LoginRequest`, `MatchRequest`). <br> 2. Serializing outgoing API responses, ensuring they conform to a defined structure (e.g., `User`, `Resume`, `Job`, `MatchResponse`, `ResumeCreateResponse`, `FullResume`). <br> These models are FastAPI's primary mechanism for data validation and ensuring clear API contracts.                                                                                                         |
| `app/models/models.py`                         | 🐘 **PostgreSQL Blueprint (SQLAlchemy)**: Defines the structure of all tables in the PostgreSQL database using SQLAlchemy ORM classes (e.g., `User`, `Resume`, `Job`, `Skill`, `Match`, `Education`, `Experience`). Specifies column types, primary keys, foreign keys, relationships (e.g., `User.resumes = relationship("Resume", ...)`), and table names. This is the "source of truth" for the relational database schema.                                                                                                                                                                       |
| `app/db/session.py`                            | 🔗 **PostgreSQL Connection Factory**: Sets up the SQLAlchemy database engine (`create_engine(settings.DATABASE_URL)`) and a `SessionLocal` (a `sessionmaker` instance). Provides the crucial `get_db()` FastAPI dependency, which yields a new SQLAlchemy session for each API request and ensures it's closed afterwards.                                                                                                                                                                                                                                                                    |
| `app/core/mongo.py`                            | 🍃 **MongoDB Connection Factory**: Initializes the asynchronous MongoDB client (`motor.AsyncIOMotorClient`) during application startup (via `lifespan` in `main.py`). Provides the `get_mongo_db()` FastAPI dependency, which yields a handle to the application's MongoDB database instance (`settings.MONGO_DB_NAME`).                                                                                                                                                                                                                                                               |
| `app/core/http_client.py`                      | 📡 **Inter-Service Communicator**: Initializes and provides a shared `httpx.AsyncClient` instance during application startup (via `lifespan`). This client is injected into endpoints via `Depends(get_async_client)` and is used to make all asynchronous HTTP calls to the other microservices (Document Processor, NLP Classifier, Matching Service). Using a shared client enables connection pooling.                                                                                                                                                                      |
| `app/core/security.py`                         | 🔑 **Authentication & Authorization Core**: <br> - `get_password_hash()` & `verify_password()`: Use `passlib` with `bcrypt` for secure password hashing and verification during user registration and login. <br> - `create_access_token()`: Generates JWTs using `python-jose` after successful login, embedding user's email (as `sub`) and other info. <br> - `oauth2_scheme = OAuth2PasswordBearer(...)`: FastAPI security utility to extract tokens. <br> - `get_current_user()` & `get_current_active_user()`: FastAPI dependencies that decode and validate JWTs from request headers, fetch the user from DB, and make the authenticated `User` object available to protected endpoints. |
| `app/core/config.py`                           | ⚙️ **Central Configuration**: Defines a Pydantic `Settings` class that loads all application configurations from environment variables (often sourced from a `.env` file). This includes database connection strings (`DATABASE_URL`, `MONGO_URI`), URLs for downstream microservices (`DOCUMENT_PROCESSOR_URL`, etc.), and JWT parameters (`SECRET_KEY`, `ALGORITHM`). Provides a single, typed source for all settings.                                                                                                                                                         |
| `app/services/nlp_client.py`                   | 📞 **NLP Service Client**: Contains the `call_nlp_classify(resume_json)` async function. This function constructs the JSON payload and uses the shared `httpx.AsyncClient` (from `core/http_client.py`) to make a POST request to the NLP Classifier service's `/classify-role` endpoint and handles its response.                                                                                                                                                                                                                                                          |
| `app/utils/crud.py`                            | 🗄️ **Database Helpers (PostgreSQL)**: Contains reusable utility functions for common database operations that might be more complex than simple repository calls. For example, `get_or_create_skills(db, skill_names_list)` checks if skills exist in the `skills` table and creates them if not, returning a list of `Skill` ORM objects. This is used when saving parsed jobs or resumes.                                                                                                                                                                       |
| `alembic/` (directory) & `alembic.ini`         | 📜 **PostgreSQL Schema Version Control**: Alembic configuration (`alembic.ini`) and the `alembic/versions/` directory (containing individual migration scripts, e.g., `be4f16570563_initial.py`) are used to manage and apply changes to the PostgreSQL database schema as the application evolves.                                                                                                                                                                                                                                                           |
| `backend/.env.template` & `backend/.env`       | 🤫 **Secrets & Settings Source**: Template and actual file for storing environment variables specific to the backend service, like database credentials and JWT secrets.                                                                                                                                                                                                                                                                                                                                                                                            |

---

### "Eva must know" (from original plan - with more detailed and confident answers for Module 4):

*   **"How the `/api/v1/resumes` (POST for upload) endpoint works in detail."**
    *   **Answer:** *(This was covered extensively in the "Detailed Operational Flow -> 4. 📄 Resume Upload & Processing Orchestration" section above. Eva should be able to walk through all 10 sub-steps, from authentication to calling DP, storing in Mongo, calling NLP, storing in Postgres, and responding to the frontend, naming key files and functions involved at each stage.)*

*   **"What the `/api/v1/resumes/{resume_id}/matches` endpoint does."**
    *   **Answer:** *(This was covered extensively in the "Detailed Operational Flow -> 6. ✨ Job Matching" section above. Eva should detail how it fetches the target resume (from Mongo & Postgres), fetches candidate jobs (from Postgres), calls the Matching Service with this data, and then returns the ranked matches to the frontend.)*

*   **"Where user data and job data are stored, and be specific about the database technology and tables/collections."**
    *   **Answer:**
        *   👤 **User Data:** Stored in **PostgreSQL** in the `users` table (defined in `app/models/models.py`). This table includes fields like `id` (PK), `email` (unique, indexed), `name`, and `hashed_password`.
        *   📄 **Resume Data (Full Parsed JSON):** Stored in **MongoDB** in the `parsed_resumes` collection (collection name from `app/api/v1/endpoints/resumes.py` or config). Each document is a complete JSON object from the Document Processor.
        *   📄 **Resume Data (Metadata):** Stored in **PostgreSQL** in the `resumes` table. This includes `id` (PK), `user_id` (FK to `users`), `mongo_doc_id` (the ObjectID string linking to the full resume in MongoDB), `predicted_role`, `role_confidence`, `title`, `original_file_path`, `created_at`.
        *   💼 **Job Data (Primary Structured Store):** Stored in **PostgreSQL** in the `jobs` table. This includes `id` (PK), `title`, `company`, `location`, `description_snippet`, `min_experience_years`, `employment_type`, `experience_level`, `target_roles`, `predicted_role` (if job description is classified), and potentially a `mongo_doc_id` if full parsed job JSONs are also stored in a `parsed_jobs` MongoDB collection (though this isn't as strongly emphasized for jobs as it is for resumes). Job skills are in a `skills` table and linked via `job_skills` join table.
        *   ✨ **Match Data:** Stored in **PostgreSQL** in the `matches` table. This includes `id` (PK), `resume_id` (FK), `job_id` (FK), and the calculated `score`.

*   **"How does the backend handle errors, for example, if a downstream microservice like the Document Processor fails?"**
    *   **Answer:** "The backend uses several error handling strategies:
        1.  **FastAPI `HTTPException`:** For known error conditions within an endpoint (e.g., resource not found, unauthorized access, invalid input not caught by Pydantic), we explicitly `raise HTTPException(status_code=..., detail=...)`.
        2.  **Pydantic Validation Errors:** If an incoming request body doesn't match the defined Pydantic schema in an endpoint's type hints, FastAPI automatically returns a 422 Unprocessable Entity error with detailed information about which fields are problematic.
        3.  **Downstream Microservice Errors:** When the backend (using `httpx.AsyncClient` from `app/core/http_client.py`) calls another service like the Document Processor:
            *   Network errors (e.g., connection refused if DP is down) are caught as `httpx.RequestError`, and the endpoint would typically raise an `HTTPException` with status 503 (Service Unavailable).
            *   Timeout errors are caught as `httpx.TimeoutException`, raising a 504 (Gateway Timeout).
            *   If the downstream service returns an HTTP error status code (e.g., DP returns 500), the backend checks `response.status_code`. If it's not 2xx, it logs the error (including details from the downstream service's response if possible) and raises an `HTTPException`, often a 502 (Bad Gateway), to the frontend. This is visible in `app/api/v1/endpoints/resumes.py` when calling the Document Processor.
        4.  **Database Errors:** SQLAlchemy operations in `try...except SQLAlchemyError as e:` blocks (or more specific exceptions like `IntegrityError` for unique constraint violations). If an error occurs, `db.rollback()` is called to prevent inconsistent state, the error is logged, and an `HTTPException` (e.g., 500 Internal Server Error or 409 Conflict) is raised.
        5.  **Logging:** All significant errors, including tracebacks for unexpected exceptions, are logged using Python's `logging` module, providing detailed context for debugging."

*   **"Explain the JWT authentication flow at a basic level: token creation, storage, and validation."**
    *   **Answer:** "The backend uses JWTs for stateless authentication, managed by `app/core/security.py`:
        1.  **🪙 Token Creation (`create_access_token`):** After a user successfully logs in (credentials verified against the `users` table in PostgreSQL), `create_access_token` is called. It takes user data (like email for the `sub` claim, and potentially name for display) and an expiry time. This payload is then digitally signed using a `SECRET_KEY` (from `app/core/config.py`) and an algorithm like `HS256` (also from config) via `python-jose`. This creates the JWT string.
        2.  **📲 Token Storage (Frontend):** This JWT is sent back to the frontend in the login response. The frontend typically stores this token in `localStorage` or `sessionStorage` in the browser.
        3.  **✅ Token Validation (FastAPI Dependency `get_current_active_user`):** For protected API endpoints in the backend, we use `Depends(get_current_active_user)`.
            *   When a request comes to such an endpoint, FastAPI (via `OAuth2PasswordBearer`) automatically extracts the JWT from the `Authorization: Bearer <token>` header.
            *   The `get_current_user` function (which `get_current_active_user` calls) then uses `jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])` to verify the token's signature and check if it has expired.
            *   If the token is valid, it extracts the `sub` claim (user's email).
            *   It then queries the PostgreSQL `users` table using this email to fetch the `User` object.
            *   If the token is invalid, expired, or the user doesn't exist, an `HTTPException` (401 Unauthorized) is raised, and access to the endpoint is denied. Otherwise, the authenticated `User` object is made available to the endpoint function."

---

### ❓ Advanced Viva Questions Eva Must Be Ready For (Module 4 - Extended & Refined)

*(Includes previous questions, plus new ones focusing on deeper backend concepts and dissertation linkage)*

1.  **"In `backend/app/main.py`, you have a `lifespan` manager. What specific resources are initialized at startup and why is it important to manage them this way?"**
    *   **Answer:** "The `lifespan` manager in `app/main.py` is critical for managing resources that should persist for the entire lifetime of the application, rather than being created and destroyed on every request. At **startup**, it initializes:
        1.  **Shared `httpx.AsyncClient` (`start_async_client`):** This creates a single, reusable asynchronous HTTP client instance. This is important because `httpx.AsyncClient` manages a connection pool. Reusing the client allows for efficient reuse of HTTP connections to our downstream microservices (Document Processor, NLP, Matcher), reducing the overhead and latency of establishing new connections for every inter-service call.
        2.  **Shared `motor.AsyncIOMotorClient` (`start_mongo_client`):** This initializes the connection pool to our MongoDB instance. Similar to the HTTP client, reusing this connection pool is much more efficient than creating new MongoDB connections for every request that needs to access resume JSONs.
        At **shutdown**, the `lifespan` manager calls `stop_async_client` and `stop_mongo_client` to gracefully close these connections and release any associated resources. This prevents resource leaks and ensures a clean shutdown."

2.  **"The `resumes.py` endpoint interacts with both MongoDB and PostgreSQL. Describe a potential data consistency issue that could arise if, for example, saving to PostgreSQL fails *after* successfully saving to MongoDB, and how you might handle it."**
    *   **Answer:** "This is a classic distributed transaction problem. If we save the full resume JSON to MongoDB successfully, but then the subsequent save of metadata (including the `mongo_doc_id`) to PostgreSQL fails (e.g., due to a DB connection issue or a constraint violation):
        *   **Issue:** We would have an "orphan" document in MongoDB – a parsed resume JSON that has no corresponding metadata record in PostgreSQL. This resume would be effectively invisible to the rest of the system that queries via PostgreSQL.
        *   **Handling (Current Implementation):** The current `upload_resume` function in `resumes.py` has separate `try...except` blocks for Mongo and SQL operations. If the SQL save fails after Mongo success, it calls `db.rollback()` for the SQL transaction. It *could* then attempt to delete the just-inserted MongoDB document to revert the state, like `await mongo_db[MONGO_COLLECTION_NAME].delete_one({"_id": inserted_mongo_id})`. This would be a manual "compensating transaction."
        *   **More Robust Solutions (Advanced):**
            1.  **Two-Phase Commit (Complex):** Implementing a full two-phase commit protocol across MongoDB and PostgreSQL is very complex and often not practical without specialized transaction managers.
            2.  **Eventual Consistency with a Retry/Reconciliation Queue:** A more common pattern is to aim for eventual consistency. If the PostgreSQL save fails, the `mongo_doc_id` and necessary metadata could be put onto a retry queue (e.g., using Redis or a simple DB table). A background worker would periodically try to process this queue and create the PostgreSQL record.
            3.  **Saga Pattern:** Break the operation into a series of local transactions with corresponding compensating transactions for rollback if a step fails.
        For this project, the compensating transaction (deleting from Mongo if SQL fails) is the most straightforward approach if atomicity is desired, otherwise accepting eventual consistency with retries is also an option."

3.  **"Explain the SQLAlchemy ORM models in `app/models/models.py`. How do they define relationships, like the many-to-many relationship between Resumes/Jobs and Skills?"**
    *   **Answer:** "The `app/models/models.py` file defines our PostgreSQL database schema using SQLAlchemy's Object Relational Mapper.
        *   Each class (e.g., `User`, `Resume`, `Job`, `Skill`) inherits from `Base = declarative_base()` and maps to a database table.
        *   Columns are defined as class attributes using `Column(DataType, ...)`, specifying types (e.g., `Integer`, `String`, `DateTime`, `JSON`), primary keys (`primary_key=True`), foreign keys (`ForeignKey('users.id')`), indexes (`index=True`), and constraints (e.g., `unique=True`, `nullable=False`).
        *   **Relationships:** SQLAlchemy's `relationship()` function defines how tables are linked:
            *   **One-to-Many:** For example, in the `User` model, `resumes = relationship("Resume", back_populates="user")` defines that a user can have many resumes, and in the `Resume` model, `user = relationship("User", back_populates="resumes")` defines the other side of this link. The `ForeignKey` is in the `Resume` model (`user_id = Column(Integer, ForeignKey('users.id'))`).
            *   **Many-to-Many (M2M):** For Resumes/Jobs and Skills, we need M2M relationships. This is achieved with **association tables**.
                *   `resume_skills_table = Table('resume_skills', Base.metadata, Column('resume_id', ForeignKey('resumes.id'), primary_key=True), Column('skill_id', ForeignKey('skills.id'), primary_key=True))` defines the join table.
                *   In the `Resume` model: `skills = relationship("Skill", secondary=resume_skills_table, back_populates="resumes")`.
                *   In the `Skill` model: `resumes = relationship("Resume", secondary=resume_skills_table, back_populates="skills")`.
                *   A similar setup (`job_skills_table`) exists for Jobs and Skills.
        This ORM approach allows us to work with database records as Python objects and query relationships easily (e.g., `my_resume.skills` would give a list of `Skill` objects associated with that resume)."

4.  **"The `backend/app/utils/crud.py` file has `get_or_create_skills`. Why is this utility useful when saving, for example, a parsed job description?"**
    *   **Answer:** "When we parse a job description (or a resume), it will contain a list of skill names (strings). Before we can link these skills to the job (or resume) record in PostgreSQL via the many-to-many association table (e.g., `job_skills`), we need to ensure that each unique skill exists as a record in the `skills` table.
        The `get_or_create_skills(db, skill_names_list)` function handles this efficiently:
        1.  It takes a list of skill name strings.
        2.  It normalizes them (e.g., converts to title case, strips whitespace).
        3.  It queries the `skills` table to see which of these normalized skills already exist.
        4.  For any skill names in the input list that *don't* yet exist in the `skills` table, it creates new `models.Skill` records and adds them to the database session.
        5.  It then returns a list of all relevant `Skill` ORM objects (both pre-existing and newly created).
        This list of `Skill` objects can then be directly assigned to the `job.skills` relationship attribute before committing the job record, and SQLAlchemy will handle creating the necessary entries in the `job_skills` join table. This utility prevents duplicate skill entries in the `skills` table and simplifies the logic in the endpoint handlers."

5.  **"If you needed to add a new field, say `industry`, to the `jobs` table, what steps would you take involving `models.py` and `alembic`?"**
    *   **Answer:**
        1.  **Modify Model (`app/models/models.py`):** I would add the new field to the `Job` SQLAlchemy class:
            ```python
            class Job(Base):
                # ... other columns ...
                industry = Column(String(100), nullable=True, index=True) # New field
            ```
        2.  **Generate Alembic Migration Script:** In the terminal, within the `backend/` directory (where `alembic.ini` is), I would run:
            `alembic revision -m "add_industry_to_jobs_table"`
            This command tells Alembic to compare the current state of my SQLAlchemy models with the last applied migration and generate a new revision script in `alembic/versions/`. This script will contain `op.add_column('jobs', sa.Column('industry', sa.String(100), ...))` and a corresponding `op.drop_column` in the `downgrade` function.
        3.  **Review Migration Script:** I would open the newly generated Python file in `alembic/versions/` to review the generated operations and ensure they are correct.
        4.  **Apply Migration to Database:** I would run:
            `alembic upgrade head`
            This command executes the new migration script, applying the `ALTER TABLE jobs ADD COLUMN industry ...` SQL command to the PostgreSQL database.
        5.  **Update Pydantic Schemas (`app/schemas/schemas.py` - Optional but good practice):** If I want this new `industry` field to be part of API requests or responses for jobs, I would add it to the relevant Pydantic models (e.g., `JobBase`, `JobCreate`, `Job`).
        6.  **Update Application Logic:** Update any code that creates or queries jobs to utilize the new `industry` field if necessary.

*(Eva should also be prepared for questions 1, 5, 6, 7 from the previous Module 4 Advanced Viva Questions, as they cover `lifespan`, Pydantic, `get_db` dependency, global `AsyncClient`, and Alembic fundamentals.)*

---

### ✅ Final Review Checklist for Eva (Module 4 - Exhaustive & Consistent)

*   [ ] **Master `app/main.py`**: Explain FastAPI app creation, the exact purpose and flow of the `lifespan` manager (initializing `httpx.AsyncClient` and `motor.AsyncIOMotorClient` on startup, closing on shutdown), CORS middleware setup, and how `APIRouter` instances from `app/api/v1/endpoints/` are included with version prefixing.
*   [ ] **Deep Dive into Pydantic Schemas (`app/schemas/schemas.py`)**:
    *   Select `UserCreate`, `Resume` (the response model), `Job` (response model), and `MatchResponse`.
    *   For each, explain its fields, data types, any validation constraints (e.g., `EmailStr`, `Optional`, default values, `min_length`).
    *   Explain how FastAPI uses these for automatic request body validation and response serialization.
*   [ ] **Deep Dive into SQLAlchemy Models (`app/models/models.py`)**:
    *   Select `User`, `Resume`, `Job`, and `Skill` tables, plus one M2M association table like `resume_skills_table`.
    *   For each, explain the table name, columns, SQLAlchemy data types (`Integer`, `String`, `DateTime`, `ForeignKey`, `JSON`), primary keys, foreign key relationships, and how `relationship()` (with `back_populates` and `secondary`) defines one-to-many and many-to-many links.
*   [ ] **Exhaustive Trace of Resume Upload (`endpoints/resumes.py` -> `upload_resume`):**
    *   **Authentication:** How `Depends(get_current_active_user)` from `core/security.py` works (JWT extraction, decoding, DB lookup).
    *   **File Handling:** Reception of `UploadFile`, temporary storage.
    *   **Call to Document Processor:** Use of shared `httpx.AsyncClient` (from `core/http_client.py`), forming the multipart request, target URL from `settings` (in `core/config.py`).
    *   **Handling DP Response:** Error checking (status codes, JSON decode), data transformation (`core/data_transformer.py`), `jsonschema` validation.
    *   **MongoDB Save:** Use of shared `motor` client (from `core/mongo.py`), `insert_one` into `parsed_resumes`.
    *   **Call to NLP Service:** Use of `services/nlp_client.py -> call_nlp_classify`.
    *   **PostgreSQL Save:** Creation of `models.Resume` instance, population with metadata and NLP results, `db.add()`, `db.commit()`.
    *   **API Response:** Structure of `ResumeCreateResponse`.
*   [ ] **Exhaustive Trace of Match Request (`endpoints/matching.py` -> `get_job_matches_for_resume`):**
    *   Fetching target resume (Postgres metadata + Mongo full JSON).
    *   Fetching candidate jobs (Postgres `jobs` and `skills` tables).
    *   Preparing payload for Matching Service.
    *   Calling Matching Service via `httpx.AsyncClient`.
    *   Handling response and returning `MatchResponse`.
*   [ ] **JWT Authentication Flow Deep Dive (`core/security.py`):**
    *   `create_access_token()`: What goes into the payload (`sub`, `exp`, user name/email), how it's signed.
    *   `get_current_active_user()` & `get_current_user()`: How `OAuth2PasswordBearer` extracts token, `jwt.decode` process, fetching user from DB via `user_repo`.
*   **Data Storage Rationale & Interaction:**
    *   Clearly articulate *why* MongoDB for full, flexible resume JSONs and *why* PostgreSQL for structured metadata and relations.
    *   Explain how `mongo_doc_id` in PostgreSQL `resumes` table links the two.
    *   How `get_db()` (`db/session.py`) provides SQLA sessions and `get_mongo_db()` (`core/mongo.py`) provides Motor DB handle.
*   **Error Handling Nuances:** Discuss specific `HTTPException` status codes used for different error types (401, 404, 409, 422, 500, 502, 503, 504) and where they are typically raised. Explain `db.rollback()` on `SQLAlchemyError`.

