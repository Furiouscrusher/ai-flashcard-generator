\# Contributing to AI Flashcard Generator Using RAG



\## 1. Project Overview



This project aims to generate study flashcards from uploaded educational documents using Retrieval-Augmented Generation (RAG).



The project is developed collaboratively. Each contributor should work on their assigned module and follow the agreed interfaces to make integration easier.



Repository: https://github.com/Furiouscrusher/ai-flashcard-generator



\## 2. Prerequisites



Install the following before starting:



\- Python 3.10 or later

\- Git

\- A GitHub account with repository access



Check your installations:



```powershell

python --version

git --version

```



\## 3. Set Up the Project



Clone the repository:



```powershell

git clone https://github.com/Furiouscrusher/ai-flashcard-generator.git

cd ai-flashcard-generator

```



Create a virtual environment:



```powershell

python -m venv .venv

```



Activate it in Windows PowerShell:



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



If PowerShell blocks activation, use the following for the current terminal session only:



```powershell

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

.\\.venv\\Scripts\\Activate.ps1

```



Install dependencies:



```powershell

python -m pip install -r requirements.txt

```



\## 4. Test the Existing Document Processor



Run the automated tests:



```powershell

python -m unittest discover -s tests -v

```



Run the PDF smoke test:



```powershell

python test\_processor.py

```



Run these checks after making changes to the document-processing module.



\## 5. Branching and Development Workflow



Do not commit directly to `main` for routine development.



Before starting a task, update your local main branch:



```powershell

git switch main

git pull origin main

```



Create a feature branch:



```powershell

git switch -c feature/your-module-name

```



Replace `your-module-name` with a short description of your assigned work.



After making changes, inspect and test them:



```powershell

git status

python -m unittest discover -s tests -v

git diff

```



Stage and commit your changes:



```powershell

git add .

git commit -m "Describe your change"

```



Push your branch:



```powershell

git push -u origin feature/your-module-name

```



Open a Pull Request on GitHub targeting `main`. Explain what changed and how it was tested. Wait for review before merging.



\## 6. Module Responsibilities



The team should agree on final assignments before implementation.



\- \*\*Document Processing:\*\* PDF/TXT extraction, text cleaning, chunking, metadata, and processing tests.

\- \*\*Retrieval and RAG:\*\* Embeddings, vector storage, and retrieving relevant document chunks.

\- \*\*Flashcard Generation:\*\* Generating question-and-answer cards from retrieved context and validating the output.

\- \*\*Interface and Integration:\*\* Connecting the modules, accepting uploads, displaying flashcards, and integration testing.



Actual implementation responsibilities should be recorded accurately. Integration and testing can be shared where necessary.



\## 7. Document Processor Interface



The current document-processing function is:



```python

process\_document(

&#x20;   file\_bytes: bytes,

&#x20;   filename: str,

&#x20;   chunk\_size: int = 1000,

&#x20;   overlap: int = 150,

) -> list\[dict]

```



It accepts PDF or TXT file bytes and returns a list of chunk dictionaries.



Each chunk contains:



\- `chunk\_id`: Identifier for the generated chunk.

\- `text`: Cleaned text for that chunk.

\- `metadata.document`: Original filename.

\- `metadata.document\_id`: Content-derived document identifier.

\- `metadata.page`: PDF page number, or `None` for TXT files.



Invalid or unreadable documents raise `DocumentProcessingError`. Invalid chunk settings raise `ValueError`.



Other modules should rely on this documented output structure rather than depending on internal implementation details.



\## 8. Coding Guidelines



\- Keep each module focused on its assigned responsibility.

\- Use meaningful function and variable names.

\- Add tests for new functionality and bug fixes.

\- Handle invalid input and failures clearly.

\- Avoid committing virtual environments, cache files, secrets, or personal configuration.

\- Update documentation when changing a module's public interface.

\- Discuss interface changes with affected contributors before merging.



\## 9. Git and Merge Safety



\- Do not force-push to shared branches.

\- Do not overwrite another contributor's work without coordination.

\- Resolve merge conflicts carefully and rerun tests afterward.

\- Keep commits focused and descriptive.

\- Never commit API keys, passwords, or `.env` files.



\## 10. Questions and Coordination



Discuss module assignments, interface changes, dependencies, and integration issues with the team before making changes that affect other modules.



